from __future__ import annotations

from typing import Iterator, Tuple, Optional, Dict, Any, List

from sentence_transformers import CrossEncoder
from chromadb.api.models.Collection import Collection

from src.env import Env
from src.database import ChromaConnector
from src.models import get_embedding_function
from src.context import ContextBuilder
from src.llm import OllamaClient


class RetrievalPipeline:
    """
    Staged retrieval workflow with dependency injection.

    Handles the core retrieval operations:
    1. Vector search (retrieve)
    2. Distance filtering (filter_by_distance)
    3. Scoring/sorting (score_candidates)

    Dependencies (collection, cross_encoder) are injected at construction,
    enabling easy testing and future flexibility (e.g., swapping rerankers).
    """

    def __init__(self, collection: Collection, cross_encoder: CrossEncoder):
        self.collection = collection
        self.cross_encoder = cross_encoder

    def retrieve(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """
        Stage 1: Vector search.

        Args:
            query: The search query text
            n_results: Number of candidates to retrieve

        Returns:
            List of candidate dicts with keys: id, text, metadata, initial_dist
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        return self._unpack_chroma_results(results)

    def filter_by_distance(self, candidates: List[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
        """
        Stage 2: Distance filtering.

        Args:
            candidates: List of candidate dicts from retrieve()
            threshold: Maximum distance (lower is more similar)

        Returns:
            Filtered list of candidates within threshold
        """
        return [c for c in candidates if c["initial_dist"] <= threshold]

    def score_candidates(
        self, query: str, candidates: List[Dict[str, Any]], use_reranker: bool
    ) -> List[Dict[str, Any]]:
        """
        Stage 3: Score and sort candidates.

        Args:
            query: The original query (needed for cross-encoder)
            candidates: List of candidate dicts
            use_reranker: If True, use cross-encoder. If False, sort by distance.

        Returns:
            Sorted list of candidates (best first)
        """
        if not candidates:
            return candidates

        if use_reranker:
            # Cross-Encoder scoring (higher = more relevant)
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.cross_encoder.predict(pairs)
            for i, candidate in enumerate(candidates):
                candidate["score"] = scores[i]
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            # Vector distance sorting (lower = more relevant)
            candidates.sort(key=lambda x: x["initial_dist"])

        return candidates

    def _unpack_chroma_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert Chroma query results to list of candidate dicts.

        Chroma returns nested lists (for batch queries). We assume single query.
        """
        if not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        candidates = []
        for i in range(len(ids)):
            candidates.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "initial_dist": dists[i],
                }
            )
        return candidates


class RAGEngine:
    """
    Orchestrator for the Retrieval Augmented Generation pipeline.
    Implements a two-step retrieval process:
    1. Bi-Encoder Retrieval (Fast, High Recall)
    2. Cross-Encoder Reranking (Slow, High Precision)
    """

    def __init__(self, env: Env):
        self.env = env

        # 1. Bi-Encoder (Fast Retrieval)
        self.ef = get_embedding_function(env.models.embedding_model)
        self.connector = ChromaConnector(env)
        self.collection = self.connector.get_collection("rob_burbea_talks", embedding_function=self.ef)

        # 2. Cross-Encoder (Precision Reranking)
        print(f"Loading Reranker: {env.models.reranker_model}...")
        self.cross_encoder = CrossEncoder(env.models.reranker_model)

        # 3. Pipeline (uses injected dependencies)
        self.pipeline = RetrievalPipeline(self.collection, self.cross_encoder)

        self.context_builder = ContextBuilder(self.collection)
        self.llm_client = OllamaClient(env)

    def retrieve_and_rerank(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        retrieval_pool_size: Optional[int] = None,
        apply_reranker: bool = True,
    ) -> Tuple[str, Dict[int, Any]]:
        """
        Performs the retrieval pipeline.

        Args:
            query_text: The user's question.
            top_k: Number of final chunks to return.
            distance_threshold: Cutoff for vector similarity (lower is better).
            retrieval_pool_size: How many candidates to fetch from DB (if reranking).
            apply_reranker: If True, fetch pool_size & rerank. If False, fetch top_k & sort by distance.

        Returns:
            - context_str: The formatted markdown context.
            - references: The ID map for the UI.
        """
        # Parameter defaults
        final_top_k = top_k if top_k is not None else self.env.rag.top_k_results
        final_threshold = distance_threshold if distance_threshold is not None else self.env.rag.similarity_threshold
        pool_size = retrieval_pool_size if retrieval_pool_size is not None else self.env.rag.retrieval_pool_size

        # Retrieval pool size depends on whether we're reranking
        initial_k = pool_size if apply_reranker else final_top_k

        # Pipeline stages
        candidates = self.pipeline.retrieve(query_text, initial_k)
        if not candidates:
            return "No relevant context found.", {}

        candidates = self.pipeline.filter_by_distance(candidates, final_threshold)
        if not candidates:
            return "No relevant context found.", {}

        candidates = self.pipeline.score_candidates(query_text, candidates, apply_reranker)
        top_hits = candidates[:final_top_k]

        # Build context
        return self.context_builder.build_from_hits(top_hits)

    def answer_query(
        self, query_text: str, top_k: Optional[int] = None, distance_threshold: Optional[float] = None
    ) -> Tuple[str, Iterator[str], Dict[int, Any]]:
        """
        Legacy wrapper for the full pipeline.
        """
        context_str, references = self.retrieve_and_rerank(query_text, top_k, distance_threshold)

        response_stream = self.llm_client.stream_answer(query=query_text, context=context_str)

        return context_str, response_stream, references
