from __future__ import annotations

from typing import Iterator, Tuple, Optional, Dict, Any

# NEW: Import CrossEncoder
from sentence_transformers import CrossEncoder

from src.env import Env
from src.database import ChromaConnector
from src.models import get_embedding_function
from src.context import ContextBuilder
from src.llm import OllamaClient


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
        # We load this once. It's small (~80MB) but adds significant precision.
        # It runs on CPU reasonably fast for small batches.
        print(f"Loading Reranker: {env.models.reranker_model}...")
        self.cross_encoder = CrossEncoder(env.models.reranker_model)

        self.context_builder = ContextBuilder(self.collection)
        self.llm_client = OllamaClient(env)

    def retrieve_and_rerank(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        retrieval_pool_size: Optional[int] = None,
    ) -> Tuple[str, Dict[int, Any]]:
        """
        Performs the full retrieval pipeline:
        1. Retrieve 'retrieval_pool_size' candidates from ChromaDB.
        2. Filter by initial distance threshold.
        3. Score (Query, Document) pairs using CrossEncoder.
        4. Sort by Score and take top_k.
        5. Reconstruct paragraphs for context.

        Returns:
            - context_str: The formatted markdown context.
            - references: The ID map for the UI.
        """
        final_top_k = top_k if top_k is not None else self.env.rag.top_k_results
        final_threshold = distance_threshold if distance_threshold is not None else self.env.rag.similarity_threshold

        # Default to env config if not overridden
        pool_size = retrieval_pool_size if retrieval_pool_size is not None else self.env.rag.retrieval_pool_size

        # --- Step 1: Broad Vector Search ---
        # Fetch the explicitly requested pool size
        results = self.collection.query(query_texts=[query_text], n_results=pool_size)

        # Unpack Chroma results (assuming single query)
        if not results["ids"]:
            return "No relevant context found.", {}

        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # --- Step 2: Initial Filtering ---
        # Filter by Bi-Encoder threshold first to save computation
        candidates = []
        for i in range(len(ids)):
            if dists[i] <= final_threshold:
                candidates.append({"id": ids[i], "text": docs[i], "metadata": metas[i], "initial_dist": dists[i]})

        if not candidates:
            return "No relevant context found.", {}

        # --- Step 3: Cross-Encoder Scoring ---
        # Prepare pairs for CrossEncoder: [[query, doc1], [query, doc2], ...]
        pairs = [[query_text, c["text"]] for c in candidates]

        # Predict returns a list of float scores (higher is better)
        # Note: These are logits (unbounded), not probabilities 0-1.
        scores = self.cross_encoder.predict(pairs)

        # Attach scores
        for i, candidate in enumerate(candidates):
            candidate["score"] = scores[i]

        # Sort by Cross-Encoder score (Higher = More Relevant)
        candidates.sort(key=lambda x: x["score"], reverse=True)

        # Slice the Top-K
        top_hits = candidates[:final_top_k]

        # --- Step 4: Build Context ---
        # We bypass the standard ContextBuilder.build because we have pre-sorted hits
        return self.context_builder.build_from_hits(top_hits)

    def answer_query(
        self, query_text: str, top_k: Optional[int] = None, distance_threshold: Optional[float] = None
    ) -> Tuple[str, Iterator[str], Dict[int, Any]]:
        """
        Legacy wrapper for the full pipeline.
        Useful for tests or simple CLI usage.
        """
        context_str, references = self.retrieve_and_rerank(query_text, top_k, distance_threshold)

        response_stream = self.llm_client.stream_answer(query=query_text, context=context_str)

        return context_str, response_stream, references
