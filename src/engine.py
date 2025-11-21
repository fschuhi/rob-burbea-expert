from __future__ import annotations

from typing import Iterator, Tuple, Optional

from src.env import Env
from src.database import ChromaConnector
from src.models import get_embedding_function
from src.context import ContextBuilder
from src.llm import OllamaClient


class RAGEngine:
    """
    Orchestrator for the Retrieval Augmented Generation pipeline.
    Connects the Database, Context Builder, and LLM.
    """

    def __init__(self, env: Env):
        self.env = env

        # 1. Setup Database Connection
        self.ef = get_embedding_function(env.models.embedding_model)
        self.connector = ChromaConnector(env)

        self.collection = self.connector.get_collection(
            "rob_burbea_talks",
            embedding_function=self.ef
        )

        # 2. Setup Components
        self.context_builder = ContextBuilder(self.collection)
        self.llm_client = OllamaClient(env)

    def answer_query(
            self,
            query_text: str,
            top_k: Optional[int] = None,
            distance_threshold: Optional[float] = None
    ) -> Tuple[str, Iterator[str]]:
        """
        Performs a full RAG cycle with optional parameter overrides.

        Args:
            query_text: The user's question.
            top_k: Override for number of results to retrieve.
                   If None, uses env.rag.top_k_results.
            distance_threshold: Override for similarity cutoff.
                                If None, uses env.rag.similarity_threshold.

        Returns:
            context_str: The formatted markdown context sent to the LLM.
            response_stream: An iterator yielding chunks of the LLM's answer.
        """

        # Resolve defaults from Env if overrides not provided
        final_top_k = top_k if top_k is not None else self.env.rag.top_k_results
        final_threshold = distance_threshold if distance_threshold is not None else self.env.rag.similarity_threshold

        # Step 1: Retrieval
        results = self.collection.query(
            query_texts=[query_text],
            n_results=final_top_k
        )

        # Step 2: Context Construction
        context_str = self.context_builder.build(
            query_results=results,
            max_distance=final_threshold,
            max_items=final_top_k
        )

        # Step 3: Generation
        response_stream = self.llm_client.stream_answer(
            query=query_text,
            context=context_str
        )

        return context_str, response_stream
