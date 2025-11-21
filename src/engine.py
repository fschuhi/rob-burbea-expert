from __future__ import annotations

from typing import Iterator, Tuple

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
        # We need the embedding function to query the DB
        self.ef = get_embedding_function(env.models.embedding_model)
        self.connector = ChromaConnector(env)

        # We assume the collection name is standard for now
        self.collection = self.connector.get_collection(
            "rob_burbea_talks",
            embedding_function=self.ef
        )

        # 2. Setup Components
        self.context_builder = ContextBuilder(self.collection)
        self.llm_client = OllamaClient(env)

    def answer_query(self, query_text: str) -> Tuple[str, Iterator[str]]:
        """
        Performs a full RAG cycle:
        1. Searches the database for relevant chunks.
        2. Builds a formatted context string.
        3. Queries the LLM with the context and user question.

        Args:
            query_text: The user's question.

        Returns:
            A tuple containing:
            - context_str: The formatted markdown context sent to the LLM.
            - response_stream: An iterator yielding chunks of the LLM's answer.
        """

        # Step 1: Retrieval
        # We query for more candidates (top_k) to let the context builder filter/sort
        results = self.collection.query(
            query_texts=[query_text],
            n_results=self.env.rag.top_k_results
        )

        # Step 2: Context Construction
        context_str = self.context_builder.build(
            query_results=results,
            max_distance=self.env.rag.similarity_threshold,
            # We could make max_items configurable, but defaulting to top_k is safe
            max_items=self.env.rag.top_k_results
        )

        # Step 3: Generation
        # We return the stream so the UI can consume it
        response_stream = self.llm_client.stream_answer(
            query=query_text,
            context=context_str
        )

        return context_str, response_stream
