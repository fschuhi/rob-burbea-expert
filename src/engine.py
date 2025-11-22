from __future__ import annotations

from typing import Iterator, Tuple, Optional, Dict, Any

from src.env import Env
from src.database import ChromaConnector
from src.models import get_embedding_function
from src.context import ContextBuilder
from src.llm import OllamaClient


class RAGEngine:
    """
    Orchestrator for the Retrieval Augmented Generation pipeline.
    """

    def __init__(self, env: Env):
        self.env = env

        self.ef = get_embedding_function(env.models.embedding_model)
        self.connector = ChromaConnector(env)
        self.collection = self.connector.get_collection(
            "rob_burbea_talks",
            embedding_function=self.ef
        )

        self.context_builder = ContextBuilder(self.collection)
        self.llm_client = OllamaClient(env)

    def answer_query(
            self,
            query_text: str,
            top_k: Optional[int] = None,
            distance_threshold: Optional[float] = None
    ) -> Tuple[str, Iterator[str], Dict[int, Any]]:
        """
        Returns:
            - context_str: The text sent to LLM.
            - response_stream: Generator for answer.
            - references: Dict mapping [1] -> Metadata for UI lookup.
        """

        final_top_k = top_k if top_k is not None else self.env.rag.top_k_results
        final_threshold = distance_threshold if distance_threshold is not None else self.env.rag.similarity_threshold

        results = self.collection.query(
            query_texts=[query_text],
            n_results=final_top_k
        )

        # Capture the references map here
        context_str, references = self.context_builder.build(
            query_results=results,
            max_distance=final_threshold,
            max_items=final_top_k
        )

        response_stream = self.llm_client.stream_answer(
            query=query_text,
            context=context_str
        )

        return context_str, response_stream, references
