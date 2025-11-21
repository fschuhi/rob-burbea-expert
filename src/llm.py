from __future__ import annotations

import ollama
from typing import Iterator

from src.env import Env


class OllamaClient:
    """
    Client for interacting with a local Ollama instance for RAG question answering.
    """

    def __init__(self, env: Env):
        """
        Initialize with environment settings.

        Args:
            env: Configuration object containing model name and Ollama URL settings.
        """
        self.model_name = env.models.default_llm_model
        self.base_url = env.ollama.base_url
        self.timeout = env.ollama.timeout

        # We use the lower-level Client object to respect our custom base_url
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout)

    def stream_answer(self, query: str, context: str) -> Iterator[str]:
        """
        Streams an answer from the LLM based on the user query and retrieved context.

        Args:
            query: The user's question.
            context: The pre-formatted context string (from ContextBuilder).

        Yields:
            String chunks of the generated response.
        """

        # 1. Construct the System Prompt
        # We instruct the model to be a helpful assistant that strictly uses the provided context.
        system_prompt = (
            "You are an expert assistant for exploring Rob Burbea's dharma talks. "
            "You will be provided with a set of context chunks, each marked with <hit> tags. "
            "Your task is to answer the user's question based ONLY on these context chunks. "
            "If the provided context does not contain the answer, explicitly state that you don't know "
            "rather than making up information. "
            "Reference the specific chunks you used if possible."
        )

        # 2. Assemble the Message History
        # We construct the "chat" structure the model expects.
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f"Context:\n{context}\n\nQuestion:\n{query}"
            }
        ]

        # 3. Call the API with streaming enabled
        stream = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        # 4. Yield content chunks as they arrive
        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content
