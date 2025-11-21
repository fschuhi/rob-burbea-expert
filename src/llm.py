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
            "You are an expert teaching assistant for Rob Burbea's dharma talks. "
            "You will be provided with excerpts from his talks. "
            "Your task is to answer the user's question based ONLY on these excerpts. "
            "\n\n"
            "STRICT GUIDELINES:\n"
            "1. **Persona**: Speak naturally and directly. Do NOT use phrases like 'Based on the context', "
            "'In the provided chunks', or 'The retrieval results show'. "
            "Instead, say 'Rob mentions...', 'The practice involves...', or 'In the talk [Name]...'.\n"
            "2. **Accuracy**: If the provided text does not contain the answer, explicitly say "
            "'I cannot find that information in the current reference material' rather than hallucinating.\n"
            "3. **Citations**: When referencing specific ideas, ALWAYS cite the source using the format "
            "**[Talk Name, Para X]**. Do not use paragraph numbers alone."
        )

        # 2. Assemble the Message History
        messages = [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': f"Reference Material:\n{context}\n\nQuestion:\n{query}"
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
