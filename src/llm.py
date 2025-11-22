from __future__ import annotations

import ollama
from typing import Iterator

from src.env import Env


class OllamaClient:
    """
    Client for interacting with a local Ollama instance for RAG question answering.
    """

    def __init__(self, env: Env):
        self.model_name = env.models.default_llm_model
        self.base_url = env.ollama.base_url
        self.timeout = env.ollama.timeout
        self.client = ollama.Client(host=self.base_url, timeout=self.timeout)

    def stream_answer(self, query: str, context: str) -> Iterator[str]:
        """
        Streams an answer from the LLM based on the user query and retrieved context.
        """

        # Simpler, more robust system prompt using numeric IDs
        system_prompt = (
            "You are an expert teaching assistant for Rob Burbea's dharma talks. "
            "Answer the user's question using ONLY the provided Reference Material. "
            "\n\n"
            "### INSTRUCTIONS ###\n"
            "1. **Citations**: Every claim must be supported by a citation.\n"
            "2. **Format**: Use the Reference ID provided in the header, e.g., **[1]** or **[2]**.\n"
            "   - Do NOT include filenames or paragraph numbers in the answer.\n"
            "   - Just use the bracketed number.\n"
            "3. **Persona**: Speak naturally."
            "\n\n"
            "### EXAMPLE ###\n"
            "Context: \n"
            "### Reference [1]: talk.md\n"
            "Breathing is good.\n"
            "\n"
            "Assistant: You should practice breathing **[1]**."
        )

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

        stream = self.client.chat(
            model=self.model_name,
            messages=messages,
            stream=True,
        )

        for chunk in stream:
            content = chunk.get('message', {}).get('content', '')
            if content:
                yield content
