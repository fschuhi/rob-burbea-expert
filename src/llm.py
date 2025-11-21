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
        # FIX: The example now uses a full, realistic filename to prevent
        # the LLM from hallucinating short names like "talk.md".
        system_prompt = (
            "You are an expert teaching assistant for Rob Burbea's dharma talks. "
            "Answer the user's question using ONLY the provided Reference Material. "
            "\n\n"
            "### CRITICAL INSTRUCTIONS ###\n"
            "1. **Citations are MANDATORY**: Every single claim you make must be immediately followed by a citation.\n"
            "2. **Citation Format**: Use the exact format **[Filename, Para X]**. \n"
            "   - COPY the filename EXACTLY as it appears in the '### Source:' header.\n"
            "   - Do NOT shorten the filename. Do NOT use 'talk.md'.\n"
            "3. **Persona**: Speak naturally, but keep the citations technical.\n"
            "\n"
            "### EXAMPLE OF CORRECT RESPONSE ###\n"
            "User: How do I work with the breath?\n"
            "Assistant: You can play with the texture of the breath to soothe the energy body "
            "**[2019-12-18-the-energy-body-and-the-whole-body-breath.md, Para 12]**. "
            "Rob suggests imagining the breath flowing through constrictions "
            "**[2019-12-21-developing-piti-developing-focus.md, Para 4]**."
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
