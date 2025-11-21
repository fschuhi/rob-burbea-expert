from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from src.llm import OllamaClient
from src.env import Env, Models, Ollama


@pytest.fixture
def mock_env():
    """Creates a dummy Env for testing."""
    env = MagicMock(spec=Env)
    env.models = Models(default_llm_model="test-model")
    env.ollama = Ollama(base_url="http://fake-host:11434", timeout=30)
    return env


@patch('src.llm.ollama.Client')
def test_client_initialization(mock_client_cls, mock_env):
    """Verifies the client is initialized with config values."""
    client = OllamaClient(mock_env)

    # Check that the library's Client was instantiated with our config
    mock_client_cls.assert_called_once_with(
        host="http://fake-host:11434",
        timeout=30
    )
    assert client.model_name == "test-model"


@patch('src.llm.ollama.Client')
def test_stream_answer_yields_chunks(mock_client_cls, mock_env):
    """Verifies that the method correctly yields chunks from the stream."""

    # 1. Setup the mock stream
    # The real Ollama client returns an iterator of dicts
    mock_stream_response = [
        {'message': {'content': 'Hello'}},
        {'message': {'content': ' '}},
        {'message': {'content': 'World'}},
        {'message': {'content': '!'}},
        # Sometimes chunks might be empty or control messages
        {'done': False}
    ]

    # Configure the mock instance returned by the class
    mock_instance = mock_client_cls.return_value
    mock_instance.chat.return_value = mock_stream_response

    # 2. Run the client
    client = OllamaClient(mock_env)
    generator = client.stream_answer(query="Hi", context="Context info")

    # 3. Collect results
    result_text = "".join(list(generator))

    # 4. Assertions
    assert result_text == "Hello World!"

    # Verify the chat method was called with correct structure
    mock_instance.chat.assert_called_once()
    call_kwargs = mock_instance.chat.call_args[1]

    assert call_kwargs['model'] == "test-model"
    assert call_kwargs['stream'] is True
    assert len(call_kwargs['messages']) == 2
    assert call_kwargs['messages'][0]['role'] == 'system'
    assert call_kwargs['messages'][1]['role'] == 'user'
