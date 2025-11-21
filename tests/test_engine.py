from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.engine import RAGEngine
from src.models import FakeEmbeddingFunction


# --- Mocks ---

@pytest.fixture
def mock_llm_client():
    """Mocks the OllamaClient to avoid real network calls."""
    with patch('src.engine.OllamaClient') as MockClientClass:
        mock_instance = MockClientClass.return_value

        # Define a dummy stream generator
        def dummy_stream(query, context):
            yield "This "
            yield "is "
            yield "a "
            yield "test."

        mock_instance.stream_answer.side_effect = dummy_stream
        yield mock_instance


@pytest.fixture
def engine(test_env, populated_collection, mock_llm_client):
    """
    Creates an engine instance using:
    - Real Test Env
    - Real (Populated) Database Collection
    - Fake Embedding Function (injected)
    - Mocked LLM Client
    """
    # We patch get_embedding_function to return our FakeEF
    # because RAGEngine calls it in __init__
    with patch('src.engine.get_embedding_function') as mock_get_ef:
        mock_get_ef.return_value = FakeEmbeddingFunction()

        engine = RAGEngine(test_env)

        # Force the engine to use our populated collection fixture
        # (Instead of creating a new empty one from the connector)
        engine.collection = populated_collection

        # Update the context builder to use this collection too
        engine.context_builder.collection = populated_collection

        return engine


# --- Tests ---

def test_engine_initialization(engine, test_env):
    """Verifies components are set up correctly."""
    assert engine.env == test_env
    assert engine.collection is not None
    assert engine.context_builder is not None
    assert engine.llm_client is not None


def test_answer_query_flow(engine, mock_llm_client):
    """
    Tests the full pipeline:
    1. Search (using FakeEmbeddingFunction on populated_collection)
    2. Context Build (using real ContextBuilder logic)
    3. Generation (using Mock LLM)
    """
    query = "test query"

    # Execute
    context_str, stream = engine.answer_query(query)

    # 1. Verify Context was built from the DB
    # Since we use populated_collection, we expect "test_talk.md" in the header
    assert "### Source: test_talk.md" in context_str

    # 2. Verify LLM was called
    mock_llm_client.stream_answer.assert_called_once()
    call_args = mock_llm_client.stream_answer.call_args
    assert call_args.kwargs['query'] == query
    assert call_args.kwargs['context'] == context_str

    # 3. Verify Stream Output
    full_response = "".join(list(stream))
    assert full_response == "This is a test."


def test_answer_query_with_overrides(engine, mock_llm_client):
    """
    Verifies that top_k and distance_threshold overrides are respected.
    """
    # We need to spy on the collection.query method to check arguments
    with patch.object(engine.collection, 'query', wraps=engine.collection.query) as mock_query:
        engine.answer_query("test", top_k=10, distance_threshold=0.9)

        # Check retrieval override
        mock_query.assert_called_once()
        assert mock_query.call_args.kwargs['n_results'] == 10
