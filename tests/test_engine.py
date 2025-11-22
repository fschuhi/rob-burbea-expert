from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.engine import RAGEngine
from src.models import FakeEmbeddingFunction

# --- Mocks ---


@pytest.fixture
def mock_llm_client():
    """Mocks the OllamaClient to avoid real network calls."""
    with patch("src.engine.OllamaClient") as MockClientClass:
        mock_instance = MockClientClass.return_value

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
    - Mocked CrossEncoder (to avoid model download in unit tests)
    """
    with patch("src.engine.get_embedding_function") as mock_get_ef, patch(
        "src.engine.CrossEncoder"
    ) as MockCrossEncoder:

        # 1. Mock the Bi-Encoder
        mock_get_ef.return_value = FakeEmbeddingFunction()

        # 2. Mock the Cross-Encoder
        # We just need it to return a list of dummy scores (floats)
        mock_ce_instance = MockCrossEncoder.return_value
        # predict() receives a list of pairs, returns a list of scores
        mock_ce_instance.predict.side_effect = lambda pairs: [0.99] * len(pairs)

        engine = RAGEngine(test_env)
        engine.collection = populated_collection
        engine.context_builder.collection = populated_collection

        return engine


# --- Tests ---


def test_engine_initialization(engine, test_env):
    """Verifies components are set up correctly."""
    assert engine.env == test_env
    assert engine.collection is not None
    assert engine.context_builder is not None
    assert engine.llm_client is not None
    assert engine.cross_encoder is not None


def test_answer_query_flow(engine, mock_llm_client):
    """
    Tests the full pipeline:
    1. Search (using FakeEmbeddingFunction on populated_collection)
    2. Rerank (Mocked)
    3. Context Build (using real ContextBuilder logic)
    4. Generation (using Mock LLM)
    """
    query = "test query"

    # Execute
    # FIX: Unpack 3 values now (context, stream, references)
    context_str, stream, references = engine.answer_query(query)

    # 1. Verify Context was built from the DB
    # Updated to check for Reference ID format
    assert "### Reference [1]: test_talk.md" in context_str

    # Verify references map exists
    assert references is not None
    assert 1 in references

    # 2. Verify LLM was called
    mock_llm_client.stream_answer.assert_called_once()
    call_args = mock_llm_client.stream_answer.call_args
    assert call_args.kwargs["query"] == query
    assert call_args.kwargs["context"] == context_str

    # 3. Verify Stream Output
    full_response = "".join(list(stream))
    assert full_response == "This is a test."


def test_answer_query_with_overrides(engine, mock_llm_client):
    """
    Verifies that top_k and distance_threshold overrides are respected.
    """
    with patch.object(engine.collection, "query", wraps=engine.collection.query) as mock_query:
        # FIX: Unpack 3 values
        engine.answer_query("test", top_k=10, distance_threshold=0.9)

        mock_query.assert_called_once()

        # FIX: The engine now fetches 5x the top_k to create a pool for the reranker
        # So if top_k=10, we expect n_results=50
        assert mock_query.call_args.kwargs["n_results"] == 50
