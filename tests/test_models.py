from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import numpy as np

from src.models import SentenceTransformerEmbeddingFunction, get_embedding_function


# --- Fixtures & Mocks ---

@pytest.fixture
def mock_sentence_transformer():
    """
    Patches the SentenceTransformer class so we don't download
    actual models during testing.
    """
    with patch("src.models.SentenceTransformer") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        # Mock encode to return a numpy array
        mock_instance.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ])

        yield mock_class, mock_instance


# --- Tests ---

def test_init_loads_correct_model(mock_sentence_transformer):
    """Test that the class initializes the underlying model with the correct name."""
    mock_class, _ = mock_sentence_transformer

    model_name = "test-model-name"
    _ = SentenceTransformerEmbeddingFunction(model_name=model_name)

    mock_class.assert_called_once_with(model_name)


def test_call_returns_list(mock_sentence_transformer):
    """Test that the function converts numpy arrays to lists (required by Chroma)."""
    _, mock_instance = mock_sentence_transformer

    ef = SentenceTransformerEmbeddingFunction()
    docs = ["doc1", "doc2"]

    results = ef(docs)

    mock_instance.encode.assert_called_once_with(docs)

    # Check we got a list of lists
    assert isinstance(results, list)
    assert isinstance(results[0], list)
    assert results == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_factory_function(mock_sentence_transformer):
    """Test the helper factory function."""
    mock_class, _ = mock_sentence_transformer

    ef = get_embedding_function("factory-test-model")

    assert isinstance(ef, SentenceTransformerEmbeddingFunction)
    mock_class.assert_called_with("factory-test-model")
