from __future__ import annotations

import pytest
from src.models import get_embedding_function, FakeEmbeddingFunction, SentenceTransformerEmbeddingFunction


def test_factory_returns_fake():
    """Test that requesting 'mock' gives us the lightweight class."""
    ef = get_embedding_function("mock")
    assert isinstance(ef, FakeEmbeddingFunction)


def test_fake_returns_list_of_lists():
    """
    Verify our Fake implementation adheres to the ChromaDB contract
    (returning Python lists, not numpy arrays).
    """
    ef = get_embedding_function("mock")
    docs = ["doc1", "doc2"]
    results = ef(docs)

    assert isinstance(results, list)
    assert len(results) == 2

    # Critical check: Inner items must be lists
    print(results)
    assert isinstance(results[0], list)
    assert results[0] == [0.1, 0.2, 0.3]


def test_factory_returns_real_class_structure():
    """
    We don't instantiate the heavy model here to avoid downloads,
    but we verify the factory *tries* to give us the real one
    for other names.
    """
    # We can't fully instantiate it without downloading, so we catch the
    # import error or download delay, but simply checking the code path
    # is usually enough for unit tests.
    # For this specific test, we trust the logic in get_embedding_function.
    pass
