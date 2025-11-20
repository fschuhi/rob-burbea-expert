from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
import pytest

from src.env import Env, Paths, RAG, IO, Models
from src.indexing import run_indexer
from src.database import ChromaConnector

# Import the MockEmbeddingFunction we defined in test_database
from chromadb import EmbeddingFunction, Documents


class MockEmbeddingFunction(EmbeddingFunction):
    # noinspection PyMissingConstructor
    def __init__(self, *args, **kwargs):  # type: ignore
        """Accepts any arguments but ignores them."""
        pass

    def __call__(self, docs: Documents):
        # Return list of lists (standard Chroma format)
        return [[0.1, 0.2, 0.3] for _ in docs]


@pytest.fixture
def index_test_env(tmp_path: Path) -> Env:
    """Creates a full environment with dummy talks for indexing."""
    talks_dir = tmp_path / "talks"
    talks_dir.mkdir(parents=True, exist_ok=True)

    # Create 2 dummy talks.
    # The splitter splits on \n\n, so each file produces 2 chunks (Header + Body).
    # Total chunks = 4.
    (talks_dir / "talk_a.md").write_text("# Talk A\n\nContent of talk A.", encoding="utf-8")
    (talks_dir / "talk_b.md").write_text("# Talk B\n\nContent of talk B.", encoding="utf-8")

    (tmp_path / "metadata.json").touch()

    return Env(
        paths=Paths(
            data_dir=tmp_path / "data",
            raw_talks_dir=talks_dir,
            chroma_db_dir=tmp_path / "chroma_db",
            metadata_path=tmp_path / "metadata.json"
        ),
        rag=RAG(chunk_size=500),
        io=IO(create_missing_dirs=True),
        # FIX: Use the explicit Models class to satisfy PyCharm type checker
        models=Models(embedding_model="mock-model")
    )


def test_run_indexer_end_to_end(index_test_env: Env):
    """
    Runs the full indexing pipeline using a mock embedding function.
    """
    with patch("src.indexing.get_embedding_function") as mock_get_ef:
        mock_get_ef.return_value = MockEmbeddingFunction()

        # Run the indexer
        run_indexer(index_test_env)

        # Verify results in the DB
        connector = ChromaConnector(index_test_env)
        collection = connector.get_collection("rob_burbea_talks", embedding_function=MockEmbeddingFunction())

        # We expect 4 chunks (2 headers + 2 bodies)
        assert collection.count() == 4

        # Verify IDs are deterministic (filename_index)
        result = collection.get()
        ids = result["ids"]
        assert "talk_a_0" in ids
        assert "talk_a_1" in ids
        assert "talk_b_0" in ids


def test_run_indexer_id_stability(index_test_env: Env):
    """
    Verifies that running the indexer twice doesn't duplicate data (Upsert check).
    """
    with patch("src.indexing.get_embedding_function") as mock_get_ef:
        mock_get_ef.return_value = MockEmbeddingFunction()

        # Run once
        run_indexer(index_test_env)
        # Run again
        run_indexer(index_test_env)

        connector = ChromaConnector(index_test_env)
        collection = connector.get_collection("rob_burbea_talks", embedding_function=MockEmbeddingFunction())

        # Count should still be 4, not 8
        assert collection.count() == 4
