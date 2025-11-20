from __future__ import annotations

from pathlib import Path
import pytest

from src.env import Env, Paths, RAG, IO, Models
from src.indexing import run_indexer
from src.database import ChromaConnector
from src.models import FakeEmbeddingFunction


@pytest.fixture
def index_test_env() -> Env:
    """Creates an Env configured with tmp/chroma_db_indexing and real fixture data."""

    # Use project root's tmp directory for ChromaDB (separate subdirectory for indexing tests)
    project_root = Path(__file__).parent.parent
    chroma_dir = project_root / "tmp" / "chroma_db_indexing"

    # Point to existing fixture data
    fixtures_dir = project_root / "tests" / "fixtures" / "data"

    return Env(
        paths=Paths(
            data_dir=fixtures_dir,
            raw_talks_dir=fixtures_dir / "raw_talks",
            chroma_db_dir=chroma_dir,
            metadata_path=fixtures_dir / "metadata.json"
        ),
        rag=RAG(chunk_size=500),
        io=IO(create_missing_dirs=True),
        # MAGIC: We just tell the env to use the "mock" model
        models=Models(embedding_model="mock")
    )


def test_run_indexer_end_to_end(index_test_env: Env):
    """Runs the full indexing pipeline using the Fake model defined in src/models."""

    # No patching needed! The env config drives the logic.
    run_indexer(index_test_env)

    connector = ChromaConnector(index_test_env)
    # We use the specific Fake class to read back
    collection = connector.get_collection("rob_burbea_talks", embedding_function=FakeEmbeddingFunction())

    # With 32 real talks, we expect many more chunks
    assert collection.count() > 0

    # Verify IDs follow expected pattern (filename_stem_index)
    ids = collection.get()["ids"]
    assert any("2019-12-17" in id for id in ids), "Should have chunks from real fixture talks"


def test_run_indexer_id_stability(index_test_env: Env):
    """Verifies that running the indexer twice produces the same IDs (no duplicates)."""
    run_indexer(index_test_env)
    first_count = ChromaConnector(index_test_env).get_collection(
        "rob_burbea_talks",
        embedding_function=FakeEmbeddingFunction()
    ).count()

    run_indexer(index_test_env)
    second_count = ChromaConnector(index_test_env).get_collection(
        "rob_burbea_talks",
        embedding_function=FakeEmbeddingFunction()
    ).count()

    assert first_count == second_count, "Upsert should maintain same count"
