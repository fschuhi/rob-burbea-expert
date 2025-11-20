from __future__ import annotations

from pathlib import Path
import pytest

from src.env import Env, Paths, RAG, IO, Models
from src.indexing import run_indexer
from src.database import ChromaConnector
from src.models import FakeEmbeddingFunction


@pytest.fixture
def index_test_env(tmp_path: Path) -> Env:
    talks_dir = tmp_path / "talks"
    talks_dir.mkdir(parents=True, exist_ok=True)

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

    assert collection.count() == 4

    ids = collection.get()["ids"]
    assert "talk_a_0" in ids


def test_run_indexer_id_stability(index_test_env: Env):
    run_indexer(index_test_env)
    run_indexer(index_test_env)

    connector = ChromaConnector(index_test_env)
    collection = connector.get_collection("rob_burbea_talks", embedding_function=FakeEmbeddingFunction())

    assert collection.count() == 4
