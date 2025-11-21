from __future__ import annotations

import pytest
from pathlib import Path
import shutil

from src.env import Env, Paths, RAG, IO
from src.database import ChromaConnector
from src.models import FakeEmbeddingFunction


@pytest.fixture
def test_env(tmp_path: Path) -> Env:
    """
    Creates a standard test environment with temporary directories.
    Using tmp_path ensures cleanup after tests.
    """
    # Define paths within the temp directory
    fixtures_dir = tmp_path / "fixtures"
    data_dir = fixtures_dir / "data"
    raw_talks_dir = data_dir / "raw_talks"
    chroma_dir = data_dir / "chroma_db"
    metadata_path = data_dir / "metadata.json"

    # Create structure
    raw_talks_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy metadata file so validation passes
    metadata_path.write_text("[]", encoding="utf-8")

    return Env(
        paths=Paths(
            data_dir=data_dir,
            raw_talks_dir=raw_talks_dir,
            chroma_db_dir=chroma_dir,
            metadata_path=metadata_path
        ),
        rag=RAG(chunk_size=500, chunk_overlap=50),
        io=IO(create_missing_dirs=True)
    )


@pytest.fixture
def db_connector(test_env: Env) -> ChromaConnector:
    """Returns an initialized connector linked to the temp env."""
    return ChromaConnector(test_env)


@pytest.fixture
def fake_ef():
    """Returns the deterministic FakeEmbeddingFunction."""
    return FakeEmbeddingFunction()


@pytest.fixture
def populated_collection(db_connector, fake_ef):
    """
    Returns a collection pre-populated with a split paragraph.
    Useful for ContextBuilder and Retrieval tests.
    """
    collection = db_connector.get_collection("integration_test", embedding_function=fake_ef)

    # Data: A paragraph split into 3 chunks
    # Source: test_talk.md, Paragraph Index: 10
    chunks = [
        {
            "id": "test_talk_0",
            "document": "Beginning of the paragraph.",
            "metadata": {"source": "test_talk.md", "paragraph_index": 10, "chunk_position": 0,
                         "total_chunks_in_para": 3}
        },
        {
            "id": "test_talk_1",
            "document": "Middle of the paragraph (the hit).",
            "metadata": {"source": "test_talk.md", "paragraph_index": 10, "chunk_position": 1,
                         "total_chunks_in_para": 3}
        },
        {
            "id": "test_talk_2",
            "document": "End of the paragraph.",
            "metadata": {"source": "test_talk.md", "paragraph_index": 10, "chunk_position": 2,
                         "total_chunks_in_para": 3}
        }
    ]

    collection.add(
        ids=[c["id"] for c in chunks],
        documents=[c["document"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks]
    )

    return collection
