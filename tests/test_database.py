from __future__ import annotations

from pathlib import Path
import pytest
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from src.env import Env, Paths, RAG, IO
from src.database import ChromaConnector


# --- Mocks & Fixtures ---

class MockEmbeddingFunction(EmbeddingFunction):
    """
    A fake embedding function for testing.
    Inherits from Chroma's base class to ensure compatibility.
    """

    def __call__(self, input: Documents) -> Embeddings:
        # Return a simple 3-float vector for every document
        return [[0.1, 0.2, 0.3] for _ in input]


@pytest.fixture
def test_env() -> Env:
    """Creates an Env configured with tmp/chroma_db for easy inspection."""

    # Use project root's tmp directory for ChromaDB
    project_root = Path(__file__).parent.parent
    chroma_dir = project_root / "tmp" / "chroma_db"

    # Point to existing fixture data
    fixtures_dir = project_root / "tests" / "fixtures" / "data"

    return Env(
        paths=Paths(
            data_dir=fixtures_dir,
            raw_talks_dir=fixtures_dir / "raw_talks",
            chroma_db_dir=chroma_dir,
            metadata_path=fixtures_dir / "metadata.json"
        ),
        rag=RAG(),
        io=IO(create_missing_dirs=True)
    )


@pytest.fixture
def db_connector(test_env: Env) -> ChromaConnector:
    """Returns an initialized connector linked to the temp env."""
    return ChromaConnector(test_env)


# --- Tests ---

def test_connector_initializes_client(test_env: Env):
    """Verifies the connector sets up the PersistentClient at the right path."""
    connector = ChromaConnector(test_env)

    # FIX: Check against the general Client API class, not the factory function
    assert isinstance(connector.client, chromadb.api.client.Client)

    assert test_env.paths.chroma_db_dir.exists()


def test_create_collection_with_mock_embedding(db_connector: ChromaConnector):
    """Verifies we can create a collection using a custom embedding function."""
    mock_ef = MockEmbeddingFunction()

    collection = db_connector.get_collection(
        name="test_collection",
        embedding_function=mock_ef
    )

    assert collection.name == "test_collection"
    assert collection.count() == 0


def test_add_and_query_documents(db_connector: ChromaConnector):
    """
    A full smoke test: Add a document and verify it exists.
    This proves the read/write cycle works.
    """
    mock_ef = MockEmbeddingFunction()
    collection = db_connector.get_collection("smoke_test", embedding_function=mock_ef)

    # Add a dummy document
    collection.add(
        ids=["doc1"],
        documents=["This is a test document about jhana."],
        metadatas=[{"source": "test_file.md"}]
    )

    # Verify it's there
    assert collection.count() == 1

    # Basic retrieval check
    result = collection.get(ids=["doc1"])
    assert result["documents"][0] == "This is a test document about jhana."
