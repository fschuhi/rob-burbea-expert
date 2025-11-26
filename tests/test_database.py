from __future__ import annotations

import pytest
import chromadb
from src.env import Env
from src.database import ChromaConnector
from src.context import get_paragraph_chunks, reconstruct_paragraph_with_hit


# --- ChromaConnector Tests ---


def test_connector_initializes_client(test_env: Env):
    """Verifies the connector sets up the PersistentClient at the right path."""
    connector = ChromaConnector(test_env)
    assert isinstance(connector.client, chromadb.api.client.Client)
    assert test_env.paths.chroma_db_dir.exists()


def test_create_collection_with_mock_embedding(db_connector: ChromaConnector, fake_ef):
    """Verifies we can create a collection using a custom embedding function."""
    collection = db_connector.get_collection(name="test_collection", embedding_function=fake_ef)
    assert collection.name == "test_collection"
    assert collection.count() == 0


def test_add_and_query_documents(db_connector: ChromaConnector, fake_ef):
    """A full smoke test: Add a document and verify it exists."""
    collection = db_connector.get_collection("smoke_test", embedding_function=fake_ef)

    collection.add(
        ids=["doc1"], documents=["This is a test document about jhana."], metadatas=[{"source": "test_file.md"}]
    )

    assert collection.count() == 1
    result = collection.get(ids=["doc1"])
    assert result["documents"][0] == "This is a test document about jhana."


# --- Paragraph Reconstruction Tests ---
# We can now use the shared 'populated_collection' fixture from conftest.py
# or create specific ones if the test needs unique data.


def test_get_paragraph_chunks_returns_all_in_order(populated_collection):
    chunks = get_paragraph_chunks(populated_collection, source="test_talk.md", paragraph_index=10)
    assert len(chunks) == 3
    assert chunks[0]["metadata"]["chunk_position"] == 0
    assert chunks[2]["metadata"]["chunk_position"] == 2


def test_reconstruct_paragraph_with_hit_marks_middle_chunk(populated_collection):
    result = reconstruct_paragraph_with_hit(
        populated_collection, source="test_talk.md", paragraph_index=10, hit_chunk_position=1
    )

    assert "<hit>Middle of the paragraph" in result["marked_text"]
    assert "Beginning of the paragraph." in result["marked_text"]

    # Ensure no tags on the non-hit parts
    assert not result["marked_text"].startswith("<hit>")


def test_reconstruct_raises_on_invalid_position(populated_collection):
    with pytest.raises(ValueError, match="Hit chunk at position 99 not found"):
        reconstruct_paragraph_with_hit(
            populated_collection, source="test_talk.md", paragraph_index=10, hit_chunk_position=99
        )
