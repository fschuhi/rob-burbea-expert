from __future__ import annotations

from pathlib import Path
import pytest
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from src.env import Env, Paths, RAG, IO
from src.database import (
    ChromaConnector,
    get_paragraph_chunks,
    reconstruct_paragraph_with_hit
)


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
    """Creates an Env configured with tmp/chroma_db_database for easy inspection."""

    # Use project root's tmp directory for ChromaDB (separate subdirectory for database tests)
    project_root = Path(__file__).parent.parent
    chroma_dir = project_root / "tmp" / "chroma_db_database"

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


# --- ChromaConnector Tests ---

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


# --- Paragraph Reconstruction Tests ---

@pytest.fixture
def collection_with_split_paragraph(db_connector: ChromaConnector):
    """
    Creates a collection with a paragraph split into 3 chunks.
    This simulates a real scenario where a long paragraph gets chunked.
    """
    mock_ef = MockEmbeddingFunction()
    collection = db_connector.get_collection(
        "paragraph_test",
        embedding_function=mock_ef
    )

    # Simulate a paragraph that was split into 3 chunks
    source = "test_talk.md"
    para_idx = 5

    chunks = [
        {
            "id": "test_talk_0",
            "text": "This is the first part of a long paragraph.",
            "metadata": {
                "source": source,
                "paragraph_index": para_idx,
                "chunk_position": 0,
                "total_chunks_in_para": 3
            }
        },
        {
            "id": "test_talk_1",
            "text": "This is the second part, continuing the thought.",
            "metadata": {
                "source": source,
                "paragraph_index": para_idx,
                "chunk_position": 1,
                "total_chunks_in_para": 3
            }
        },
        {
            "id": "test_talk_2",
            "text": "And this is the final part of the paragraph.",
            "metadata": {
                "source": source,
                "paragraph_index": para_idx,
                "chunk_position": 2,
                "total_chunks_in_para": 3
            }
        }
    ]

    # Add chunks to collection
    for chunk in chunks:
        collection.add(
            ids=[chunk["id"]],
            documents=[chunk["text"]],
            metadatas=[chunk["metadata"]]
        )

    return collection


def test_get_paragraph_chunks_returns_all_in_order(collection_with_split_paragraph):
    """
    Verifies that get_paragraph_chunks fetches all chunks for a paragraph
    and returns them sorted by chunk_position.
    """
    chunks = get_paragraph_chunks(
        collection_with_split_paragraph,
        source="test_talk.md",
        paragraph_index=5
    )

    # Should get all 3 chunks
    assert len(chunks) == 3

    # Should be in order
    assert chunks[0]['metadata']['chunk_position'] == 0
    assert chunks[1]['metadata']['chunk_position'] == 1
    assert chunks[2]['metadata']['chunk_position'] == 2

    # Verify texts
    assert "first part" in chunks[0]['text']
    assert "second part" in chunks[1]['text']
    assert "final part" in chunks[2]['text']


def test_get_paragraph_chunks_returns_empty_for_nonexistent(collection_with_split_paragraph):
    """
    Verifies that get_paragraph_chunks returns empty list when no chunks match.
    """
    chunks = get_paragraph_chunks(
        collection_with_split_paragraph,
        source="nonexistent.md",
        paragraph_index=999
    )

    assert chunks == []


def test_reconstruct_paragraph_with_hit_marks_first_chunk(collection_with_split_paragraph):
    """
    Tests reconstruction with hit on the first chunk.
    """
    result = reconstruct_paragraph_with_hit(
        collection_with_split_paragraph,
        source="test_talk.md",
        paragraph_index=5,
        hit_chunk_position=0
    )

    # Check structure
    assert 'full_text' in result
    assert 'marked_text' in result
    assert 'hit_text' in result
    assert 'num_chunks' in result

    # Verify full_text (no markup)
    assert result['full_text'] == (
        "This is the first part of a long paragraph. "
        "This is the second part, continuing the thought. "
        "And this is the final part of the paragraph."
    )

    # Verify marked_text has <hit> tags around first chunk
    assert result['marked_text'].startswith("<hit>This is the first part")
    assert "</hit> This is the second part" in result['marked_text']

    # Verify hit_text is just the first chunk
    assert result['hit_text'] == "This is the first part of a long paragraph."

    # Verify num_chunks
    assert result['num_chunks'] == 3


def test_reconstruct_paragraph_with_hit_marks_middle_chunk(collection_with_split_paragraph):
    """
    Tests reconstruction with hit on the middle chunk.
    """
    result = reconstruct_paragraph_with_hit(
        collection_with_split_paragraph,
        source="test_talk.md",
        paragraph_index=5,
        hit_chunk_position=1
    )

    # Verify marked_text has <hit> tags around middle chunk
    assert "paragraph. <hit>This is the second part" in result['marked_text']
    assert "</hit> And this is the final" in result['marked_text']

    # First chunk should NOT have <hit> tags
    assert not result['marked_text'].startswith("<hit>")

    # Verify hit_text is just the middle chunk
    assert result['hit_text'] == "This is the second part, continuing the thought."


def test_reconstruct_paragraph_with_hit_marks_last_chunk(collection_with_split_paragraph):
    """
    Tests reconstruction with hit on the last chunk.
    """
    result = reconstruct_paragraph_with_hit(
        collection_with_split_paragraph,
        source="test_talk.md",
        paragraph_index=5,
        hit_chunk_position=2
    )

    # Verify marked_text has <hit> tags around last chunk
    assert "thought. <hit>And this is the final part" in result['marked_text']
    assert result['marked_text'].endswith("</hit>")

    # Verify hit_text is just the last chunk
    assert result['hit_text'] == "And this is the final part of the paragraph."


def test_reconstruct_paragraph_with_hit_raises_on_nonexistent_paragraph(
    collection_with_split_paragraph
):
    """
    Verifies that reconstruct_paragraph_with_hit raises ValueError
    when the paragraph doesn't exist.
    """
    with pytest.raises(ValueError, match="No chunks found"):
        reconstruct_paragraph_with_hit(
            collection_with_split_paragraph,
            source="nonexistent.md",
            paragraph_index=999,
            hit_chunk_position=0
        )


def test_reconstruct_paragraph_with_hit_raises_on_invalid_position(
    collection_with_split_paragraph
):
    """
    Verifies that reconstruct_paragraph_with_hit raises ValueError
    when hit_chunk_position is invalid.
    """
    with pytest.raises(ValueError, match="Hit chunk at position 99 not found"):
        reconstruct_paragraph_with_hit(
            collection_with_split_paragraph,
            source="test_talk.md",
            paragraph_index=5,
            hit_chunk_position=99
        )


def test_reconstruct_single_chunk_paragraph(db_connector: ChromaConnector):
    """
    Tests reconstruction of a paragraph that wasn't split (single chunk).
    """
    mock_ef = MockEmbeddingFunction()
    collection = db_connector.get_collection(
        "single_chunk_test",
        embedding_function=mock_ef
    )

    # Add a single-chunk paragraph
    collection.add(
        ids=["single_0"],
        documents=["This is a short paragraph that fits in one chunk."],
        metadatas=[{
            "source": "short_talk.md",
            "paragraph_index": 1,
            "chunk_position": 0,
            "total_chunks_in_para": 1
        }]
    )

    result = reconstruct_paragraph_with_hit(
        collection,
        source="short_talk.md",
        paragraph_index=1,
        hit_chunk_position=0
    )

    # Should work fine with single chunk
    assert result['num_chunks'] == 1
    assert result['full_text'] == "This is a short paragraph that fits in one chunk."
    assert result['marked_text'] == "<hit>This is a short paragraph that fits in one chunk.</hit>"
    assert result['hit_text'] == "This is a short paragraph that fits in one chunk."
