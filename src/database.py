from __future__ import annotations

from typing import Any
import chromadb
from chromadb.api.models.Collection import Collection
from src.env import Env


class ChromaConnector:
    """
    Wraps ChromaDB client handling to ensure consistent configuration
    across the application.
    """

    def __init__(self, env: Env):
        """
        Initialize the ChromaDB client using the path defined in Env.
        We use PersistentClient to ensure data is saved to disk at:
        env.paths.chroma_db_dir
        """
        self.db_path = env.paths.chroma_db_dir
        self.client = chromadb.PersistentClient(path=str(self.db_path))

    def get_collection(self, name: str, embedding_function=None) -> Collection:
        """
        Get or create a ChromaDB collection.
        Args:
            name: The name of the collection (e.g., "rob_burbea_talks").
            embedding_function: The function used to embed text.
                                If None, ChromaDB uses its default (all-MiniLM-L6-v2).
        We allow passing this to support dependency injection (testing).

        Returns:
            A ChromaDB Collection object.
        """
        # We explicitly set cosine distance as it is standard for semantic search
        return self.client.get_or_create_collection(
            name=name, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        """
        Resets the database.
        WARNING: This is destructive and mostly useful for testing clean slates.
        """
        self.client.reset()


# --- Paragraph Reconstruction Helpers ---


def get_paragraph_chunks(collection: Collection, source: str, paragraph_index: int | str) -> list[dict[str, Any]]:
    """
    Fetch all chunks belonging to a specific paragraph, sorted by chunk_position.
    Args:
        collection: ChromaDB collection to query
        source: Source file path (e.g., "path/to/talk.md")
        paragraph_index: The paragraph index in the source document (int or str)

    Returns:
        List of dicts with 'id', 'text', 'metadata' keys, sorted by chunk_position.
        Returns empty list if no chunks found.
    """
    # REVERT: Do not force str() conversion here.
    # Production uses Strings ("10"), Tests use Integers (10).
    # We trust the caller to pass the correct type for the environment they are in.

    # Query with metadata filter for source AND paragraph_index
    results = collection.get(where={"$and": [{"source": source}, {"paragraph_index": paragraph_index}]})

    # Combine into list of dicts
    chunks = []
    for i in range(len(results["ids"])):
        chunks.append({"id": results["ids"][i], "text": results["documents"][i], "metadata": results["metadatas"][i]})

    # Sort by chunk_position (Robust Sort)
    # We attempt to cast to int for sorting, so "10" comes after "2".
    try:
        chunks.sort(key=lambda x: int(x["metadata"]["chunk_position"]))
    except (ValueError, TypeError):
        # Fallback for weird data
        chunks.sort(key=lambda x: str(x["metadata"].get("chunk_position", "0")))

    return chunks


def reconstruct_paragraph_with_hit(
    collection: Collection, source: str, paragraph_index: int | str, hit_chunk_position: int | str
) -> dict[str, Any]:
    """
    Reconstruct a full paragraph from its chunks and mark the hit chunk with XML tags.

    Args:
        collection: ChromaDB collection to query
        source: Source file path
        paragraph_index: The paragraph index in the source document
        hit_chunk_position: Which chunk within the paragraph was the search hit

    Returns:
        Dict with:
            - 'full_text': Complete paragraph text (plain, no markup)
            - 'marked_text': Full paragraph with <hit>...</hit> tags around the hit chunk
            - 'hit_text': The text of just the hit chunk
            - 'num_chunks': Total number of chunks in this paragraph

    Raises:
        ValueError: If no chunks found or hit_chunk_position is invalid
    """
    chunks = get_paragraph_chunks(collection, source, paragraph_index)

    if not chunks:
        raise ValueError(f"No chunks found for source='{source}', paragraph_index={paragraph_index}")

    # Reconstruct full text (no markup)
    full_text = " ".join(chunk["text"] for chunk in chunks)

    # Find the hit chunk
    # Robust comparison: Convert both to Int for the equality check
    hit_chunk = None
    try:
        target_pos = int(hit_chunk_position)
    except ValueError:
        target_pos = hit_chunk_position  # Fallback

    for chunk in chunks:
        try:
            current_pos = int(chunk["metadata"]["chunk_position"])
        except ValueError:
            current_pos = chunk["metadata"]["chunk_position"]

        if current_pos == target_pos:
            hit_chunk = chunk
            break

    if hit_chunk is None:
        raise ValueError(
            f"Hit chunk at position {hit_chunk_position} not found in paragraph "
            f"(available positions: {[c['metadata']['chunk_position'] for c in chunks]})"
        )

    # Build marked text with <hit> tags around the hit chunk
    marked_parts = []
    for chunk in chunks:
        try:
            current_pos = int(chunk["metadata"]["chunk_position"])
        except ValueError:
            current_pos = chunk["metadata"]["chunk_position"]

        if current_pos == target_pos:
            marked_parts.append(f"<hit>{chunk['text']}</hit>")
        else:
            marked_parts.append(chunk["text"])

    marked_text = " ".join(marked_parts)

    return {
        "full_text": full_text,
        "marked_text": marked_text,
        "hit_text": hit_chunk["text"],
        "num_chunks": len(chunks),
    }
