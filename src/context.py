from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, Dict, List

from chromadb.api.models.Collection import Collection


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


class ContextBuilder:
    """
    Constructs the system context for the LLM.
    Assigns simple integer IDs [1], [2] to chunks so the LLM doesn't have to
    cite complex filenames.
    """

    def __init__(self, collection: Collection):
        self.collection = collection

    def build(
        self, query_results: dict[str, Any], max_distance: float = 0.6, max_items: int = 5
    ) -> Tuple[str, Dict[int, Any]]:
        """
        Standard retrieval build (from Chroma results).

        Returns:
            - context_str: Markdown string for the LLM.
            - references: Dictionary mapping ID (int) -> Citation Metadata.
        """
        # Chroma returns list-of-lists. Assume single query.
        ids = query_results["ids"][0]
        distances = query_results["distances"][0]
        metadatas = query_results["metadatas"][0]

        # 1. Filter and sort
        hits = []
        for i, dist in enumerate(distances):
            if dist <= max_distance:
                hits.append({"id": ids[i], "distance": dist, "metadata": metadatas[i]})

        hits.sort(key=lambda x: x["distance"])
        hits = hits[:max_items]

        if not hits:
            return "No relevant context found.", {}

        # 2. Build Context & Map
        return self.build_from_hits(hits)

    def build_from_hits(self, hits: List[Dict[str, Any]]) -> Tuple[str, Dict[int, Any]]:
        """
        Builds context from a list of pre-ranked hits (dictionaries).
        Expected hit keys: 'metadata'.
        """
        context_parts = []
        references = {}

        for i, hit in enumerate(hits, start=1):
            # We assign a simple integer ID here
            ref_id = i

            # Store metadata for lookup later
            references[ref_id] = hit

            # Format the text block with the ID clearly visible
            formatted_para = self._format_paragraph(hit, ref_id)
            context_parts.append(formatted_para)

        if not context_parts:
            return "No relevant context found.", {}

        return "\n\n".join(context_parts), references

    def _format_paragraph(self, hit: dict[str, Any], ref_id: int) -> str:
        """
        Reconstructs a paragraph and pre-pends the Reference ID [x].
        """
        meta = hit["metadata"]
        source_path = meta.get("source", "Unknown")
        para_idx = meta.get("paragraph_index")
        hit_chunk_pos = meta.get("chunk_position")

        filename = Path(source_path).name

        # HEADER format: [ID] Filename (Para X)
        header = f"### Reference [{ref_id}]: {filename} (Paragraph {para_idx})"

        if para_idx is None or hit_chunk_pos is None:
            return f"{header}\n[Error: Missing metadata]"

        chunks = get_paragraph_chunks(self.collection, source=source_path, paragraph_index=para_idx)

        if not chunks:
            return f"{header}\n[Error: Could not retrieve text]"

        text_parts = []
        for chunk in chunks:
            text_parts.append(chunk["text"])

        full_text = " ".join(text_parts)
        return f"{header}\n{full_text}"
