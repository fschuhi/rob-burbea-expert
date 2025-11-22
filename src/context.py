from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple, Dict, List

from chromadb.api.models.Collection import Collection
from src.database import get_paragraph_chunks


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
