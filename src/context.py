from __future__ import annotations

from pathlib import Path
from typing import Any

from chromadb.api.models.Collection import Collection
from src.database import get_paragraph_chunks


class ContextBuilder:
    """
    Constructs the system context for the LLM by re-assembling full paragraphs
    from search hits and formatting them with metadata and hit-highlighting.
    """

    def __init__(self, collection: Collection):
        self.collection = collection

    def build(
            self,
            query_results: dict[str, Any],
            max_distance: float = 0.6,
            max_items: int = 5
    ) -> str:
        """
        Builds a formatted markdown string from ChromaDB query results.

        Args:
            query_results: The raw dictionary returned by collection.query().
                           Assumes a single query (index 0).
            max_distance: Cutoff for relevance (lower is better).
            max_items: Maximum number of paragraphs to include.

        Returns:
            A single markdown string containing the constructed context.
        """
        # Chroma returns list-of-lists for batch queries. We assume 1 query.
        ids = query_results['ids'][0]
        distances = query_results['distances'][0]
        metadatas = query_results['metadatas'][0]

        # 1. Filter and structure the hits
        hits = []
        for i, dist in enumerate(distances):
            if dist > max_distance:
                continue

            hits.append({
                'id': ids[i],
                'distance': dist,
                'metadata': metadatas[i]
            })

        # 2. Sort by distance (ascending = most relevant first)
        hits.sort(key=lambda x: x['distance'])

        # 3. Limit count
        hits = hits[:max_items]

        if not hits:
            return "No relevant context found."

        # 4. Build the context string
        context_parts = []

        for hit in hits:
            formatted_para = self._format_paragraph(hit)
            context_parts.append(formatted_para)

        return "\n\n".join(context_parts)

    def _format_paragraph(self, hit: dict[str, Any]) -> str:
        """
        Reconstructs a single paragraph and applies the <hit> markup.
        """
        meta = hit['metadata']
        source_path = meta.get('source', 'Unknown')
        para_idx = meta.get('paragraph_index')
        hit_chunk_pos = meta.get('chunk_position')

        # Header Information
        filename = Path(source_path).name
        header = f"### Source: {filename} (Paragraph {para_idx})"

        # If we lack metadata for reconstruction, fallback to just the header + error
        if para_idx is None or hit_chunk_pos is None:
            return f"{header}\n[Error: Missing reconstruction metadata]"

        # Fetch all chunks for this paragraph
        chunks = get_paragraph_chunks(
            self.collection,
            source=source_path,
            paragraph_index=para_idx
        )

        if not chunks:
            return f"{header}\n[Error: Could not retrieve paragraph chunks]"

        # Reassemble text
        text_parts = []
        for chunk in chunks:
            chunk_pos = chunk['metadata']['chunk_position']
            text = chunk['text']

            if chunk_pos == hit_chunk_pos:
                # Apply the rich markup requested
                score = hit['distance']
                text_parts.append(f'<hit distance="{score:.4f}">{text}</hit>')
            else:
                text_parts.append(text)

        full_text = " ".join(text_parts)

        return f"{header}\n{full_text}"
