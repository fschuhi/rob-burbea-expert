from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence
import re

from src.env import Env, RAG


class Document:
    """Represents a single chunk of text with associated metadata."""

    def __init__(self, page_content: str, metadata: dict[str, str | Path]):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return (
            f"Document(source='{self.metadata.get('source')}', "
            f"chars={len(self.page_content)})"
        )


def load_markdown_talks(talks_dir: Path) -> Iterator[tuple[str, Path]]:
    """
    Loads all Markdown files from the specified directory.

    Yields:
        Tuple of (content, file_path) for each markdown file found.
    """
    if not talks_dir.exists() or not talks_dir.is_dir():
        raise FileNotFoundError(f"Talks directory not found: {talks_dir}")

    for file_path in talks_dir.glob("*.md"):
        yield file_path.read_text(encoding="utf-8"), file_path


def _split_text_with_overlap(
        text: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str]
) -> list[str]:
    """
    Manual text splitter that recursively tries separators.

    Splits text using separators in order, applying overlap between chunks.
    This is a lightweight alternative to RecursiveCharacterTextSplitter.

    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        separators: List of separators to try in order ["\n", " ", ""]

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    # Try each separator in order
    for sep in separators:
        if sep == "":
            # Last resort: character-level split with overlap
            chunks = []
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                chunks.append(text[start:end])
                start = end - chunk_overlap if chunk_overlap > 0 else end
                if start >= len(text):
                    break
            return chunks

        # Split on separator
        parts = text.split(sep)
        if len(parts) == 1:
            # Separator not found, try next one
            continue

        # Reassemble parts into chunks, respecting chunk_size
        chunks = []
        current_chunk = []
        current_size = 0

        for i, part in enumerate(parts):
            part_size = len(part) + (len(sep) if i > 0 else 0)  # Account for separator

            # If single part exceeds chunk_size, recursively split it
            if part_size > chunk_size and not current_chunk:
                # Try next separator in the hierarchy
                next_sep_idx = separators.index(sep) + 1
                if next_sep_idx < len(separators):
                    sub_chunks = _split_text_with_overlap(
                        part, chunk_size, chunk_overlap, separators[next_sep_idx:]
                    )
                    chunks.extend(sub_chunks)
                else:
                    # No more separators, force split
                    chunks.append(part[:chunk_size])
                continue

            # Check if adding this part would exceed chunk_size
            if current_size + part_size > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = sep.join(current_chunk)
                chunks.append(chunk_text)

                # Handle overlap: keep last part of previous chunk
                if chunk_overlap > 0 and len(chunk_text) > chunk_overlap:
                    # Find where to split for overlap
                    overlap_text = chunk_text[-chunk_overlap:]
                    # Start new chunk with overlap
                    current_chunk = [overlap_text, part] if part else [overlap_text]
                    current_size = len(overlap_text) + len(sep) + len(part)
                else:
                    current_chunk = [part] if part else []
                    current_size = len(part)
            else:
                # Add part to current chunk
                current_chunk.append(part)
                current_size += part_size

        # Add remaining chunk
        if current_chunk:
            chunks.append(sep.join(current_chunk))

        return chunks

    # Fallback: return as-is if no splitting worked
    return [text]


def _split_with_manual_splitter(
        text_content: str,
        rag_config: RAG,
        source_path: Path
) -> Sequence[Document]:
    """
    Manual semantic-first splitter implementation.

    Phase 1: Split on paragraphs
    Phase 2: For oversized paragraphs, split with overlap

    This is faster than langchain because it has no ML dependencies.
    """
    # Normalize line endings
    clean_content = text_content.replace('\r\n', '\n').replace('\r', '\n')
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)

    # Phase 1: Split on paragraphs
    paragraphs = clean_content.split('\n\n')

    # Phase 2: Process each paragraph
    chunks: list[Document] = []

    for paragraph in paragraphs:
        para_stripped = paragraph.strip()

        if not para_stripped:
            continue

        if len(para_stripped) <= rag_config.chunk_size:
            # Paragraph fits, keep as-is
            chunks.append(Document(
                page_content=para_stripped,
                metadata={"source": str(source_path)}
            ))
        else:
            # Paragraph too large, split it with overlap
            sub_chunks = _split_text_with_overlap(
                para_stripped,
                rag_config.chunk_size,
                rag_config.chunk_overlap,
                ["\n", " ", ""]  # Try line breaks, then words, then characters
            )
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():
                    chunks.append(Document(
                        page_content=sub_chunk.strip(),
                        metadata={"source": str(source_path)}
                    ))

    return chunks


def _split_with_langchain(
        text_content: str,
        rag_config: RAG,
        source_path: Path
) -> Sequence[Document]:
    """
    Langchain-based splitter (slower due to ML library imports).

    This is kept as a fallback option to validate the manual splitter.
    """
    # Lazy import to avoid loading heavy dependencies unless needed
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Normalize line endings
    clean_content = text_content.replace('\r\n', '\n').replace('\r', '\n')
    clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)

    # Phase 1: Split on paragraphs
    paragraphs = clean_content.split('\n\n')

    # Phase 2: Prepare splitter for oversized paragraphs
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=rag_config.chunk_size,
        chunk_overlap=rag_config.chunk_overlap,
        separators=["\n", " ", ""],
        length_function=len,
        is_separator_regex=False
    )

    # Phase 3: Process each paragraph
    chunks: list[Document] = []

    for paragraph in paragraphs:
        para_stripped = paragraph.strip()

        if not para_stripped:
            continue

        if len(para_stripped) <= rag_config.chunk_size:
            chunks.append(Document(
                page_content=para_stripped,
                metadata={"source": str(source_path)}
            ))
        else:
            sub_chunks = text_splitter.split_text(para_stripped)
            for sub_chunk in sub_chunks:
                if sub_chunk.strip():
                    chunks.append(Document(
                        page_content=sub_chunk.strip(),
                        metadata={"source": str(source_path)}
                    ))

    return chunks


def split_documents(
        text_content: str,
        rag_config: RAG,
        source_path: Path
) -> Sequence[Document]:
    """
    Splits a document using a semantic-first, two-phase approach.

    Strategy:
    1. Normalize line endings (Windows CRLF → Unix LF)
    2. Normalize excessive blank lines (3+ newlines → 2)
    3. Split on paragraph boundaries (\\n\\n)
    4. For each paragraph:
       - If ≤ chunk_size: keep as single chunk
       - If > chunk_size: split with overlap (respecting line/word boundaries)

    The implementation can use either:
    - Manual splitter (fast, no ML dependencies) - DEFAULT
    - Langchain splitter (slower, battle-tested) - FALLBACK

    Controlled by rag_config.use_langchain_splitter

    Args:
        text_content: Raw markdown text to split
        rag_config: Configuration with chunk_size, chunk_overlap, use_langchain_splitter
        source_path: Source file path for metadata

    Returns:
        Sequence of Document objects with page_content and metadata
    """
    text_content = text_content.strip()

    if rag_config.use_langchain_splitter:
        return _split_with_langchain(text_content, rag_config, source_path)
    else:
        return _split_with_manual_splitter(text_content, rag_config, source_path)


def prepare_data_pipeline(env: Env) -> Sequence[Document]:
    """
    Runs the full document loading and splitting pipeline.

    Processes all markdown files in the configured talks directory,
    splitting each into appropriately-sized chunks for embedding.

    Args:
        env: Environment configuration with paths and RAG settings

    Returns:
        Sequence of all Document chunks ready for embedding
    """
    all_documents: list[Document] = []

    print(f"Loading talks from: {env.paths.raw_talks_dir}")

    for content, path in load_markdown_talks(env.paths.raw_talks_dir):
        all_documents.extend(split_documents(
            text_content=content,
            rag_config=env.rag,
            source_path=path
        ))

    print(f"Prepared {len(all_documents)} chunks.")
    return all_documents
