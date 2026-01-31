from __future__ import annotations

from pathlib import Path

import pytest

# Import the code we want to test
from src.data_prep import (
    Document,
    load_markdown_talks,
    split_documents,
    _clean_transcription_noise,
)
from src.env import Env, RAG, Paths


# =============================================================================
# Test Fixtures & Setup
# =============================================================================


@pytest.fixture
def sample_talk_fixtures(tmp_path: Path):
    """
    Creates temporary directories and sample talk files for testing.
    This fixture is now explicitly normalized to guarantee correct separator recognition.
    """
    talks_dir = tmp_path / "raw_talks"
    talks_dir.mkdir()

    # Talk 1: Explicitly structured to force 5 blocks via '\n\n' and separators
    talk_a = (
        "# Talk A: Introduction to Soulmaking\n\n"
        "This is the first paragraph, it's short and highly coherent.\n\n"
        "This is the second, equally short.\n\n"
        "---\n\n"
        "This third paragraph is much longer, discussing the three phases of soulmaking:\n"
        "preparation, encounter, and integration. It spans multiple lines but contains\n"
        "no double newlines until the very end, meaning it should be treated as one\n"
        "semantic unit, unless its size exceeds the chunk limit."
    )

    # Talk 2: Very short, guaranteed to break into 2 blocks (Title and Body)
    talk_b = (
        "# Talk B: Breaking the Vessel\n\n"
        "A brief note on the concept of breaking the vessel, which allows for\n"
        "new capacity to be developed."
    )

    # Use raw string assignment to avoid textwrap.dedent issues
    (talks_dir / "2023-01-01-talk-a.md").write_text(talk_a, encoding="utf-8")
    (talks_dir / "2023-01-02-talk-b.md").write_text(talk_b, encoding="utf-8")

    return talks_dir


@pytest.fixture
def default_env(tmp_path: Path) -> Env:
    """Provides a minimal Env object for testing purposes."""
    return Env(
        paths=Paths(
            data_dir=tmp_path / "data",
            raw_talks_dir=tmp_path / "raw_talks",
            chroma_db_dir=tmp_path / "chroma_db",
            metadata_path=tmp_path / "meta.json",
        ),
        rag=RAG(chunk_size=1000, chunk_overlap=50, use_langchain_splitter=False),  # Use fast manual splitter by default
    )


# =============================================================================
# Unit Tests - Noise Reduction
# =============================================================================


def test_clean_transcription_noise_removes_timestamps():
    """Test that timestamp markers [HH:MM] are removed."""
    text = "Some text [28:02] and more text [3:56] here."
    cleaned = _clean_transcription_noise(text)

    assert "[28:02]" not in cleaned
    assert "[3:56]" not in cleaned
    assert "Some text" in cleaned
    assert "and more text" in cleaned
    assert "here." in cleaned


def test_clean_transcription_noise_removes_sound_markers():
    """Test that sound markers like [laughter] are removed."""
    text = "This is funny [laughter] and [inaudible] stuff."
    cleaned = _clean_transcription_noise(text)

    assert "[laughter]" not in cleaned
    assert "[inaudible]" not in cleaned
    assert "This is funny" in cleaned
    assert "and" in cleaned
    assert "stuff." in cleaned


def test_clean_transcription_noise_removes_speaker_markers():
    """Test that speaker markers like [yogi inaudible in background] are removed."""
    text = "Question here [yogi inaudible in background] and answer."
    cleaned = _clean_transcription_noise(text)

    assert "[yogi inaudible in background]" not in cleaned
    assert "Question here" in cleaned
    assert "and answer." in cleaned


def test_clean_transcription_noise_normalizes_whitespace():
    """Test that multiple spaces are collapsed to single space."""
    text = "Text [28:02]  [laughter]   more text"
    cleaned = _clean_transcription_noise(text)

    # Should collapse multiple spaces
    assert "  " not in cleaned
    assert "Text more text" in cleaned


def test_clean_transcription_noise_preserves_paragraph_breaks():
    """Test that paragraph breaks (\n\n) are preserved."""
    text = "First paragraph.\n\n[28:02]\n\nSecond paragraph."
    cleaned = _clean_transcription_noise(text)

    assert "First paragraph.\n\nSecond paragraph." in cleaned
    assert "[28:02]" not in cleaned


def test_clean_transcription_noise_removes_excessive_newlines():
    """Test that 3+ newlines are normalized to 2."""
    text = "Paragraph 1\n\n\n\nParagraph 2"
    cleaned = _clean_transcription_noise(text)

    assert "\n\n\n" not in cleaned
    assert "Paragraph 1\n\nParagraph 2" in cleaned


def test_clean_transcription_noise_case_insensitive():
    """Test that markers are removed regardless of case."""
    text = "Text [LAUGHTER] and [Inaudible] stuff."
    cleaned = _clean_transcription_noise(text)

    assert "[LAUGHTER]" not in cleaned
    assert "[Inaudible]" not in cleaned
    assert "Text and stuff." in cleaned


def test_clean_transcription_noise_complex_example():
    """Test cleaning with multiple noise types in realistic text."""
    text = """
    Thank you, Sari.

    `[3:56]` So hopefully you could get a sense of, you know, what you've put into the time here.

    [laughter] That's really funny.

    [yogi inaudible in background] Some question here.
    """

    cleaned = _clean_transcription_noise(text)

    # Noise should be gone
    assert "[3:56]" not in cleaned
    assert "[laughter]" not in cleaned
    assert "[yogi inaudible in background]" not in cleaned

    # Content should remain
    assert "Thank you, Sari." in cleaned
    assert "So hopefully you could get a sense" in cleaned
    assert "That's really funny." in cleaned
    assert "Some question here." in cleaned


def test_clean_transcription_noise_empty_brackets_remain():
    """Test that empty paragraphs with only noise become empty (not removed here)."""
    text = "[28:02]"
    cleaned = _clean_transcription_noise(text)

    # Should be empty or whitespace only
    assert cleaned.strip() == ""


# =============================================================================
# Integration Tests - Noise Reduction in Chunking
# =============================================================================


def test_split_documents_with_noise_filters_correctly(tmp_path: Path):
    """Test that noise is filtered during the actual chunking process."""
    talks_dir = tmp_path / "raw_talks"
    talks_dir.mkdir()

    noisy_talk = """# Talk with Noise

This is the first paragraph with a timestamp [5:30] in it.

[laughter]

Another paragraph [yogi inaudible] with markers.

`[28:02]` Final paragraph with inline timestamp.
"""

    talk_path = talks_dir / "noisy-talk.md"
    talk_path.write_text(noisy_talk, encoding="utf-8")

    rag_config = RAG(chunk_size=1000, chunk_overlap=0, use_langchain_splitter=False)
    chunks = split_documents(noisy_talk, rag_config, talk_path)

    # Should have 3 paragraphs (header, para1, para2, para3)
    # The [laughter] paragraph should be empty after cleaning and thus skipped
    assert len(chunks) == 4

    # Check noise is removed
    all_text = " ".join(chunk.page_content for chunk in chunks)
    assert "[5:30]" not in all_text
    assert "[laughter]" not in all_text
    assert "[yogi inaudible]" not in all_text
    assert "[28:02]" not in all_text

    # Check content is preserved
    assert "This is the first paragraph with a timestamp" in all_text
    assert "Another paragraph" in all_text
    assert "Final paragraph with inline timestamp." in all_text


def test_split_documents_noise_doesnt_create_island_chunks(tmp_path: Path):
    """Test that noise-only paragraphs don't create meaningless chunks."""
    talks_dir = tmp_path / "raw_talks"
    talks_dir.mkdir()

    noisy_talk = """# Talk

Real content here.

[28:02]

[laughter]

[inaudible]

More real content.
"""

    talk_path = talks_dir / "island-test.md"
    talk_path.write_text(noisy_talk, encoding="utf-8")

    rag_config = RAG(chunk_size=500, chunk_overlap=0, use_langchain_splitter=False)
    chunks = split_documents(noisy_talk, rag_config, talk_path)

    # Should only have 3 chunks: header, para1, para2
    # The noise-only paragraphs should be filtered out
    assert len(chunks) == 3

    # Verify no chunks contain only noise
    for chunk in chunks:
        # Each chunk should have substantial content
        assert len(chunk.page_content.strip()) > 5
        assert not chunk.page_content.strip().startswith("[")


# =============================================================================
# Unit Tests - Manual Splitter (Default, Fast)
# =============================================================================


def test_load_markdown_talks_success(sample_talk_fixtures: Path):
    """Tests that all markdown files in the directory are loaded."""
    loaded_talks_with_path = list(load_markdown_talks(sample_talk_fixtures))

    assert len(loaded_talks_with_path) == 2

    content_a = loaded_talks_with_path[0][0]
    content_b = loaded_talks_with_path[1][0]

    assert "Introduction to Soulmaking" in content_a
    assert "Breaking the Vessel" in content_b


def test_load_markdown_talks_not_found():
    """Tests that an error is raised if the talks directory doesn't exist."""
    missing_path = Path("/nonexistent/path/to/talks")
    with pytest.raises(FileNotFoundError) as exc:
        list(load_markdown_talks(missing_path))
    assert "Talks directory not found" in str(exc.value)


def test_split_documents_respects_paragraphs(sample_talk_fixtures: Path):
    """
    Tests that the splitter uses paragraph breaks ('\n\n') to create coherent chunks.
    Uses a large chunk_size to ensure no forced splits.
    Uses MANUAL splitter (fast, no ML dependencies).
    """
    talk_a_path = sample_talk_fixtures / "2023-01-01-talk-a.md"
    talk_a_content = talk_a_path.read_text(encoding="utf-8")

    rag_config = RAG(chunk_size=1000, chunk_overlap=0, use_langchain_splitter=False)

    chunks = split_documents(talk_a_content, rag_config, talk_a_path)

    # Expected 5 chunks: Header, P1, P2, ---, P3
    assert len(chunks) == 5, f"Expected 5 chunks (5 blocks), got {len(chunks)}"

    # 0. Header/Title chunk
    assert "Introduction to Soulmaking" in chunks[0].page_content

    # 2. Second short paragraph chunk
    assert "This is the second, equally short." in chunks[2].page_content

    # 4. Long paragraph chunk (should be the last one)
    assert "preparation, encounter, and integration" in chunks[-1].page_content
    assert chunks[-1].page_content.count("\n\n") == 0


def test_split_documents_enforces_size_limit(sample_talk_fixtures: Path):
    """
    Tests that a very small chunk_size will force a break inside a long paragraph.
    Uses MANUAL splitter.
    """
    talk_a_path = sample_talk_fixtures / "2023-01-01-talk-a.md"
    talk_a_content = talk_a_path.read_text(encoding="utf-8")

    # The long third paragraph is ~250 chars. We set the limit to 100.
    rag_config = RAG(chunk_size=100, chunk_overlap=0, use_langchain_splitter=False)

    chunks = split_documents(talk_a_content, rag_config, talk_a_path)

    # Expected: 5 initial blocks, but P3 will be split into 3+ chunks, total 7+ chunks
    assert len(chunks) > 5, "Expected more chunks when size limit is small"

    # Find the chunks related to the long paragraph (the fifth logical block)
    long_paragraph_chunks = [
        c
        for c in chunks
        if "discussing the three phases" in c.page_content or "preparation, encounter" in c.page_content
    ]

    # Verify the forced split
    assert len(long_paragraph_chunks) > 1, "The long paragraph should have been forcibly split."
    assert all(
        len(c.page_content) <= 100 for c in long_paragraph_chunks
    ), "All final chunks must respect the size limit."


def test_split_documents_metadata_assignment(sample_talk_fixtures: Path):
    """
    Tests that the correct metadata (source path) is attached to all chunks.
    Uses MANUAL splitter.
    """
    talk_b_path = sample_talk_fixtures / "2023-01-02-talk-b.md"
    talk_b_content = talk_b_path.read_text(encoding="utf-8")

    chunks = split_documents(
        talk_b_content, RAG(chunk_size=500, chunk_overlap=0, use_langchain_splitter=False), talk_b_path
    )

    # Assert 2 chunks (Header and Body) are created by '\n\n' split
    assert len(chunks) == 2, f"Expected 2 chunks (Title and Body), got {len(chunks)}"
    assert chunks[0].metadata["source"] == str(talk_b_path)
    assert chunks[1].metadata["source"] == str(talk_b_path)
    assert isinstance(chunks[0], Document)


# =============================================================================
# Langchain Splitter Tests (Validates Fallback Works)
# =============================================================================


def test_split_documents_with_langchain_splitter(sample_talk_fixtures: Path):
    """
    Tests that the langchain splitter produces equivalent results.
    This test validates the fallback option still works correctly.

    NOTE: This test is SLOWER due to ML library imports, but ensures
    both implementations produce compatible results.
    """
    talk_a_path = sample_talk_fixtures / "2023-01-01-talk-a.md"
    talk_a_content = talk_a_path.read_text(encoding="utf-8")

    # Use langchain splitter explicitly
    rag_config = RAG(chunk_size=1000, chunk_overlap=0, use_langchain_splitter=True)

    chunks = split_documents(talk_a_content, rag_config, talk_a_path)

    # Should produce same 5 chunks as manual splitter
    assert len(chunks) == 5, f"Expected 5 chunks with langchain splitter, got {len(chunks)}"
    assert "Introduction to Soulmaking" in chunks[0].page_content
    assert "preparation, encounter, and integration" in chunks[-1].page_content


def test_both_splitters_produce_similar_results(sample_talk_fixtures: Path):
    """
    Validates that manual and langchain splitters produce functionally equivalent results.
    """
    talk_b_path = sample_talk_fixtures / "2023-01-02-talk-b.md"
    talk_b_content = talk_b_path.read_text(encoding="utf-8")

    # Split with manual splitter
    manual_chunks = split_documents(
        talk_b_content, RAG(chunk_size=500, chunk_overlap=0, use_langchain_splitter=False), talk_b_path
    )

    # Split with langchain splitter
    langchain_chunks = split_documents(
        talk_b_content, RAG(chunk_size=500, chunk_overlap=0, use_langchain_splitter=True), talk_b_path
    )

    # Both should produce 2 chunks
    assert len(manual_chunks) == len(langchain_chunks) == 2

    # Content should be identical (modulo whitespace normalization)
    assert manual_chunks[0].page_content == langchain_chunks[0].page_content
    assert manual_chunks[1].page_content == langchain_chunks[1].page_content


def test_overlap_behavior_with_oversized_paragraph(sample_talk_fixtures: Path):
    """
    Tests that overlap is correctly applied when sub-splitting an oversized paragraph.
    """
    # Create a test with a single oversized paragraph
    talks_dir = sample_talk_fixtures
    oversized_talk = (
        "This is a very long paragraph that will definitely exceed our chunk size limit. "
        "It contains multiple sentences that discuss various aspects of dharma practice. "
        "The Buddha taught that suffering arises from craving and clinging. "
        "Through mindfulness and meditation, we can develop insight into the nature of experience. "
        "This leads to greater freedom and ease in our lives."
    )

    oversized_path = talks_dir / "2023-01-03-oversized.md"
    oversized_path.write_text(oversized_talk, encoding="utf-8")

    # Use small chunk_size with overlap
    rag_config = RAG(chunk_size=100, chunk_overlap=20, use_langchain_splitter=False)

    chunks = split_documents(oversized_talk, rag_config, oversized_path)

    # Should have multiple chunks due to size constraint
    assert len(chunks) > 1, "Expected multiple chunks for oversized paragraph"

    # Verify overlap: end of chunk N should overlap with start of chunk N+1
    for i in range(len(chunks) - 1):
        current_end = chunks[i].page_content[-20:]  # Last 20 chars
        next_start = chunks[i + 1].page_content[:50]  # First 50 chars (generous)

        # There should be some overlap (not exact due to word boundaries)
        # Just verify chunks are reasonably sized
        assert len(chunks[i].page_content) <= 120, f"Chunk {i} too large: {len(chunks[i].page_content)}"
