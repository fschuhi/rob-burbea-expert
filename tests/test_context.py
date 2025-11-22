from __future__ import annotations

import pytest
from src.context import ContextBuilder


@pytest.fixture
def builder(populated_collection):
    return ContextBuilder(populated_collection)


def test_build_context_end_to_end(builder, populated_collection):
    """
    Simulate a semantic search query and verify the context builder
    reconstructs the paragraph correctly using real DB data.
    """
    # 1. Simulate a Query Result
    db_data = populated_collection.get(ids=["test_talk_1"])

    query_results = {"ids": [["test_talk_1"]], "distances": [[0.2]], "metadatas": [db_data["metadatas"]]}

    # 2. Build Context
    # FIX: Unpack the tuple (context, references)
    context, references = builder.build(query_results)

    # 3. Verify Output
    print(f"DEBUG CONTEXT:\n{context}")

    # Check Header (Updated format)
    assert "### Reference [1]: test_talk.md (Paragraph 10)" in context

    # Check Reconstruction
    assert "Beginning of the paragraph." in context
    assert "End of the paragraph." in context

    # Check Reference Map
    assert 1 in references
    assert references[1]["id"] == "test_talk_1"


def test_build_context_filters_high_distance(builder, populated_collection):
    """Verify that hits above max_distance are ignored."""

    db_data = populated_collection.get(ids=["test_talk_1"])

    query_results = {
        "ids": [["test_talk_1"]],
        "distances": [[0.9]],  # Very high distance
        "metadatas": [db_data["metadatas"]],
    }

    # FIX: Unpack tuple
    context, references = builder.build(query_results, max_distance=0.5)

    assert context == "No relevant context found."
    assert references == {}


def test_error_handling_missing_chunks(builder, populated_collection):
    """
    Test what happens if the DB index finds a hit, but the paragraph chunks
    cannot be retrieved.
    """
    query_results = {
        "ids": [["ghost_chunk"]],
        "distances": [[0.1]],
        "metadatas": [[{"source": "ghost_file.md", "paragraph_index": 999, "chunk_position": 0}]],
    }

    # FIX: Unpack tuple
    context, references = builder.build(query_results)

    # Check Header (Updated format)
    assert "### Reference [1]: ghost_file.md" in context
    assert "[Error: Could not retrieve text]" in context
