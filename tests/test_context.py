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
    # 1. Simulate a Query Result (as if we queried for "Middle")
    # We know 'test_talk_1' is the middle chunk.
    # We simulate that it came back with distance 0.2

    # Retrieve the actual metadata to ensure our query simulation matches DB reality
    db_data = populated_collection.get(ids=["test_talk_1"])

    query_results = {
        'ids': [['test_talk_1']],
        'distances': [[0.2]],
        'metadatas': [db_data['metadatas']]
    }

    # 2. Build Context
    context = builder.build(query_results)

    # 3. Verify Output
    print(f"DEBUG CONTEXT:\n{context}")

    # Check Header
    assert "### Source: test_talk.md (Paragraph 10)" in context

    # Check Reconstruction (All parts present)
    assert "Beginning of the paragraph." in context
    assert "End of the paragraph." in context

    # Check Markup
    # The middle chunk should be wrapped in <hit> with the distance
    expected_hit = '<hit distance="0.2000">Middle of the paragraph (the hit).</hit>'
    assert expected_hit in context


def test_build_context_filters_high_distance(builder, populated_collection):
    """Verify that hits above max_distance are ignored."""

    db_data = populated_collection.get(ids=["test_talk_1"])

    query_results = {
        'ids': [['test_talk_1']],
        'distances': [[0.9]],  # Very high distance (bad match)
        'metadatas': [db_data['metadatas']]
    }

    context = builder.build(query_results, max_distance=0.5)

    assert context == "No relevant context found."


def test_error_handling_missing_chunks(builder, populated_collection):
    """
    Test what happens if the DB index finds a hit, but the paragraph chunks
    cannot be retrieved (e.g. inconsistent state).
    """
    # Create a query result pointing to a file/paragraph that DOES NOT EXIST in the DB
    query_results = {
        'ids': [['ghost_chunk']],
        'distances': [[0.1]],
        'metadatas': [[{
            'source': 'ghost_file.md',
            'paragraph_index': 999,
            'chunk_position': 0
        }]]
    }

    context = builder.build(query_results)

    assert "### Source: ghost_file.md" in context
    assert "[Error: Could not retrieve paragraph chunks]" in context
