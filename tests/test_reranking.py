from __future__ import annotations

import pytest
from src.engine import RAGEngine
from src.env import load_env


@pytest.mark.skip(reason="Requires model download - run manually if model is cached")
def test_reranker_reorders_results(populated_collection):
    """
    End-to-end test to ensure Cross-Encoder improves precision.

    Scenario:
    We inject two documents:
    1. "The First Jhana is composed of piti and sukha." (Target)
    2. "The Third Jhana drops piti but keeps sukha." (Distractor)

    Query: "What characterizes the First Jhana?"

    Without reranking (Vector only), 'Third Jhana' often scores high due to
    word overlap (Jhana, piti, sukha).
    With reranking, 'First Jhana' should be strictly #1.
    """
    # 1. Setup Env & Engine
    env = load_env()
    # We assume the environment is set up with a valid reranker model in config

    # Initialize Engine
    try:
        engine = RAGEngine(env)
    except Exception:
        pytest.skip("Skipping reranker test: Model likely not downloaded.")

    # Override engine collection with our test fixture
    engine.collection = populated_collection

    # 2. Inject Test Data
    populated_collection.add(
        ids=["doc_first", "doc_third"],
        documents=[
            "The First Jhana is accompanied by applied and sustained thought, rapture, and happiness.",
            "In the Third Jhana, rapture fades away, and one remains mindful and equanimous.",
        ],
        metadatas=[
            {"source": "test", "paragraph_index": 1, "chunk_position": 0},
            {"source": "test", "paragraph_index": 2, "chunk_position": 0},
        ],
    )

    query = "What factors are present in the First Jhana?"

    # 3. Run Retrieve & Rerank
    # We request top_k=1. If reranker works, we MUST get 'doc_first'.
    # We set distance_threshold high (2.0) to ensure both candidates are considered initially.
    context_str, refs = engine.retrieve_and_rerank(query, top_k=1, distance_threshold=2.0)

    # 4. Verify
    # The context string should contain the text of the First Jhana doc
    print(f"DEBUG Context: {context_str}")

    assert "First Jhana" in context_str
    assert "Third Jhana" not in context_str

    # Double check the reference ID maps to the correct doc ID
    assert refs[1]["id"] == "doc_first"
