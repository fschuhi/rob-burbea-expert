from __future__ import annotations

from pathlib import Path
import pytest
import numpy as np
import shutil

from src.env import Env, Paths, RAG, IO, Models
from src.indexing import run_indexer
from src.database import ChromaConnector
from src.models import get_embedding_function


@pytest.fixture(scope="module")
def retrieval_env() -> Env:
    """
    Creates an Env for retrieval testing with REAL embeddings.
    Uses module scope so the indexing only happens once for all tests.
    """
    project_root = Path(__file__).parent.parent
    chroma_dir = project_root / "tmp" / "chroma_db_retrieval"
    fixtures_dir = project_root / "tests" / "fixtures" / "data"

    return Env(
        paths=Paths(
            data_dir=fixtures_dir,
            raw_talks_dir=fixtures_dir / "raw_talks",
            chroma_db_dir=chroma_dir,
            metadata_path=fixtures_dir / "metadata.json"
        ),
        rag=RAG(chunk_size=500, chunk_overlap=0),
        io=IO(create_missing_dirs=True),
        # Use REAL embeddings for retrieval tests
        models=Models(embedding_model="all-MiniLM-L6-v2")
    )


@pytest.fixture(scope="module")
def indexed_collection(retrieval_env: Env):
    """
    Runs the indexer once and returns the collection for all tests to use.
    Module scope means this only runs once for the entire test file.
    """
    print("\n--- Running indexer for retrieval tests ---")

    # DELETE the database directory to clear any previous collections with wrong dimensions
    chroma_dir = retrieval_env.paths.chroma_db_dir
    if chroma_dir.exists():
        print(f"Deleting existing database at {chroma_dir}")
        shutil.rmtree(chroma_dir)

    run_indexer(retrieval_env)

    connector = ChromaConnector(retrieval_env)
    ef = get_embedding_function(retrieval_env.models.embedding_model)
    collection = connector.get_collection("rob_burbea_talks", embedding_function=ef)

    print(f"Collection has {collection.count()} chunks indexed")
    return collection


# =============================================================================
# Embedding Determinism Tests
# =============================================================================

def test_embedding_function_is_deterministic(retrieval_env: Env):
    """
    Verifies that the same text produces the same embedding vector.
    This is crucial for reproducible results.
    """
    ef = get_embedding_function(retrieval_env.models.embedding_model)

    test_text = "The practice of jhana meditation brings deep concentration."

    # Generate embedding twice
    embedding1 = ef([test_text])[0]
    embedding2 = ef([test_text])[0]

    # They should be identical (use numpy comparison for arrays)
    assert np.array_equal(embedding1, embedding2), "Embeddings should be deterministic"

    # Verify it's a proper vector (list of floats or numpy array)
    assert len(embedding1) > 0
    assert all(isinstance(x, (float, np.floating)) for x in embedding1)


def test_different_texts_produce_different_embeddings(retrieval_env: Env):
    """
    Verifies that different texts produce different embedding vectors.
    """
    ef = get_embedding_function(retrieval_env.models.embedding_model)

    text1 = "The practice of jhana meditation brings deep concentration."
    text2 = "Cooking pasta requires boiling water and salt."

    embedding1 = ef([text1])[0]
    embedding2 = ef([text2])[0]

    # They should be different (use numpy comparison for arrays)
    assert not np.array_equal(embedding1, embedding2), "Different texts should have different embeddings"


# =============================================================================
# Basic Retrieval Tests
# =============================================================================

def test_query_returns_results(indexed_collection):
    """
    Basic smoke test: Verify that querying returns results.
    """
    results = indexed_collection.query(
        query_texts=["jhana meditation"],
        n_results=5
    )

    # Should return results
    assert len(results["ids"]) > 0
    assert len(results["documents"]) > 0
    assert len(results["metadatas"]) > 0

    # Should return up to 5 results
    assert len(results["ids"][0]) <= 5


def test_query_for_jhana_returns_relevant_chunks(indexed_collection):
    """
    Tests that querying for 'jhana' returns chunks that actually mention jhana.
    """
    results = indexed_collection.query(
        query_texts=["jhana"],
        n_results=10
    )

    documents = results["documents"][0]

    # At least some of the top results should mention "jhana" or related terms
    relevant_count = sum(
        1 for doc in documents
        if "jhana" in doc.lower() or "jhāna" in doc.lower()
    )

    assert relevant_count > 0, "Should find chunks mentioning 'jhana'"


def test_query_for_breath_returns_relevant_chunks(indexed_collection):
    """
    Tests that querying for breathing-related terms returns relevant chunks.
    """
    results = indexed_collection.query(
        query_texts=["breathing meditation energy body"],
        n_results=10
    )

    documents = results["documents"][0]

    # Check if results mention breathing/breath/energy
    relevant_count = sum(
        1 for doc in documents
        if any(term in doc.lower() for term in ["breath", "breathing", "energy body"])
    )

    assert relevant_count > 0, "Should find chunks about breathing"


# =============================================================================
# Metadata Tests
# =============================================================================

def test_results_include_source_metadata(indexed_collection):
    """
    Verifies that retrieved chunks include source file information.
    """
    results = indexed_collection.query(
        query_texts=["meditation"],
        n_results=3
    )

    metadatas = results["metadatas"][0]

    # Every result should have metadata
    assert len(metadatas) > 0

    # Every metadata should have a 'source' field
    for meta in metadatas:
        assert "source" in meta
        assert isinstance(meta["source"], str)
        assert ".md" in meta["source"]


def test_results_include_chunk_ids(indexed_collection):
    """
    Verifies that chunks have deterministic IDs in format: filename_index
    """
    results = indexed_collection.query(
        query_texts=["jhana"],
        n_results=5
    )

    ids = results["ids"][0]

    # Every ID should follow the pattern: some-filename_0, some-filename_1, etc.
    for chunk_id in ids:
        assert "_" in chunk_id, f"ID should contain underscore: {chunk_id}"
        parts = chunk_id.split("_")
        # Last part should be a number
        assert parts[-1].isdigit(), f"Last part should be numeric index: {chunk_id}"


# =============================================================================
# Similarity Score Tests
# =============================================================================

def test_results_include_distances(indexed_collection):
    """
    Verifies that query results include similarity distances.
    """
    results = indexed_collection.query(
        query_texts=["jhana"],
        n_results=5
    )

    # ChromaDB returns 'distances' (lower is more similar for cosine)
    assert "distances" in results
    assert len(results["distances"][0]) > 0

    distances = results["distances"][0]

    # All distances should be floats
    assert all(isinstance(d, float) for d in distances)

    # For cosine distance, values should be between 0 and 2
    # (0 = identical, 1 = perpendicular, 2 = opposite)
    assert all(0 <= d <= 2 for d in distances)


def test_results_are_ordered_by_relevance(indexed_collection):
    """
    Verifies that results are ordered by similarity (closest first).
    For cosine distance, this means ascending order (smaller = more similar).
    """
    results = indexed_collection.query(
        query_texts=["jhana meditation practice"],
        n_results=10
    )

    distances = results["distances"][0]

    # Distances should be in ascending order (most similar first)
    assert distances == sorted(distances), "Results should be ordered by similarity"


# =============================================================================
# Multiple Query Test
# =============================================================================

def test_different_queries_return_different_results(indexed_collection):
    """
    Verifies that different queries return different top results.
    """
    results1 = indexed_collection.query(
        query_texts=["jhana concentration"],
        n_results=3
    )

    results2 = indexed_collection.query(
        query_texts=["metta loving-kindness"],
        n_results=3
    )

    ids1 = set(results1["ids"][0])
    ids2 = set(results2["ids"][0])

    # The top results should be different
    assert ids1 != ids2, "Different queries should return different results"


# =============================================================================
# Performance Benchmark Test
# =============================================================================

def test_query_performance_benchmark(indexed_collection):
    """
    Simple benchmark to measure query performance.
    Not a pass/fail test, just prints timing information.
    """
    import time

    query_text = "What is the practice of jhana meditation?"
    iterations = 10

    start = time.time()
    for _ in range(iterations):
        indexed_collection.query(
            query_texts=[query_text],
            n_results=10
        )
    end = time.time()

    avg_time = (end - start) / iterations
    print(f"\n--- Query Performance ---")
    print(f"Average query time: {avg_time * 1000:.2f}ms")
    print(f"Queries per second: {1 / avg_time:.1f}")

    # Just a sanity check - queries should be reasonably fast
    assert avg_time < 1.0, "Queries should take less than 1 second"
