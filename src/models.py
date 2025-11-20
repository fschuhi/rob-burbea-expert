from __future__ import annotations

# noinspection PyUnresolvedReferences,PyProtectedMember
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    A wrapper for sentence-transformers to be compatible with ChromaDB.
    """

    # noinspection PyMissingConstructor
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):  # type: ignore
        self.model_name = model_name
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
        self.model = SentenceTransformer(model_name)

    def __call__(self, docs: Documents) -> Embeddings:
        # Real model encoding
        embeddings = self.model.encode(docs)
        return embeddings.tolist()


class FakeEmbeddingFunction(EmbeddingFunction):
    """
    A deterministic fake for testing.
    Returns consistent vectors without loading any ML models.
    """

    # noinspection PyMissingConstructor
    def __init__(self):  # type: ignore
        pass

    def __call__(self, docs: Documents) -> Embeddings:
        # Return a deterministic 3-dimensional vector for every doc.
        # Using list of lists to match ChromaDB requirement.
        return [[0.1, 0.2, 0.3] for _ in docs]


def get_embedding_function(model_name: str) -> EmbeddingFunction:
    """
    Factory function. Returns a Real or Fake model based on configuration.
    """
    if model_name == "mock":
        return FakeEmbeddingFunction()

    return SentenceTransformerEmbeddingFunction(model_name=model_name)
