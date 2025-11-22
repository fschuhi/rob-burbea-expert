from __future__ import annotations

# noinspection PyUnresolvedReferences
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    A wrapper for sentence-transformers to be compatible with ChromaDB.
    """

    # noinspection PyMissingConstructor
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):  # type: ignore[override]
        self.model_name = model_name
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")
        self.model = SentenceTransformer(model_name)

    # noinspection PyShadowingBuiltins
    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        embeddings = self.model.encode(input)
        return embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings


class FakeEmbeddingFunction:
    """
    Deterministic test double that returns pure Python lists.
    NOTE: We intentionally do NOT inherit from chromadb.api.types.EmbeddingFunction,
    because its injected __call__ normalizes outputs to NumPy arrays.
    Instead, we implement only the minimal surface area Chroma expects.
    """

    _VECTOR = [0.1, 0.2, 0.3]

    @staticmethod
    def name() -> str:
        """Mirror the naming hook Chroma’s validation checks for."""
        return "mock"

    # noinspection PyShadowingBuiltins
    def __call__(self, input: Documents) -> Embeddings:
        # type: ignore
        return [self._VECTOR.copy() for _ in input]

    # noinspection PyShadowingBuiltins
    def embed_query(self, input: Documents) -> Embeddings:
        """
        Alias for __call__ to satisfy Chroma/LangChain interfaces during query.
        """
        return self(input)

    # noinspection PyShadowingBuiltins
    def embed_documents(self, input: Documents) -> Embeddings:
        """
        Alias for __call__ to satisfy Chroma/LangChain interfaces during indexing.
        """
        return self(input)


def get_embedding_function(model_name: str) -> EmbeddingFunction | FakeEmbeddingFunction:
    """
    Factory function.
    Returns a real sentence-transformer embedding function
    or the lightweight fake, depending on the requested model name.
    """
    if model_name == "mock":
        return FakeEmbeddingFunction()

    return SentenceTransformerEmbeddingFunction(model_name=model_name)
