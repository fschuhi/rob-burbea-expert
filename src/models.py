from __future__ import annotations

# noinspection PyUnresolvedReferences,PyProtectedMember
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
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model.
        Args:
            model_name: The name of the model to load from HuggingFace.
        """
        self.model_name = model_name
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is not installed.")

        self.model = SentenceTransformer(model_name)

    # Renamed 'input' to 'docs'
    def __call__(self, docs: Documents) -> Embeddings:
        """
        Generate embeddings for a list of documents.
        """
        # sentence-transformers returns numpy arrays; Chroma expects lists
        embeddings = self.model.encode(docs)
        return embeddings.tolist()


def get_embedding_function(model_name: str) -> EmbeddingFunction:
    """
    Factory function to get the configured embedding function.
    """
    return SentenceTransformerEmbeddingFunction(model_name=model_name)
