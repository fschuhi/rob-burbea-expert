from __future__ import annotations

import chromadb
from chromadb.api.models.Collection import Collection
from src.env import Env


class ChromaConnector:
    """
    Wraps ChromaDB client handling to ensure consistent configuration
    across the application.
    """

    def __init__(self, env: Env):
        """
        Initialize the ChromaDB client using the path defined in Env.
        We use PersistentClient to ensure data is saved to disk at:
        env.paths.chroma_db_dir
        """
        self.db_path = env.paths.chroma_db_dir
        self.client = chromadb.PersistentClient(path=str(self.db_path))

    def get_collection(self, name: str, embedding_function=None) -> Collection:
        """
        Get or create a ChromaDB collection.
        Args:
            name: The name of the collection (e.g., "rob_burbea_talks").
            embedding_function: The function used to embed text.
                                If None, ChromaDB uses its default (all-MiniLM-L6-v2).
        We allow passing this to support dependency injection (testing).

        Returns:
            A ChromaDB Collection object.
        """
        # We explicitly set cosine distance as it is standard for semantic search
        return self.client.get_or_create_collection(
            name=name, embedding_function=embedding_function, metadata={"hnsw:space": "cosine"}
        )

    def reset(self):
        """
        Resets the database.
        WARNING: This is destructive and mostly useful for testing clean slates.
        """
        self.client.reset()
