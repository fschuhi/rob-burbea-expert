from __future__ import annotations

from pathlib import Path
from typing import List, Dict

from src.env import Env, load_env
from src.data_prep import prepare_data_pipeline, Document
from src.database import ChromaConnector
from src.models import get_embedding_function


def generate_chunk_id(doc: Document, local_index: int) -> str:
    """
    Generates a deterministic ID based on the filename and its local chunk index.
    Format: <filename_stem>_<local_index>
    Example: 2019-12-17-orienting_0
    """
    source_path = Path(doc.metadata.get("source", "unknown"))
    return f"{source_path.stem}_{local_index}"


def run_indexer(env: Env):
    """
    Main indexing logic, separated for easier testing.
    """
    print(f"--- Rob Burbea Expert Indexer ---")
    print(f"Target Database: {env.paths.chroma_db_dir}")
    print(f"Embedding Model: {env.models.embedding_model}")

    # 1. Initialize Embedding Function
    print("Initializing embedding model...")
    ef = get_embedding_function(env.models.embedding_model)

    # 2. Initialize Database
    connector = ChromaConnector(env)
    collection = connector.get_collection(
        name="rob_burbea_talks",
        embedding_function=ef
    )
    print(f"Connected to collection: {collection.name}")

    # 3. Load and Split Data
    print("Loading and splitting documents...")
    documents = prepare_data_pipeline(env)

    if not documents:
        print("No documents found to index.")
        return

    # 4. Prepare Batch Data with Per-File IDs
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[dict] = []

    # Track index per source file to ensure stable IDs
    source_counters: Dict[str, int] = {}

    print("Preparing batch data...")
    for doc in documents:
        source = str(doc.metadata.get("source", "unknown"))

        # Increment counter for this specific file
        if source not in source_counters:
            source_counters[source] = 0
        else:
            source_counters[source] += 1

        local_index = source_counters[source]

        # Generate ID
        chunk_id = generate_chunk_id(doc, local_index)

        ids.append(chunk_id)
        texts.append(doc.page_content)

        # Ensure metadata values are strings
        clean_meta = {k: str(v) for k, v in doc.metadata.items()}
        metadatas.append(clean_meta)

    # 5. Upsert to ChromaDB
    batch_size = 100
    total_docs = len(documents)

    print(f"Indexing {total_docs} chunks (Batch size: {batch_size})...")

    for i in range(0, total_docs, batch_size):
        end_idx = min(i + batch_size, total_docs)

        batch_ids = ids[i:end_idx]
        batch_texts = texts[i:end_idx]
        batch_metas = metadatas[i:end_idx]

        collection.upsert(
            ids=batch_ids,
            documents=batch_texts,
            metadatas=batch_metas
        )
        print(f"  Processed {end_idx}/{total_docs}")

    print("--- Indexing Complete ---")
    print(f"Total items in collection: {collection.count()}")


def main():
    try:
        env = load_env()
        run_indexer(env)
    except Exception as e:
        print(f"Fatal Error: {e}")


if __name__ == "__main__":
    main()
