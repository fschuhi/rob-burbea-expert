"""
Search Explorer - Semantic search interface for Rob Burbea talks.

Usage:
    streamlit run apps/search_explorer.py
    OR
    make app
"""

import streamlit as st
from pathlib import Path
import shutil

from src.env import Env, Paths, RAG, IO, Models
from src.indexing import run_indexer
from src.database import ChromaConnector
from src.models import get_embedding_function


@st.cache_resource
def get_collection():
    """
    Initialize the database and return the collection.
    Uses Streamlit cache so this only runs once per session.
    """
    # Use the same setup as test_retrieval.py
    project_root = Path(__file__).parent.parent
    chroma_dir = project_root / "tmp" / "chroma_db_retrieval"
    fixtures_dir = project_root / "tests" / "fixtures" / "data"

    env = Env(
        paths=Paths(
            data_dir=fixtures_dir,
            raw_talks_dir=fixtures_dir / "raw_talks",
            chroma_db_dir=chroma_dir,
            metadata_path=fixtures_dir / "metadata.json"
        ),
        rag=RAG(chunk_size=500, chunk_overlap=0),
        io=IO(create_missing_dirs=True),
        models=Models(embedding_model="all-MiniLM-L6-v2")
    )

    # Check if database exists and has data
    connector = ChromaConnector(env)
    ef = get_embedding_function(env.models.embedding_model)
    collection = connector.get_collection("rob_burbea_talks", embedding_function=ef)

    # If empty, index the talks
    if collection.count() == 0:
        st.info("🔄 Database is empty. Indexing 32 talks... (this takes ~20 seconds)")
        with st.spinner("Indexing talks..."):
            run_indexer(env)
        st.success(f"✅ Indexed {collection.count()} chunks!")
        st.rerun()  # Refresh to show the updated state

    return collection, env


def format_source(source_path: str) -> str:
    """Extract a readable filename from the full path."""
    return Path(source_path).stem.replace("-", " ").title()


def main():
    st.set_page_config(
        page_title="Rob Burbea Expert - Search Explorer",
        page_icon="🧘",
        layout="wide"
    )

    st.title("🧘 Rob Burbea Expert - Semantic Search Explorer")
    st.markdown("*Exploring 32 talks from the 'Practising the Jhānas' retreat (2019-2020)*")

    # Initialize collection
    collection, env = get_collection()

    # Show database info
    with st.sidebar:
        st.header("📊 Database Info")
        st.metric("Total Chunks", collection.count())
        st.metric("Embedding Model", env.models.embedding_model)
        st.metric("Chunk Size", env.rag.chunk_size)
        st.info(f"💾 Database: `{env.paths.chroma_db_dir}`")

        st.markdown("---")
        st.header("⚙️ Search Settings")
        n_results = st.slider("Number of results", min_value=1, max_value=50, value=10)
        show_scores = st.checkbox("Show similarity scores", value=True)
        show_chunk_ids = st.checkbox("Show chunk IDs", value=False)

    # Query input
    st.markdown("---")
    query = st.text_input(
        "🔍 Enter your search query:",
        placeholder="e.g., 'energy body practice' or 'what is the first jhana?'",
        help="Search semantically across all 32 talks"
    )

    if query:
        # Perform search
        with st.spinner("Searching..."):
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )

        # Display results
        st.markdown(f"### Found {len(results['ids'][0])} results")
        st.markdown("---")

        for idx, (chunk_id, document, metadata, distance) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
        ), start=1):
            # Result container
            with st.container():
                # Header with source and score
                col1, col2 = st.columns([3, 1])

                with col1:
                    source_file = metadata.get('source', 'Unknown')
                    st.markdown(f"**#{idx} · {format_source(source_file)}**")

                with col2:
                    if show_scores:
                        # Lower distance = more similar (cosine distance)
                        similarity_pct = max(0, (1 - distance) * 100)
                        st.metric("Similarity", f"{similarity_pct:.1f}%")

                # Show chunk ID if enabled
                if show_chunk_ids:
                    st.caption(f"Chunk ID: `{chunk_id}`")

                # Show the chunk text
                st.markdown(f"> {document}")

                # Show raw distance if scores enabled
                if show_scores:
                    st.caption(f"*Distance: {distance:.4f}*")

                st.markdown("---")

        # Summary at the bottom
        st.info(f"💡 Showing top {len(results['ids'][0])} of {collection.count()} total chunks")

    else:
        # Show example queries when no query entered
        st.markdown("### 💭 Example Queries to Try:")
        examples = [
            "energy body meditation",
            "what is the first jhana?",
            "differences between jhanas",
            "breath practice instructions",
            "hindrances to meditation",
            "metta practice",
            "cessation of perception"
        ]

        for example in examples:
            if st.button(f"🔍 {example}", key=example):
                st.rerun()


if __name__ == "__main__":
    main()
