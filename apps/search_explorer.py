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

    # Custom CSS for denser layout (Excel 8pt style)
    st.markdown("""
        <style>
        /* Reduce all spacing and font sizes for dense layout */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        h1 {
            font-size: 1.5rem !important;
            margin-bottom: 0.25rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        .stMarkdown p {
            font-size: 0.75rem !important;
            line-height: 1.3 !important;
            margin-bottom: 0.25rem !important;
        }
        blockquote {
            font-size: 0.75rem !important;
            line-height: 1.3 !important;
            margin: 0.25rem 0 !important;
            padding: 0.25rem 0.5rem !important;
        }
        .stMetric {
            font-size: 0.7rem !important;
        }
        .stMetric label {
            font-size: 0.7rem !important;
        }
        .stMetric [data-testid="stMetricValue"] {
            font-size: 0.9rem !important;
        }
        hr {
            margin: 0.5rem 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧘 Rob Burbea Expert - Semantic Search Explorer")
    st.markdown("*32 talks from 'Practising the Jhānas' (2019-2020)*")

    # Initialize collection
    collection, env = get_collection()

    # Show database info
    with st.sidebar:
        st.header("📊 Database")
        st.metric("Chunks", collection.count())
        st.metric("Model", env.models.embedding_model)
        st.caption(f"DB: `tmp/chroma_db_retrieval`")

        st.markdown("---")
        st.header("⚙️ Settings")

        # Distance threshold slider (lower = more similar)
        distance_threshold = st.slider(
            "Max distance",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.05,
            help="Lower = more similar. Cosine distance: 0=identical, 1=perpendicular, 2=opposite"
        )

        st.caption(f"**Quality guide:**")
        st.caption("• 0.0-0.3: Highly relevant")
        st.caption("• 0.3-0.5: Relevant")
        st.caption("• 0.5-0.8: Loosely related")
        st.caption("• 0.8+: Likely not relevant")

        st.markdown("---")
        show_scores = st.checkbox("Show scores", value=True)
        show_chunk_ids = st.checkbox("Show IDs", value=False)

    # Query input
    st.markdown("---")
    query = st.text_input(
        "🔍 Search:",
        placeholder="e.g., 'energy body practice' or 'what is the first jhana?'",
        help="Semantic search across all talks"
    )

    if query:
        # Perform search - get many results, then filter by distance
        with st.spinner("Searching..."):
            results = collection.query(
                query_texts=[query],
                n_results=100  # Get plenty, filter by distance
            )

        # Filter by distance threshold
        filtered_results = []
        for idx in range(len(results['ids'][0])):
            if results['distances'][0][idx] <= distance_threshold:
                filtered_results.append({
                    'id': results['ids'][0][idx],
                    'document': results['documents'][0][idx],
                    'metadata': results['metadatas'][0][idx],
                    'distance': results['distances'][0][idx]
                })

        # Display filtered results
        if filtered_results:
            st.markdown(f"**{len(filtered_results)} chunks with distance ≤ {distance_threshold}**")
            st.markdown("---")

            for idx, result in enumerate(filtered_results, start=1):
                # Compact result display
                col1, col2 = st.columns([4, 1])

                with col1:
                    source_file = result['metadata'].get('source', 'Unknown')
                    st.markdown(f"**#{idx} · {format_source(source_file)}**")

                with col2:
                    if show_scores:
                        similarity_pct = max(0, (1 - result['distance']) * 100)
                        st.metric("Sim", f"{similarity_pct:.0f}%")

                # Show chunk ID if enabled
                if show_chunk_ids:
                    st.caption(f"`{result['id']}`")

                # Show the chunk text
                st.markdown(f"> {result['document']}")

                # Show raw distance if scores enabled
                if show_scores:
                    st.caption(f"*d={result['distance']:.4f}*")

                st.markdown("---")

            # Summary
            st.caption(f"💡 {len(filtered_results)} of {collection.count()} chunks shown")
        else:
            st.warning(f"No chunks found with distance ≤ {distance_threshold}. Try increasing the threshold.")

    else:
        # Show example queries when no query entered
        st.markdown("**💭 Example Queries:**")
        examples = [
            "energy body meditation",
            "what is the first jhana?",
            "differences between jhanas",
            "breath practice instructions",
            "hindrances to meditation",
            "metta practice",
            "cessation of perception"
        ]

        cols = st.columns(3)
        for idx, example in enumerate(examples):
            with cols[idx % 3]:
                if st.button(f"🔍 {example}", key=example, use_container_width=True):
                    st.rerun()


if __name__ == "__main__":
    main()
