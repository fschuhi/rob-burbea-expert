"""
Search Explorer - Semantic search interface for Rob Burbea talks.

Usage:
    streamlit run apps/search_explorer.py
    OR
    make app
"""

import streamlit as st
from pathlib import Path

from src.env import Env, Paths, RAG, IO, Models
from src.indexing import run_indexer
from src.database import ChromaConnector, reconstruct_paragraph_with_hit
from src.models import get_embedding_function


@st.cache_resource
def get_collection():
    """
    Initialize the database and return the collection.
    Uses Streamlit cache so this only runs once per session.
    """
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

    # Custom CSS
    st.markdown("""
        <style>
        /* 1. Global Layout Tightening */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 2rem;
        }

        /* 2. Dense Headers (Restored from your original) */
        h1 {
            font-size: 1.5rem !important;
            margin-bottom: 0.25rem !important;
            padding-top: 1.0rem !important;
            line-height: 1.5 !important;
        }
        h3 {
            font-size: 1.1rem !important;
            margin-top: 0.5rem !important;
            margin-bottom: 0.5rem !important;
        }

        /* 3. Tighten Horizontal Lines */
        hr {
            margin: 0.15rem 0 !important; /* Very tight vertical spacing */
        }

        /* 4. Result Card Styling */
        .result-card {
            font-family: sans-serif;
            margin-bottom: 0px;
        }

        /* 5. The Custom Hit Highlight Tag */
        hit {
            color: #7CFC00 !important;
            background-color: rgba(124, 252, 0, 0.1);
            font-weight: normal;
            border-radius: 3px;
            padding: 0 2px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧘 Rob Burbea Expert - Semantic Search Explorer")
    st.markdown("*32 talks from 'Practising the Jhānas' (2019-2020)*")

    # Initialize collection
    collection, env = get_collection()

    # Sidebar Settings
    with st.sidebar:
        st.header("📊 Database")
        st.metric("Chunks", collection.count())
        st.caption(f"DB: `tmp/chroma_db_retrieval`")

        st.markdown("---")
        st.header("⚙️ Settings")

        distance_threshold = st.slider(
            "Max distance",
            min_value=0.0,
            max_value=2.0,
            value=0.6,
            step=0.05,
            help="Lower = more similar. Cosine distance: 0=identical"
        )

        st.caption("• 0.0-0.3: Highly relevant\n• 0.3-0.5: Relevant\n• 0.5-0.8: Loosely related")
        st.markdown("---")
        show_full_paragraph = st.checkbox("Show full paragraph", value=True)
        show_chunk_ids = st.checkbox("Show IDs", value=False)
        show_chunk_debug = st.checkbox("Show chunk debug info", value=False)

    # Query Handling
    if 'search_query' not in st.session_state:
        st.session_state.search_query = ""

    st.markdown("---")
    query = st.text_input(
        "🔍 Search:",
        value=st.session_state.search_query,
        placeholder="e.g., 'energy body practice' or 'what is the first jhana?'"
    )

    if query != st.session_state.search_query:
        st.session_state.search_query = query

    if query:
        with st.spinner("Searching..."):
            results = collection.query(
                query_texts=[query],
                n_results=100
            )

        # Filter results
        filtered_results = []
        for idx in range(len(results['ids'][0])):
            if results['distances'][0][idx] <= distance_threshold:
                filtered_results.append({
                    'id': results['ids'][0][idx],
                    'document': results['documents'][0][idx],
                    'metadata': results['metadatas'][0][idx],
                    'distance': results['distances'][0][idx]
                })

        if filtered_results:
            st.markdown(f"**{len(filtered_results)} chunks with distance ≤ {distance_threshold}**")
            st.markdown("---")

            for idx, result in enumerate(filtered_results, start=1):
                display_text = result['document']
                reconstruction_error = None

                if show_full_paragraph:
                    try:
                        source = result['metadata'].get('source')
                        para_idx = result['metadata'].get('paragraph_index')
                        chunk_pos = result['metadata'].get('chunk_position')

                        if source is not None and para_idx is not None and chunk_pos is not None:
                            reconstruction = reconstruct_paragraph_with_hit(
                                collection,
                                source=source,
                                paragraph_index=para_idx,
                                hit_chunk_position=chunk_pos
                            )
                            display_text = reconstruction['marked_text']
                        else:
                            reconstruction_error = "⚠️ Missing metadata"
                    except Exception as e:
                        reconstruction_error = f"⚠️ Reconstruction failed: {str(e)}"

                source_file = result['metadata'].get('source', 'Unknown')

                # Optional: Chunk ID HTML
                chunk_id_div = ""
                if show_chunk_ids:
                    chunk_id_div = f'<div style="font-size: 0.7rem; color: #666; margin-bottom: 2px; font-family: monospace;">ID: {result["id"]}</div>'

                # ---------------------------------------------------------
                # HTML CONSTRUCTION
                # ---------------------------------------------------------
                card_html = (
                    f'<div class="result-card" style="margin-bottom: 0px;">'
                    # Header Row (Flexbox)
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">'
                    # Left: Index + Filename
                    f'<span style="font-weight: bold; font-size: 1rem;">#{idx} · {format_source(source_file)}</span>'
                    # Right: Distance (Bold and same size as title)
                    f'<span style="font-weight: bold; font-size: 1rem;">{result["distance"]:.4f}</span>'
                    f'</div>'

                    # ID (optional)
                    f'{chunk_id_div}'

                    # Content Block (Simulated Blockquote)
                    f'<div style="margin-top: 2px; margin-bottom: 0; padding: 0.25rem 0.75rem; border-left: 3px solid #444; background-color: transparent; font-size: 0.9rem; line-height: 1.5;">'
                    f'{display_text}'
                    f'</div>'
                    f'</div>'
                )

                st.markdown(card_html, unsafe_allow_html=True)

                if reconstruction_error:
                    st.caption(reconstruction_error)

                if show_chunk_debug and 'paragraph_index' in result['metadata']:
                    st.caption(
                        f"🔧 Debug: para_idx={result['metadata']['paragraph_index']}, chunk_pos={result['metadata'].get('chunk_position')}")

                st.markdown("---")

            st.caption(f"💡 {len(filtered_results)} of {collection.count()} chunks shown")
        else:
            st.warning(f"No chunks found with distance ≤ {distance_threshold}.")

    else:
        # Example Queries
        st.markdown("**💭 Example Queries:**")
        examples = [
            "energy body meditation",
            "what is the first jhana?",
            "differences between jhanas",
            "metta practice",
            "cessation of perception"
        ]

        cols = st.columns(3)
        for idx, example in enumerate(examples):
            with cols[idx % 3]:
                if st.button(f"🔍 {example}", key=f"btn_{idx}", use_container_width=True):
                    st.session_state.search_query = example
                    st.rerun()


if __name__ == "__main__":
    main()
