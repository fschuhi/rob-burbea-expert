"""
Search Explorer - Semantic search interface for Rob Burbea talks.

Features:
- Semantic Vector Search (ChromaDB)
- Dynamic "Max Results" Slider (Parity Party)
- Full Paragraph Reconstruction with Highlights
- Production Environment Loading

Usage:
    make app
"""

import streamlit as st
import markdown
from pathlib import Path

from src.env import load_env
from src.database import ChromaConnector, reconstruct_paragraph_with_hit
from src.models import get_embedding_function


@st.cache_resource
def get_collection():
    """
    Initialize the database connection using the project configuration.
    Uses Streamlit cache to avoid reloading the heavy embedding model.
    """
    # Load the real environment (rb_expert.toml)
    env = load_env()

    # Connect to the Production Database
    connector = ChromaConnector(env)
    ef = get_embedding_function(env.models.embedding_model)
    collection = connector.get_collection("rob_burbea_talks", embedding_function=ef)

    return collection, env


def format_source(source_path: str) -> str:
    """Extract a readable filename from the full path."""
    return Path(source_path).stem.replace("-", " ").title()


def main():
    st.set_page_config(page_title="Rob Burbea Expert - Search Explorer", page_icon="🧘", layout="wide")

    # Custom CSS for the "Card" look
    st.markdown(
        """
        <style>
        /* 1. Global Layout Tightening */
        .block-container {
            padding-top: 3rem;
            padding-bottom: 2rem;
        }

        /* 2. Dense Headers */
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
            margin: 0.15rem 0 !important;
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
    """,
        unsafe_allow_html=True,
    )

    st.title("🧘 Rob Burbea Expert - Semantic Search Explorer")

    try:
        collection, env = get_collection()
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📊 Database")
        st.metric("Total Chunks", collection.count())

        st.caption(f"**Embedder:** `{env.models.embedding_model}`")
        st.caption(f"**Path:** `{env.paths.chroma_db_dir.name}`")

        st.markdown("---")
        st.header("⚙️ Tuning")

        # PARITY PARTY: Added Max Results slider
        n_results = st.slider(
            "Max Results",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="How many chunks to retrieve from the vector database.",
        )

        distance_threshold = st.slider(
            "Max Distance",
            min_value=0.0,
            max_value=2.0,
            value=env.rag.similarity_threshold,  # Use config default
            step=0.05,
            help="Lower = more similar. Cosine distance: 0=identical, 1=unrelated.",
        )

        st.caption("• 0.0-0.7: Relevant\n• 0.7-1.0: Loosely related")
        st.markdown("---")
        show_full_paragraph = st.checkbox("Show full paragraph", value=True)
        show_chunk_ids = st.checkbox("Show IDs", value=False)
        show_chunk_debug = st.checkbox("Show debug info", value=False)

    # --- QUERY HANDLING ---
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    query = st.text_input(
        "🔍 Search:",
        value=st.session_state.search_query,
        placeholder="e.g., 'energy body practice' or 'what is the first jhana?'",
    )

    if query != st.session_state.search_query:
        st.session_state.search_query = query

    if query:
        with st.spinner("Searching..."):
            # Use the dynamic slider value for n_results
            results = collection.query(query_texts=[query], n_results=n_results)

        # Filter results by distance
        filtered_results = []
        if results["ids"]:
            # Access [0] because we sent a single query
            for idx in range(len(results["ids"][0])):
                dist = results["distances"][0][idx]
                if dist <= distance_threshold:
                    filtered_results.append(
                        {
                            "id": results["ids"][0][idx],
                            "document": results["documents"][0][idx],
                            "metadata": results["metadatas"][0][idx],
                            "distance": dist,
                        }
                    )

        if filtered_results:
            st.markdown(f"Found **{len(filtered_results)}** chunks (Distance ≤ {distance_threshold})")
            st.markdown("---")

            for idx, result in enumerate(filtered_results, start=1):
                display_text = result["document"]
                reconstruction_error = None

                # Attempt to reconstruct full paragraph context
                if show_full_paragraph:
                    try:
                        source = result["metadata"].get("source")
                        # Convert to int if stored as float/str in DB
                        para_idx = int(result["metadata"].get("paragraph_index", -1))
                        chunk_pos = int(result["metadata"].get("chunk_position", -1))

                        if source is not None and para_idx >= 0 and chunk_pos >= 0:
                            reconstruction = reconstruct_paragraph_with_hit(
                                collection, source=source, paragraph_index=para_idx, hit_chunk_position=chunk_pos
                            )
                            display_text = reconstruction["marked_text"]
                        else:
                            reconstruction_error = "⚠️ Missing metadata for reconstruction"
                    except Exception as e:
                        # Fallback to raw text if reconstruction fails
                        reconstruction_error = f"⚠️ Reconstruction failed: {str(e)}"

                # HTML Formatting
                html_content = markdown.markdown(display_text)
                # Strip <p> tags for tighter layout
                html_content = html_content.replace("<p>", "").replace("</p>", "")

                source_file = result["metadata"].get("source", "Unknown")

                # Optional Debug info
                chunk_id_div = ""
                if show_chunk_ids:
                    chunk_id_div = f'<div style="font-size: 0.7rem; color: #666; margin-bottom: 2px; font-family: monospace;">ID: {result["id"]}</div>'

                # Card HTML Construction
                card_html = (
                    f'<div class="result-card">'
                    f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">'
                    # Left: Filename
                    f'<span style="font-weight: bold; font-size: 1rem;">#{idx} · {format_source(source_file)}</span>'
                    # Right: Distance Score
                    f'<span style="font-weight: bold; font-size: 1rem; color: #888;">{result["distance"]:.4f}</span>'
                    f"</div>"
                    f"{chunk_id_div}"
                    # Content Body
                    f'<div style="margin-top: 2px; margin-bottom: 0; padding: 0.25rem 0.75rem; border-left: 3px solid #444; font-size: 0.9rem; line-height: 1.5;">'
                    f"{html_content}"
                    f"</div>"
                    f"</div>"
                )

                st.markdown(card_html, unsafe_allow_html=True)

                if reconstruction_error and show_chunk_debug:
                    st.caption(reconstruction_error)

                if show_chunk_debug:
                    st.json(result["metadata"])

                st.markdown("---")
        else:
            st.warning(f"No chunks found with distance ≤ {distance_threshold}. Try increasing the Max Distance.")

    else:
        # Landing Page Examples
        st.markdown("**💭 Example Queries:**")
        examples = [
            "energy body meditation",
            "what is the first jhana?",
            "differences between jhanas",
            "metta practice",
            "cessation of perception",
        ]

        cols = st.columns(3)
        for idx, example in enumerate(examples):
            with cols[idx % 3]:
                if st.button(f"🔍 {example}", key=f"btn_{idx}", use_container_width=True):
                    st.session_state.search_query = example
                    st.rerun()


if __name__ == "__main__":
    main()
