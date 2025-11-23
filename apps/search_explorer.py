"""
Search Explorer - Semantic search interface for Rob Burbea talks.

Features:
- Inspect Raw Vector Search (Distance-based)
- Inspect Reranked Results (Score-based)
- Toggle between the two views to "debug" the Reranker's impact
- Production Environment Loading

Usage:
    make app
"""

import streamlit as st
import markdown
from pathlib import Path

from src.env import load_env
from src.engine import RAGEngine
from src.database import reconstruct_paragraph_with_hit


@st.cache_resource
def get_engine():
    """
    Initialize the RAG Engine (cached).
    We use the Engine here (instead of just the collection) so we can access
    the shared CrossEncoder model without reloading it from disk.
    """
    env = load_env()
    return RAGEngine(env)


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

    st.title("🧘 Rob Burbea Expert - Search Explorer")

    try:
        engine = get_engine()
        collection = engine.collection
    except Exception as e:
        st.error(f"Failed to load database: {e}")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("📊 Database")
        st.metric("Total Chunks", collection.count())

        st.caption(f"**Embedder:** `{engine.env.models.embedding_model}`")
        # Added Reranker Info below Embedder
        reranker_name = engine.env.models.reranker_model.replace("cross-encoder/", "")
        st.caption(f"**Reranker:** `{reranker_name}`")

        st.markdown("---")
        st.header("⚙️ Tuning")

        # 1. The Pool Size (Chunks to fetch from DB)
        n_results = st.slider(
            "Retrieval Pool Size",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            help="How many chunks to fetch from the Vector Database.",
        )

        # 2. The Reranker Switch
        use_reranker = st.checkbox(
            "Apply Reranker",
            value=False,
            help="Sort results by Semantic Relevance (Cross-Encoder) instead of Vector Distance.",
        )

        if use_reranker:
            st.success(f"Re-ordering the top **{n_results}** chunks by meaning.")
        else:
            st.info("Showing raw database order (Distance).")

        st.markdown("---")

        # 3. Distance Filter (Renamed)
        distance_threshold = st.slider(
            "Max Distance",
            min_value=0.0,
            max_value=2.0,
            value=engine.env.rag.similarity_threshold,
            step=0.05,
            help="Chunks beyond this distance are visually dimmed or flagged.",
        )

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
            # 1. RAW RETRIEVAL
            results = collection.query(query_texts=[query], n_results=n_results)

        if not results["ids"]:
            st.warning("No results found.")
            st.stop()

        # Unpack Chroma structure
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        # Convert to list of dicts for easier handling
        candidates = []
        for i in range(len(ids)):
            candidates.append(
                {
                    "id": ids[i],
                    "document": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                    "rank_score": 0.0,  # Placeholder
                }
            )

        # 2. OPTIONAL RERANKING
        if use_reranker:
            with st.spinner("Reranking..."):
                pairs = [[query, c["document"]] for c in candidates]
                scores = engine.cross_encoder.predict(pairs)

                for i, candidate in enumerate(candidates):
                    candidate["rank_score"] = scores[i]

                # Sort by Score (Descending)
                candidates.sort(key=lambda x: x["rank_score"], reverse=True)
        else:
            # Keep DB order (Distance Ascending)
            pass

        # 3. DISPLAY LOOP
        st.markdown(f"Showing **{len(candidates)}** chunks")
        st.markdown("---")

        for idx, result in enumerate(candidates, start=1):
            display_text = result["document"]
            reconstruction_error = None

            # --- Reconstruction Logic ---
            if show_full_paragraph:
                try:
                    source = result["metadata"].get("source")
                    para_idx = result["metadata"].get("paragraph_index")
                    chunk_pos = result["metadata"].get("chunk_position")

                    # Robust checks for metadata (handle string vs int)
                    if source is not None and para_idx is not None and chunk_pos is not None:
                        reconstruction = reconstruct_paragraph_with_hit(
                            collection, source=source, paragraph_index=para_idx, hit_chunk_position=chunk_pos
                        )
                        display_text = reconstruction["marked_text"]
                    else:
                        reconstruction_error = "⚠️ Missing metadata for reconstruction"
                except Exception as e:
                    reconstruction_error = f"⚠️ Reconstruction failed: {str(e)}"

            # --- HTML Formatting ---
            html_content = markdown.markdown(display_text)
            html_content = html_content.replace("<p>", "").replace("</p>", "")
            source_file = result["metadata"].get("source", "Unknown")

            # --- Header Metrics ---
            if use_reranker:
                # Show SCORE as primary metric
                metric_html = f'<span style="font-weight: bold; font-size: 1rem; color: #4CAF50;">Score: {result["rank_score"]:.2f}</span>'
                sub_metric = f'<span style="color: #888; font-size: 0.8rem; margin-left: 10px;">(Dist: {result["distance"]:.3f})</span>'
            else:
                # Show DISTANCE as primary metric
                # Highlight good distances in green, bad in grey
                color = "#4CAF50" if result["distance"] <= distance_threshold else "#888"
                metric_html = f'<span style="font-weight: bold; font-size: 1rem; color: {color};">Dist: {result["distance"]:.4f}</span>'
                sub_metric = ""

            # --- Debug Info ---
            chunk_id_div = ""
            if show_chunk_ids:
                chunk_id_div = f'<div style="font-size: 0.7rem; color: #666; margin-bottom: 2px; font-family: monospace;">ID: {result["id"]}</div>'

            # --- Card Construction ---
            card_html = (
                f'<div class="result-card">'
                f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px;">'
                # Left: Index + Filename
                f'<span style="font-weight: bold; font-size: 1rem;">#{idx} · {format_source(source_file)}</span>'
                # Right: Metric
                f"<div>{metric_html}{sub_metric}</div>"
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
        # Landing Page
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
