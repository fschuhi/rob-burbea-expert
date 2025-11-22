"""
Answer Generator - RAG-based Question Answering with Ollama.

Usage:
    make chat
"""

import streamlit as st
import re
from src.env import load_env
from src.engine import RAGEngine
from src.database import reconstruct_paragraph_with_hit


@st.cache_resource
def get_engine():
    """Initialize the RAG Engine (cached)."""
    env = load_env()
    return RAGEngine(env)


def resolve_references(text: str) -> list[int]:
    """
    Robustly parses citation tags like [1], [1, 2], [8-10] from the text.
    Returns a list of integer IDs.
    """
    raw_matches = re.findall(r"\[([\d,\s\-]+)\]", text)
    unique_ids = set()

    for match in raw_matches:
        parts = match.split(",")
        for part in parts:
            part = part.strip()
            if "-" in part:
                try:
                    start, end = map(int, part.split("-"))
                    if end - start < 20:
                        unique_ids.update(range(start, end + 1))
                except ValueError:
                    continue
            else:
                try:
                    unique_ids.add(int(part))
                except ValueError:
                    continue

    return sorted(list(unique_ids))


def main():
    st.set_page_config(page_title="Rob Burbea Expert - AI Chat", page_icon="🧘", layout="centered")

    st.markdown(
        """
        <style>
        .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }

        /* Styling for the **[1]** citations - Global Strong Override */
        strong {
            color: #FFD700 !important;
            font-weight: 900 !important;
        }

        /* Compact info boxes in sidebar */
        div[data-testid="stAlert"] {
            padding: 0.5rem 0.75rem;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("🧘 Rob Burbea Expert AI")
    st.caption("Ask questions based on the 'Practising the Jhānas' retreat.")

    try:
        engine = get_engine()
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.metric("Database", "Rob Burbea Talks")

        st.subheader("Model Stack")
        # Clean up names for display
        reranker_name = engine.env.models.reranker_model.replace("cross-encoder/", "")
        embedder_name = engine.env.models.embedding_model
        llm_name = engine.env.models.default_llm_model

        st.caption(f"**LLM:** `{llm_name}`")
        st.caption(f"**Reranker:** `{reranker_name}`")
        st.caption(f"**Embedder:** `{embedder_name}`")

        st.markdown("---")
        st.subheader("Tuning")

        top_k = st.slider(
            "Final Context Chunks",
            min_value=1,
            max_value=15,
            value=engine.env.rag.top_k_results,
            help="How many chunks are actually sent to the LLM.",
        )

        rerank_mult = st.slider(
            "Rerank Scan Depth (x)",
            min_value=1,
            max_value=10,
            value=engine.env.rag.rerank_depth_multiplier,
            help=f"Retrieves {top_k} × (this value) candidates from DB, then selects the best {top_k}.",
        )

        dist_threshold = st.slider(
            "Max Distance (Pre-filter)",
            min_value=0.0,
            max_value=2.0,
            value=engine.env.rag.similarity_threshold,
            step=0.05,
            help="Initial cutoff before reranking. Higher = More candidates allowed.",
        )

        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- CHAT HISTORY ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render Citations
            if msg.get("citations"):
                refs_map = msg.get("references_map", {})
                citation_ids = msg["citations"]

                if citation_ids:
                    st.markdown("---")
                    st.caption("📚 References:")
                    for ref_id in citation_ids:
                        ref_id = int(ref_id)
                        if ref_id in refs_map:
                            meta = refs_map[ref_id]["metadata"]
                            source_name = meta.get("source", "Unknown").split("/")[-1]

                            # FIX: Keep raw strings for DB lookup, use int only for Display
                            para_str = meta.get("paragraph_index", "0")
                            chunk_str = meta.get("chunk_position", "0")

                            try:
                                human_para = int(para_str) + 1
                            except ValueError:
                                human_para = "?"

                            with st.expander(f"[{ref_id}] {source_name} (Para {human_para})"):
                                try:
                                    reconstruction = reconstruct_paragraph_with_hit(
                                        engine.collection,
                                        source=meta["source"],
                                        paragraph_index=para_str,
                                        hit_chunk_position=chunk_str,
                                    )
                                    st.markdown(reconstruction["marked_text"], unsafe_allow_html=True)
                                except Exception as e:
                                    st.caption(f"Error retrieving text: {e}")
                        else:
                            st.warning(f"⚠️ Reference [{ref_id}] cited but not found.")

    # --- INPUT ---
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        col1, col2, col3 = st.columns(3)
        if col1.button("How do I work with the energy body?"):
            st.session_state.example_input = "How do I work with the energy body?"
        if col2.button("What is the role of pīti?"):
            st.session_state.example_input = "What is the role of pīti?"
        if col3.button("Explain the relationship between the energy body and light."):
            st.session_state.example_input = "Explain the relationship between the energy body and light."

    default_input = st.session_state.get("example_input", "")
    if "example_input" in st.session_state:
        del st.session_state["example_input"]

    prompt = st.chat_input("Ask a question...", key="chat_input")
    final_prompt = prompt or (default_input if default_input else None)

    if final_prompt:
        st.chat_message("user").markdown(final_prompt)
        st.session_state.messages.append({"role": "user", "content": final_prompt})

        with st.chat_message("assistant"):

            status_box = st.empty()
            answer_placeholder = st.empty()

            full_response = ""
            references_map = {}
            used_ids = []
            stream = None

            # --- PHASE 1: RETRIEVAL & SETUP ---
            with status_box.status("🧠 Thinking...", expanded=True) as status:
                try:
                    # Step 1: Retrieval with variable depth
                    # Calculate total candidates for UI feedback
                    total_candidates = top_k * rerank_mult
                    st.write(f"🔍 Scanning {total_candidates} candidates...")

                    context_str, references_map = engine.retrieve_and_rerank(
                        final_prompt,
                        top_k=top_k,
                        distance_threshold=dist_threshold,
                        rerank_depth_multiplier=rerank_mult,  # Pass the new slider value
                    )

                    st.write("⚖️ Reranking complete...")
                    st.write("✍️ Connecting to LLM...")
                    stream = engine.llm_client.stream_answer(query=final_prompt, context=context_str)

                    status.update(label="✍️ Generating answer...", state="running", expanded=True)

                except Exception as e:
                    status.update(label="Error", state="error")
                    st.error(f"Pipeline failed: {e}")
                    st.stop()

            # --- PHASE 2: WAIT FOR FIRST TOKEN ---
            if stream:
                try:
                    first_chunk = next(stream)
                    status_box.empty()

                    full_response += first_chunk
                    answer_placeholder.markdown(full_response + "▌")

                    for chunk in stream:
                        full_response += chunk
                        answer_placeholder.markdown(full_response + "▌")

                    answer_placeholder.markdown(full_response)

                    # 4. REFERENCE RESOLUTION
                    used_ids = resolve_references(full_response)

                    if used_ids:
                        st.markdown("---")
                        st.caption("📚 References:")
                        for ref_id in used_ids:
                            if ref_id in references_map:
                                meta = references_map[ref_id]["metadata"]
                                source_name = meta.get("source", "Unknown").split("/")[-1]

                                para_str = meta.get("paragraph_index", "0")
                                chunk_str = meta.get("chunk_position", "0")

                                try:
                                    human_para = int(para_str) + 1
                                except ValueError:
                                    human_para = "?"

                                with st.expander(f"[{ref_id}] {source_name} (Para {human_para})"):
                                    try:
                                        reconstruction = reconstruct_paragraph_with_hit(
                                            engine.collection,
                                            source=meta["source"],
                                            paragraph_index=para_str,
                                            hit_chunk_position=chunk_str,
                                        )
                                        st.markdown(reconstruction["marked_text"], unsafe_allow_html=True)
                                    except Exception as e:
                                        st.caption(f"Error retrieving text: {e}")
                            else:
                                st.warning(f"⚠️ Reference [{ref_id}] cited but not found.")

                except StopIteration:
                    status_box.empty()
                    answer_placeholder.markdown("No response generated.")

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response, "references_map": references_map, "citations": used_ids}
        )
        st.rerun()


if __name__ == "__main__":
    main()
