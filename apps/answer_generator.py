"""
Answer Generator - RAG-based Question Answering with Ollama.

Features:
- Semantic Search + Reranking
- Simple "Thinking..." Spinner
- Telemetry: Vertical "Log Style" List in Expander
- Cinematic Wide Layout
- Footnote Citation System

Usage:
    make chat
"""

import streamlit as st
import re
import time
from src.env import load_env
from src.engine import RAGEngine
from src.database import reconstruct_paragraph_with_hit


@st.cache_resource
def get_engine():
    """Initialize the RAG Engine (cached)."""
    env = load_env()
    return RAGEngine(env)


def resolve_references(text: str) -> list[int]:
    """Robustly parses citation tags like [1], [1, 2] from the text."""
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


def render_telemetry_box(container, telemetry: dict, final: bool = False):
    """
    Renders the metrics as a clean, vertical log list.
    """
    if not telemetry:
        return

    retrieval = telemetry.get("retrieval", 0.0)
    ttft = telemetry.get("ttft", 0.0)

    # Dynamic header based on state
    if final:
        total_wall = telemetry.get("total_wall", 0.0)
        header = f"⏱️ Finished in {total_wall:.2f}s"
    else:
        header = "⏱️ Processing..."

    # Build the log lines
    lines = []
    lines.append(f"**Retrieval & Rerank:** `{retrieval:.2f}s`")
    lines.append(f"**Time to First Token:** `{ttft:.2f}s`")

    if final:
        total_gen = telemetry.get("total_gen", 0.0)
        speed = telemetry.get("speed", 0.0)
        lines.append(f"**Full Generation:** `{total_gen:.2f}s`")
        lines.append(f"**Speed:** `{speed:.1f} words/s`")

    # Use double-space + newline for tight markdown line breaks (Log Style)
    content = "  \n".join(lines)

    with container.expander(header, expanded=False):
        st.markdown(content)


def main():
    # 1. CINEMATIC VIEW: Use wide layout
    st.set_page_config(page_title="Rob Burbea Expert - AI Chat", page_icon="🧘", layout="wide")

    # 2. CSS: Constrain the wide layout to a comfortable reading width
    st.markdown(
        """
        <style>
        /* Center the main block and limit width for readability */
        .block-container {
            max-width: 1000px;
            padding-top: 2rem;
            padding-bottom: 2rem;
            margin: auto;
        }
        .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }

        /* Compact info boxes in sidebar */
        div[data-testid="stAlert"] { padding: 0.5rem 0.75rem; }
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
        reranker_name = engine.env.models.reranker_model.replace("cross-encoder/", "")
        embedder_name = engine.env.models.embedding_model
        llm_name = engine.env.models.default_llm_model

        st.caption(f"**LLM:** `{llm_name}`")
        st.caption(f"**Reranker:** `{reranker_name}`")
        st.caption(f"**Embedder:** `{embedder_name}`")

        st.markdown("---")
        st.subheader("Tuning")

        top_k = st.slider("Final Context Chunks", 1, 15, engine.env.rag.top_k_results)
        rerank_mult = st.slider(
            "Rerank Scan Depth (x)",
            1,
            10,
            engine.env.rag.rerank_depth_multiplier,
            help=f"Retrieves {top_k} × (this value) candidates from DB.",
        )
        dist_threshold = st.slider("Max Distance (Pre-filter)", 0.0, 2.0, engine.env.rag.similarity_threshold, 0.05)

        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- CHAT HISTORY RENDER LOOP ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # A. Render Telemetry (Static from history)
            if "telemetry" in msg:
                t_box = st.empty()
                render_telemetry_box(t_box, msg["telemetry"], final=True)

            # B. Render Content
            st.markdown(msg["content"])

            # C. Render Citations
            if msg.get("citations"):
                refs_map = msg.get("references_map", {})
                citation_ids = msg["citations"]
                if citation_ids:
                    with st.expander(f"📚 Sources ({len(citation_ids)})"):
                        for ref_id in citation_ids:
                            ref_id = int(ref_id)
                            if ref_id in refs_map:
                                meta = refs_map[ref_id]["metadata"]
                                source_name = meta.get("source", "Unknown").split("/")[-1]
                                para_str = meta.get("paragraph_index", "0")
                                chunk_str = meta.get("chunk_position", "0")

                                try:
                                    human_para = int(para_str) + 1
                                except ValueError:
                                    human_para = "?"

                                st.markdown(f"**[{ref_id}] {source_name} (Para {human_para})**")
                                try:
                                    reconstruction = reconstruct_paragraph_with_hit(
                                        engine.collection,
                                        source=meta["source"],
                                        paragraph_index=para_str,
                                        hit_chunk_position=chunk_str,
                                    )
                                    st.markdown(reconstruction["marked_text"], unsafe_allow_html=True)
                                    st.divider()
                                except Exception as e:
                                    st.caption(f"Error retrieving text: {e}")

    # --- INPUT HANDLING ---
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

            # 1. Create Layout Placeholders
            telemetry_box = st.empty()
            answer_box = st.empty()

            full_response = ""
            references_map = {}
            used_ids = []
            stream = None
            telemetry = {}

            t_start_total = time.time()

            # --- PHASE 1: PROCESSING (Spinner) ---
            with st.spinner("🧠 Thinking..."):
                try:
                    # 1. Retrieval
                    t0 = time.time()
                    context_str, references_map = engine.retrieve_and_rerank(
                        final_prompt,
                        top_k=top_k,
                        distance_threshold=dist_threshold,
                        rerank_depth_multiplier=rerank_mult,
                    )
                    t1 = time.time()
                    telemetry["retrieval"] = t1 - t0

                    # 2. Connection
                    stream = engine.llm_client.stream_answer(query=final_prompt, context=context_str)

                    # 3. First Token (Latency Check)
                    first_chunk = next(stream)
                    t2 = time.time()
                    telemetry["ttft"] = t2 - t1

                except StopIteration:
                    st.error("LLM returned empty response.")
                    st.stop()
                except Exception as e:
                    st.error(f"Pipeline failed: {e}")
                    st.stop()

            # --- PHASE 2: RENDERING ---

            # A. Initial Telemetry (TTFT known)
            render_telemetry_box(telemetry_box, telemetry, final=False)

            # B. Stream
            full_response += first_chunk
            answer_box.markdown(full_response + "▌")

            for chunk in stream:
                full_response += chunk
                answer_box.markdown(full_response + "▌")

            answer_box.markdown(full_response)

            # --- PHASE 3: FINAL METRICS ---
            t_end_total = time.time()

            gen_time = t_end_total - t2
            word_count = len(full_response.split())
            speed = word_count / gen_time if gen_time > 0 else 0

            telemetry["total_gen"] = gen_time
            telemetry["speed"] = speed
            telemetry["total_wall"] = t_end_total - t_start_total

            render_telemetry_box(telemetry_box, telemetry, final=True)

            used_ids = resolve_references(full_response)

        # --- SAVE TO HISTORY ---
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": full_response,
                "references_map": references_map,
                "citations": used_ids,
                "telemetry": telemetry,
            }
        )
        st.rerun()


if __name__ == "__main__":
    main()
