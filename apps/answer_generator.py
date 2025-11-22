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


def resolve_references(references_map: dict, text: str) -> list[int]:
    """
    Parses the text for citation tags like [1], [2] and returns
    the list of referenced IDs found.
    """
    # Match [1], [12], **[1]** etc.
    matches = re.findall(r"\[(\d+)\]", text)
    unique_ids = sorted(list(set([int(m) for m in matches])))

    # Filter to only IDs that actually exist in our map
    valid_ids = [i for i in unique_ids if i in references_map]
    return valid_ids


def main():
    st.set_page_config(page_title="Rob Burbea Expert - AI Chat", page_icon="🧘", layout="centered")

    st.markdown("""
        <style>
        .stChatMessage { padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem; }

        /* Styling for the **[1]** citations */
        div[data-testid="stMarkdownContainer"] strong {
            color: #FFD700 !important; /* Gold/Amber */
            font-weight: 900 !important;
        }
        </style>
    """, unsafe_allow_html=True)

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
        st.info(f"Model: `{engine.env.models.default_llm_model}`")
        st.markdown("---")
        st.subheader("Tuning")
        top_k = st.slider("Max Context Chunks", 1, 20, engine.env.rag.top_k_results)
        dist_threshold = st.slider("Max Distance", 0.0, 2.0, engine.env.rag.similarity_threshold, 0.05)
        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- CHAT HISTORY ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            # Render Citations if present
            if msg.get("citations"):
                refs_map = msg.get("references_map", {})
                citation_ids = msg["citations"]

                if citation_ids:
                    st.markdown("---")
                    st.caption("📚 References:")
                    for ref_id in citation_ids:
                        ref_id_int = int(ref_id)  # Ensure int key
                        if ref_id_int in refs_map:
                            meta = refs_map[ref_id_int]['metadata']
                            source_name = meta.get('source', 'Unknown').split('/')[-1]

                            # FIX: Explicit int() cast to handle string metadata from Chroma
                            try:
                                para_idx = int(meta.get('paragraph_index', 0))
                                human_para = para_idx + 1
                            except ValueError:
                                human_para = "?"

                            with st.expander(f"[{ref_id}] {source_name} (Para {human_para})"):
                                try:
                                    reconstruction = reconstruct_paragraph_with_hit(
                                        engine.collection,
                                        source=meta['source'],
                                        paragraph_index=para_idx,
                                        hit_chunk_position=int(meta.get('chunk_position', 0))
                                    )
                                    st.markdown(reconstruction['marked_text'], unsafe_allow_html=True)
                                except:
                                    st.caption("Error retrieving paragraph text.")

    # --- INPUT ---
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        # Use 3 columns for the 3 buttons
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
            placeholder = st.empty()
            full_response = ""
            references_map = {}

            try:
                # Engine now returns references_map
                context_used, stream, references_map = engine.answer_query(
                    final_prompt,
                    top_k=top_k,
                    distance_threshold=dist_threshold
                )

                for chunk in stream:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                # Extract used IDs
                used_ids = resolve_references(references_map, full_response)

                if used_ids:
                    st.markdown("---")
                    st.caption("📚 References:")
                    for ref_id in used_ids:
                        meta = references_map[ref_id]['metadata']
                        source_name = meta.get('source', 'Unknown').split('/')[-1]

                        # FIX: Explicit int() cast here too
                        try:
                            para_idx = int(meta.get('paragraph_index', 0))
                            human_para = para_idx + 1
                        except ValueError:
                            human_para = "?"

                        with st.expander(f"[{ref_id}] {source_name} (Para {human_para})"):
                            try:
                                reconstruction = reconstruct_paragraph_with_hit(
                                    engine.collection,
                                    source=meta['source'],
                                    paragraph_index=para_idx,
                                    hit_chunk_position=int(meta.get('chunk_position', 0))
                                )
                                st.markdown(reconstruction['marked_text'], unsafe_allow_html=True)
                            except:
                                st.write("Text lookup failed.")

            except Exception as e:
                st.error(f"Error: {e}")
                full_response = "Error generating response."
                used_ids = []

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "references_map": references_map,
            "citations": used_ids
        })
        st.rerun()


if __name__ == "__main__":
    main()
