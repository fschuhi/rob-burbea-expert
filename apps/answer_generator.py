"""
Answer Generator - RAG-based Question Answering with Ollama.

Usage:
    make chat
"""

import streamlit as st
from src.env import load_env
from src.engine import RAGEngine


@st.cache_resource
def get_engine():
    """Initialize the RAG Engine (cached)."""
    env = load_env()
    return RAGEngine(env)


def main():
    st.set_page_config(
        page_title="Rob Burbea Expert - AI Chat",
        page_icon="🧘",
        layout="centered"
    )

    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .streamlit-expanderHeader {
            font-size: 0.85rem;
            color: #666;
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

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.header("⚙️ Settings")
        st.metric("Database", "Rob Burbea Talks")

        # Model Info
        st.info(f"Model: `{engine.env.models.default_llm_model}`")
        st.markdown("---")

        # RAG Tuning Controls
        st.subheader("Tuning")

        # 1. Top K (Max Chunks) - Default from TOML
        top_k = st.slider(
            "Max Context Chunks",
            min_value=1,
            max_value=20,
            value=engine.env.rag.top_k_results,
            help="How many paragraph chunks to retrieve for the LLM."
        )

        # 2. Distance Threshold - Default from TOML
        dist_threshold = st.slider(
            "Max Distance",
            min_value=0.0,
            max_value=2.0,
            value=engine.env.rag.similarity_threshold,
            step=0.05,
            help="Higher = Less strict matching (more context, but maybe less relevant)."
        )

        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- CHAT INTERFACE ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "context" in msg:
                with st.expander("📚 View Context Used"):
                    st.markdown(msg["context"])

    if prompt := st.chat_input("Ask a question..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            context_used = ""

            try:
                # Pass the sidebar values to the engine
                context_used, stream = engine.answer_query(
                    prompt,
                    top_k=top_k,
                    distance_threshold=dist_threshold
                )

                for chunk in stream:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                with st.expander("📚 View Context Used"):
                    st.markdown(context_used)

            except Exception as e:
                st.error(f"Error generating response: {e}")
                full_response = "I encountered an error."

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "context": context_used
        })


if __name__ == "__main__":
    main()
