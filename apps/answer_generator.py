"""
Answer Generator - RAG-based Question Answering with Ollama.

Usage:
    streamlit run apps/answer_generator.py
"""

import streamlit as st
import time
from src.env import load_env
from src.engine import RAGEngine


@st.cache_resource
def get_engine():
    """
    Initialize the RAG Engine.
    Cached to prevent reloading embeddings/DB connection on every interaction.
    """
    env = load_env()
    return RAGEngine(env)


def main():
    st.set_page_config(
        page_title="Rob Burbea Expert - AI Chat",
        page_icon="🧘",
        layout="centered"  # Centered often looks better for chat interfaces
    )

    # Custom CSS to make the chat look a bit cleaner
    st.markdown("""
        <style>
        .stChatMessage {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        /* Style the source expander to be subtle */
        .streamlit-expanderHeader {
            font-size: 0.85rem;
            color: #666;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧘 Rob Burbea Expert AI")
    st.caption("Ask questions based on the 'Practising the Jhānas' retreat.")

    # Initialize Engine
    try:
        engine = get_engine()
    except Exception as e:
        st.error(f"Failed to initialize RAG Engine: {e}")
        st.stop()

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Settings")
        st.metric("Database", "Rob Burbea Talks")

        # We can add model selection later if we expose it in the Engine
        st.success(f"Model: `{engine.env.models.default_llm_model}`")

        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # If there is context saved with the message, show it
            if "context" in msg:
                with st.expander("📚 View Context Used"):
                    st.markdown(msg["context"])

    # Chat Input
    if prompt := st.chat_input("Ask a question (e.g., 'How do I work with the energy body?')..."):

        # 1. Display User Message
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 2. Generate Assistant Response
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            context_used = ""

            try:
                # Call the Engine
                # We get the context string immediately, and a generator for the answer
                context_used, stream = engine.answer_query(prompt)

                # Stream the response
                for chunk in stream:
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")

                placeholder.markdown(full_response)

                # Show Sources immediately after generation
                with st.expander("📚 View Context Used"):
                    st.markdown(context_used)

            except Exception as e:
                # Handle Ollama connection errors gracefully
                st.error(f"Error generating response: {e}")
                if "Connection refused" in str(e):
                    st.info("💡 Is Ollama running? Try running `ollama serve` in a terminal.")
                full_response = "I encountered an error."

        # 3. Save Assistant Message to History
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "context": context_used
        })


if __name__ == "__main__":
    main()
