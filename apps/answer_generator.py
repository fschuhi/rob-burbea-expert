"""
Answer Generator - RAG-based Question Answering with Ollama.

Usage:
    make chat
"""

import streamlit as st
import re
from src.env import load_env
from src.engine import RAGEngine


@st.cache_resource
def get_engine():
    """Initialize the RAG Engine (cached)."""
    env = load_env()
    return RAGEngine(env)


def find_context_for_citation(citation_label: str, full_context: str) -> str | None:
    """
    Locates the text block in the full context that matches the citation.
    strategies:
    1. Exact Match.
    2. Text-only Match (ignoring dates).
    3. Date-only Match (handling "2019-12-21-talk.md" hallucinations).
    """
    try:
        # Parsing: Expecting "2019-12-21-filename.md, Para 36"
        parts = citation_label.split(",")
        if len(parts) < 2:
            return None

        filename_fragment = parts[0].strip()
        para_fragment = parts[1].strip()

        para_num_match = re.search(r"\d+", para_fragment)
        if not para_num_match:
            return None
        para_num = para_num_match.group(0)

        # Target Header Stub
        target_para_stub = f"(Paragraph {para_num})"

        # Extract Date from Citation if present (YYYY-MM-DD)
        citation_date_match = re.search(r"(\d{4}-\d{2}-\d{2})", filename_fragment)
        citation_date = citation_date_match.group(1) if citation_date_match else None

        blocks = full_context.split("### Source:")

        for block in blocks:
            if not block.strip():
                continue

            first_line = block.split("\n")[0]

            # Strategy 1: Exact Match
            if filename_fragment in first_line and target_para_stub in first_line:
                return f"### Source:{block}"

            # Strategy 2: Fuzzy Text Match (ignoring date prefix)
            citation_clean = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', filename_fragment)
            header_clean = re.sub(r'^\d{4}-\d{2}-\d{2}-', '', first_line)

            if len(citation_clean) > 5 and citation_clean in header_clean and target_para_stub in first_line:
                return f"### Source:{block}"

            # Strategy 3: Date Match (Backup for hallucinations like '...-talk.md')
            # Only applies if we have a date and the paragraph number matches
            if citation_date and citation_date in first_line and target_para_stub in first_line:
                return f"### Source:{block}"

        return None

    except Exception:
        return None


def process_text_with_footnotes(text: str) -> tuple[str, list[str]]:
    """
    Replaces verbose citations [File, Para] with footnotes [1].
    Returns:
        - clean_text: The text with [1], [2] numbers.
        - citations: The ordered list of original citation strings.
    """
    # Find all citations: bold **[...]** or normal [...]
    # We look for the specific "Para" keyword to avoid capturing other bracketed text
    raw_matches = re.findall(r"(\*\*\[.*?Para.*?\]\*\*|\[.*?Para.*?\])", text)

    citations_map = {}  # Map "original_string" -> Index
    ordered_citations = []
    counter = 1

    clean_text = text

    for match in raw_matches:
        # Strip brackets and bolding for the "Label"
        # e.g. "**[File, Para 1]**" -> "File, Para 1"
        clean_label = match.replace("**", "").replace("[", "").replace("]", "")

        if clean_label not in citations_map:
            citations_map[clean_label] = counter
            ordered_citations.append(clean_label)
            counter += 1

        # Replace in text with **[1]**
        idx = citations_map[clean_label]
        # We escape the match for regex safety in replacement
        safe_match = re.escape(match)
        clean_text = re.sub(safe_match, f"**[{idx}]**", clean_text)

    return clean_text, ordered_citations


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
        /* Make footnotes stand out */
        strong {
            color: #4A90E2;
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
        st.info(f"Model: `{engine.env.models.default_llm_model}`")
        st.markdown("---")

        st.subheader("Tuning")
        top_k = st.slider("Max Context Chunks", 1, 20, engine.env.rag.top_k_results)
        dist_threshold = st.slider("Max Distance", 0.0, 2.0, engine.env.rag.similarity_threshold, 0.05)

        st.markdown("---")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- CHAT INTERFACE ---
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            # If it's an assistant message, we might have processed it already
            # But we re-process here to be safe or just render what we saved
            if msg["role"] == "assistant" and "citations" in msg:
                st.markdown(msg["content"])  # This is the clean text with [1]

                # Render Reference List
                citations = msg["citations"]
                if citations:
                    st.markdown("---")
                    st.caption("📚 References:")
                    for i, label in enumerate(citations, 1):
                        source_text = find_context_for_citation(label, msg.get("context", ""))

                        # Accordion for the source
                        title_short = label.split(',')[0].replace('.md', '')[:40] + "..."
                        with st.expander(f"[{i}] {title_short}"):
                            if source_text:
                                st.markdown(source_text, unsafe_allow_html=True)
                            else:
                                st.caption(f"⚠️ Source text not found: {label}")
            else:
                st.markdown(msg["content"])

    # --- EXAMPLE PROMPTS ---
    if not st.session_state.messages:
        st.markdown("### 💡 Try asking:")
        col1, col2 = st.columns(2)
        if col1.button("How do I work with the energy body?"):
            st.session_state.example_input = "How do I work with the energy body?"
        if col2.button("What is the role of pīti?"):
            st.session_state.example_input = "What is the role of pīti?"

    # --- INPUT HANDLING ---
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
            raw_response = ""
            context_used = ""

            try:
                context_used, stream = engine.answer_query(
                    final_prompt,
                    top_k=top_k,
                    distance_threshold=dist_threshold
                )

                # Streaming Loop (Raw output)
                for chunk in stream:
                    raw_response += chunk
                    placeholder.markdown(raw_response + "▌")

                # Post-Processing: Footnotes
                clean_text, citations = process_text_with_footnotes(raw_response)

                # Final Render (Clean)
                placeholder.markdown(clean_text)

                # Render References immediately
                if citations:
                    st.markdown("---")
                    st.caption("📚 References:")
                    for i, label in enumerate(citations, 1):
                        source_text = find_context_for_citation(label, context_used)
                        title_short = label.split(',')[0].replace('.md', '')[:40] + "..."

                        with st.expander(f"[{i}] {title_short}"):
                            if source_text:
                                st.markdown(source_text, unsafe_allow_html=True)
                            else:
                                st.caption(f"⚠️ Source text not found: {label}")

            except Exception as e:
                st.error(f"Error generating response: {e}")
                clean_text = "I encountered an error."
                citations = []

        # Save state (we save the CLEAN text so we don't re-process unnecessarily)
        st.session_state.messages.append({
            "role": "assistant",
            "content": clean_text,
            "context": context_used,
            "citations": citations
        })
        st.rerun()


if __name__ == "__main__":
    main()
