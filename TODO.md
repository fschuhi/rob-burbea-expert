# --- BACKEND & CORE LOGIC ---

- **Live Wire** - Implement a dynamic model fetcher in `src/llm.py` that calls `ollama list` to populate the model selector options in real-time.
- **Silencing the Linter** - Fix "Type of '__init__' is incompatible with 'EmbeddingFunction'" warnings in src/models.py (PyCharm strict type checking issue).
- **Data Liberation** - Add export functionality to save search results to markdown/JSON.
- **Etched in Stone** - Implement a "Save as Default" feature that writes current slider values back to `rb_expert.toml` (handling the file-watcher reload gracefully).
- **Future: Semantic Surgery** - Consider sentence-based chunking instead of paragraph-overlap approach for better semantic coherence.

# --- USER EXPERIENCE (UX) & FRONTEND ---

- [DONE] **Performance Telemetry** - Keep status indicators visible and show execution time (latency) for each step.
- [DONE] **Clutter Control** - Group the "Reference" expanders in the Answer Generator under a single parent expander or "Sources" section to keep the chat clean.
- [DONE] **Cinematic View** - Widen the main text area in the Answer Generator using custom CSS (`max-width`) so the conversation has more room to breathe.
- **The Menu** - Add a Model Selector dropdown in the Answer Generator sidebar (populated by `Live Wire` or config).
- **Branding Beacon** - Replace the text title "Rob Burbea Talks" in the sidebar with a configurable image (e.g., Rob's photo or a logo) in both apps.
- **Slice & Dice** - Implement retreat/date range filters in Search Explorer sidebar.
- **Parity Party** - Add the `Top K` slider to the Search Explorer sidebar so it matches the flexibility of the Answer Generator.
