### --- BACKEND & CORE LOGIC ---

- **Live Wire** - Implement a dynamic model fetcher in `src/llm.py` that calls `ollama list` to populate the model selector options in real-time.
- **Data Liberation** - Add export functionality to save search results to markdown/JSON.
- **Etched in Stone** - Implement a "Save as Default" feature that writes current slider values back to `rb_expert.toml` (handling the file-watcher reload gracefully).
- **Future: Semantic Surgery** - Consider sentence-based chunking instead of paragraph-overlap approach for better semantic coherence.

### --- USER EXPERIENCE (UX) & FRONTEND ---

- **The Menu** - Add a Model Selector dropdown in the Answer Generator sidebar (populated by `Live Wire` or config).
- **Branding Beacon** - Replace the text title "Rob Burbea Talks" in the sidebar with a configurable image (e.g., Rob's photo or a logo) in both apps.
- **Slice & Dice** - Implement retreat/date range filters in Search Explorer sidebar.

### --- Devops ----

- **Silencing the Linter** - Fix "Type of '__init__' is incompatible with 'EmbeddingFunction'" warnings in src/models.py (PyCharm strict type checking issue).
- **Zero Warnings** - Go through all code and fix any PyCharm warnings.
- **Complete Rob** - Continue to have the Practicing the Jhanas production, but also the full data set.  