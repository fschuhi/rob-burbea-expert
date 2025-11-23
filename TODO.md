(The TODOs in this list are loosely ordered by priority, not necessarily in order of completion. See also Goals.md).

### --- BACKEND & CORE LOGIC ---

- **Live Wire** - Implement a dynamic model fetcher in `src/llm.py` that calls `ollama list` to populate the model selector options in real-time.
- **Cohesive Context** - Relocate paragraph reconstruction helpers from `src/database.py` to `src/context.py`, update imports, and keep database utilities focused on persistence concerns.
- **Telemetry Carve-Out** - Extract telemetry capture/formatting logic into a shared helper so both apps (and future clients) can reuse consistent metrics.
- **Streaming Response Carve-Out** - Move streaming-response buffering, caret display handling, and citation parsing into a helper/service for cleaner UIs.
- **Data Liberation** - Add export functionality to save search results to markdown/JSON.
- **Etched in Stone** - Implement a "Save as Default" feature that writes current slider values back to `rb_expert.toml` (handling the file-watcher reload gracefully).
- **Future: Semantic Surgery** - Introduce sentence-based chunking (likely via LangChain) as a configurable option, replacing the manual splitter when ready.
- **Lazy Reranker Loader** - Delay loading the cross-encoder until reranking is actually needed to avoid startup delays/offline surprises.
- **Engine Workflow Simplification** - Break `RAGEngine.retrieve_and_rerank` into smaller, testable stages (retrieve, filter, rerank, context build) to improve maintainability.

### --- USER EXPERIENCE (UX) & FRONTEND ---

- **The Menu** - Add a Model Selector dropdown in the Answer Generator sidebar (populated by `Live Wire` or config).
- **AG Whitespace** - Remove whitespace from left pane (Answer Generator)
- **Branding Beacon** - Replace the text title "Rob Burbea Talks" in the sidebar with a configurable image (e.g., Rob's photo or a logo) in both apps.
- **Slice & Dice** - Implement retreat/date range filters in Search Explorer sidebar.
- **Telemetry Carve-Out (UI wiring)** - Update both Streamlit apps to consume the new telemetry helper once it exists.
- **Streaming Response Carve-Out (UI wiring)** - Update both apps to use the shared streaming helper for consistent UX across future frontends.
- **Alternative UI Prototype** - Build a minimal non-Streamlit client (CLI/textual/lightweight web) to validate backend boundaries.

### --- Devops ----

- **Silencing the Linter** - Fix "Type of '__init__' is incompatible with 'EmbeddingFunction'" warnings in src/models.py (PyCharm strict type checking issue).
- **Zero Warnings** - Go through all code and fix any PyCharm warnings.
- **Complete Rob** - Continue to have the Practicing the Jhanas production, but also the full data set.
- 