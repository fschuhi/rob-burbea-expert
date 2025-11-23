(This is a current collection of potential goals for the conversation with the co-developers or LLMs. Higher on the list means either more urgent or more important or both, i.e. the list shows what is currently  important to tackle next. The list is ordered by The overlap with TODO.md is intentional, the latter being more of a scratchpad.)

## Live Wire (Dynamic Ollama Models)
- **Intent**: Call `ollama list` at runtime to populate available LLMs, cache them, and expose to the UI.
- **Why it matters**: Keeps the “Menu” selector in sync with the actual local models without hand-editing the config.
- **Definition of done**: Backend utility that returns model metadata; error handling for offline/no-Ollama scenarios; unit coverage with mocks.

## The Menu (Model Selector UI)
- in conjunction with Live Wire
- **Intent**: Add a sidebar selector in both apps that lets the user switch the active LLM without restarting.
- **Why it matters**: Enables on-the-fly experimentation with different models (stateless per request).
- **Definition of done**: UI control wired into `OllamaClient` (or a lightweight wrapper) so each query uses the currently selected model.

## Cohesive Context
- **Intent**: Move paragraph reconstruction helpers (`get_paragraph_chunks`, `reconstruct_paragraph_with_hit`) out of `src/database.py` into `src/context.py`, so all context-assembly logic lives together.
- **Why it matters**: Streamlit apps, tests, and future backends can rely on a single module for context formatting, reducing cross-module dependencies.
- **Definition of done**: Database layer keeps only persistence helpers; context builder exposes reconstruction utilities with updated imports/tests.

## Telemetry Carve-Out
- **Intent**: Centralize telemetry measurement (retrieval time, TTFT, generation speed) in a reusable helper/service.
- **Why it matters**: Both apps duplicate timing logic and UI rendering; extracting it improves readability and enables unit tests.
- **Definition of done**: Helper exposes a clear API (e.g., start/stop phases, serialize to dict) and both apps consume it.

## Streaming Response Carve-Out
- **Intent**: Encapsulate streaming LLM responses (buffering, caret display, citation extraction) in a dedicated helper.
- **Why it matters**: Current in-line loops are long and hard to follow; a helper would make alternative UIs easier to implement.
- **Definition of done**: Helper yields UI-ready chunks, tracks final text, and reports used citations; apps delegate to it.

## Lazy Reranker Loader
- **Intent**: Delay loading the cross-encoder until reranking is actually requested.
- **Why it matters**: Avoids unnecessary startup downloads and makes “vector-only” workflows lighter.
- **Definition of done**: `RAGEngine` initializes instantly; reranker loads on first use and can be released/reloaded for alternative models.

## Future: Semantic Surgery
- **Intent**: Transition from paragraph/character chunking to sentence-based splits, likely leveraging LangChain splitters.
- **Why it matters**: Sentence-aware chunks should improve semantic cohesion and retrieval precision.
- **Definition of done**: Config option selects sentence splitter; tests cover both manual and new splitter; benchmarking notes captured.

## Engine Workflow Simplification
- **Intent**: Refactor `RAGEngine.retrieve_and_rerank` into smaller steps (retrieve, filter, score, build context) without changing behavior.
- **Why it matters**: Improves testability, prepares for backend reuse, and clarifies the boundary between core logic and presentation.
- **Definition of done**: Each stage has unit coverage; method reads as a high-level orchestration.

## Alternative UI Prototype (New)
- **Intent**: Experiment with a non-Streamlit front end (e.g., textual, CLI, or lightweight web framework) reusing the carved-out backend helpers.
- **Why it matters**: Validates that the backend boundaries are clean and offers options if Streamlit becomes limiting.
- **Definition of done**: Minimal prototype demonstrating query→answer flow via the shared backend.
