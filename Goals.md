# Goals

Strategic work that shapes capabilities and architecture. Each goal has intent, rationale, and definition of done. See also [`TODO.md`](TODO.md) for tactical quick-wins.

> **Implementation guidance:** See [`REFACTORING.md`](REFACTORING.md) for the architectural roadmap behind the top two priorities (Cohesive Context and Engine Workflow Simplification).

---

## Engine Workflow Simplification
- **Intent**: Refactor `RAGEngine.retrieve_and_rerank` into smaller steps (retrieve, filter, score, build context) without changing behavior.
- **Why it matters**: Improves testability, prepares for backend reuse, and clarifies the boundary between core logic and presentation.
- **Definition of done**: Each stage has unit coverage; method reads as a high-level orchestration.

## Model Selection
- **Intent**: Systematically evaluate candidate base models (e.g., dolphin-mistral, gemma3n-abliterated, qwen variants) to identify the best default for the Expert.
- **Why it matters**: The LLM determines answer quality, tone, instruction-following, and citation accuracy. This choice shapes everything downstream, including prompt design.
- **Evaluation criteria**:
  1. **Faithfulness**: Does the model stick to the provided context without hallucinating?
  2. **Citation accuracy**: When it writes [1], does the claim actually come from Reference [1]?
  3. **Instruction following**: Does it respect the system prompt format and constraints?
  4. **Synthesis ability**: Can it weave multiple passages into coherent answers (vs. sequential quoting)?
  5. **Completeness**: Does it address the full question?
  6. **Tone**: Warm, precise, practitioner-to-practitioner (matching Rob's style)?
  7. **Speed**: Interactive responsiveness on M4 hardware (~15 words/sec minimum)
- **Approach**:
  1. Create probe set: 10-20 questions spanning factual, procedural, comparative, conceptual
  2. Run each candidate model with identical retrieval and prompt
  3. Score each answer on the criteria above (Good/Okay/Bad)
  4. Pairwise comparison where quantitative metrics are insufficient
  5. Document strengths/weaknesses of each model
- **Definition of done**: Recommended default model with documented rationale; runner-up alternatives noted for specific use cases.

## Domain-Tuned Reranker
- **Intent**: Fine-tune the cross-encoder on dharma-specific relevance judgments collected via pairwise comparison.
- **Why it matters**: MS MARCO was trained on web search; a reranker tuned on "what Rob meant" would better distinguish nuanced contemplative concepts (e.g., pīti vs. sukha, first jhāna vs. third jhāna).
- **Components**:
  1. Question generation: Use NotebookLM (or similar) to generate hundreds of questions from the 32 talks
  2. Comparison UI: App presenting query + two chunks; keyboard-driven A/B/Equal judgments
  3. Training pipeline: Export preferences as triplets, fine-tune `ms-marco-MiniLM-L-6-v2`
- **Definition of done**: ~1,000 pairwise judgments collected; fine-tuned model outperforms baseline on held-out probe set; model integrated as optional reranker in config.

## Streaming Response Carve-Out
- **Intent**: Encapsulate streaming LLM responses (buffering, caret display, citation extraction) in a dedicated helper.
- **Why it matters**: Current in-line loops are long and hard to follow; a helper would make alternative UIs easier to implement.
- **Definition of done**: Helper yields UI-ready chunks, tracks final text, and reports used citations; both apps delegate to it.

## Lazy Reranker Loader
- **Intent**: Delay loading the cross-encoder until reranking is actually requested.
- **Why it matters**: Avoids unnecessary startup downloads and makes "vector-only" workflows lighter.
- **Definition of done**: `RAGEngine` initializes instantly; reranker loads on first use and can be released/reloaded for alternative models.

## Prompt Laboratory
- **Intent**: Make system prompts editable, saveable, and switchable from the Answer Generator UI.
- **Why it matters**: The system prompt is high-leverage and zero-cost to change; rapid iteration requires persistence and comparison. Different prompts suit different query types (practical instruction vs. conceptual exploration).
- **Components**:
  1. Storage: `prompts/` directory with TOML templates including freeform `notes` field for documenting model/prompt combination use cases
  2. Selector: Dropdown in sidebar to switch active prompt
  3. Editor: In-app text area for live editing with token count
  4. Management: Duplicate, rename, delete prompts
- **Template structure**:
  ```toml
  name = "Strict Grounding"
  description = "No outside knowledge, dense citations"
  notes = """
  Works well with dolphin-mistral for factual queries.
  Tends to over-cite with gemma3n. Consider 'Conversational' prompt for that model.
  """
  
  [template]
  system = "..."
  ```
- **Definition of done**: Default prompt ships with install; user can create/edit/switch prompts without touching code; prompt choice persists across sessions.

## Future: Semantic Surgery
- **Intent**: Transition from paragraph/character chunking to sentence-based splits, likely leveraging LangChain splitters.
- **Why it matters**: Sentence-aware chunks should improve semantic cohesion and retrieval precision. Lower priority than tuning chunk_size/overlap via config, which require no code changes.
- **Definition of done**: Config option selects sentence splitter; tests cover both manual and new splitter; benchmarking notes captured.

## Future: Alternative UI Prototype
- **Intent**: Experiment with a non-Streamlit front end (e.g., textual, CLI, or lightweight web framework) reusing the carved-out backend helpers.
- **Status**: Deferred - Streamlit is working well for current use case. Revisit if UI framework becomes limiting.
- **Definition of done**: Minimal prototype demonstrating query→answer flow via the shared backend.
