# Refactoring Roadmap

This document explains the architectural decisions behind the ordering and implementation of strategic goals in [`Goals.md`](Goals.md).

---

## Design Principles

1. **Foundation First**: Refactor core abstractions before building on them
2. **Separation of Concerns**: Each module should have a single, clear responsibility
3. **Testability**: Break monoliths into testable stages
4. **Incremental Safety**: Each phase passes all tests before proceeding

---

## Two-Phase Refactoring Plan

### Phase A: Cohesive Context ✅ Priority 1

**Goal:** Consolidate all context-assembly logic in `src/context.py`

**Problem:** 
- `src/database.py` contains paragraph reconstruction functions
- These are **data retrieval** operations, not **persistence** operations
- `src/context.py` imports from `src/database.py` creating coupling

**Solution:** Move reconstruction helpers to where they belong

**Functions to Move:**
```python
# FROM: src/database.py
# TO:   src/context.py

def get_paragraph_chunks(collection, source, paragraph_index) -> list[dict]
def reconstruct_paragraph_with_hit(collection, source, paragraph_index, hit_chunk_position) -> dict
```

**Impact Analysis:**
- `src/context.py` already uses both functions
- `apps/answer_generator.py` uses `reconstruct_paragraph_with_hit`
- `apps/search_explorer.py` uses `reconstruct_paragraph_with_hit`
- Tests need import updates

**After Phase A:**
- `src/database.py` = Pure persistence (ChromaConnector only, ~53 lines)
- `src/context.py` = Context assembly + paragraph reconstruction (~200 lines)
- No circular dependencies

**Estimated Effort:** 1-2 hours

---

### Phase B: Engine Workflow Simplification ✅ Priority 2

**Goal:** Refactor `RAGEngine.retrieve_and_rerank()` into testable stages

**Problem:**
- Current method is 70+ lines of mixed concerns
- Hard to test individual stages (retrieval, filtering, scoring)
- Hard to reuse stages (e.g., batch evaluation needs scoring without UI)
- Difficult to understand control flow

**Solution:** Extract helper class with clear stages

**Proposed Structure:**
```python
class RetrievalPipeline:
    """Helper class for staged retrieval workflow."""
    
    def __init__(self, collection, cross_encoder):
        self.collection = collection
        self.cross_encoder = cross_encoder
    
    def retrieve(self, query: str, n_results: int) -> list[dict]:
        """Stage 1: Vector search."""
        results = self.collection.query(...)
        return self._unpack_chroma_results(results)
    
    def filter_by_distance(self, candidates: list, threshold: float) -> list[dict]:
        """Stage 2: Distance filtering."""
        return [c for c in candidates if c["initial_dist"] <= threshold]
    
    def score_candidates(self, query: str, candidates: list, use_reranker: bool) -> list[dict]:
        """Stage 3: Score and sort."""
        if use_reranker:
            pairs = [[query, c["text"]] for c in candidates]
            scores = self.cross_encoder.predict(pairs)
            for i, c in enumerate(candidates):
                c["score"] = scores[i]
            candidates.sort(key=lambda x: x["score"], reverse=True)
        else:
            candidates.sort(key=lambda x: x["initial_dist"])
        return candidates
    
    def _unpack_chroma_results(self, results: dict) -> list[dict]:
        """Convert Chroma format to candidate dicts."""
        ...

class RAGEngine:
    def __init__(self, env: Env):
        # ... existing init ...
        self.pipeline = RetrievalPipeline(self.collection, self.cross_encoder)
    
    def retrieve_and_rerank(...) -> Tuple[str, Dict]:
        """High-level orchestration."""
        candidates = self.pipeline.retrieve(query_text, initial_k)
        candidates = self.pipeline.filter_by_distance(candidates, final_threshold)
        candidates = self.pipeline.score_candidates(query_text, candidates, apply_reranker)
        top_hits = candidates[:final_top_k]
        return self.context_builder.build_from_hits(top_hits)
```

**Benefits:**
- Each stage is independently testable
- Clear data flow: candidates → filtered → scored → sliced
- Reusable for batch evaluation (call stages directly)
- Orchestration method is ~10 lines (readable at a glance)

**Testing Strategy:**
- Existing integration tests remain (verify end-to-end still works)
- Add focused unit tests for new helper class:
  - `test_retrieve()` - mock collection.query
  - `test_filter_by_distance()` - pass candidates, verify threshold
  - `test_score_candidates_with_reranker()` - mock cross_encoder.predict
  - `test_score_candidates_without_reranker()` - verify distance sort

**Estimated Effort:** 3-4 hours

---

## Why This Ordering?

### Cohesive Context Before Engine Simplification

**Reason 1: Complexity**
- Phase A is simpler (move functions, update imports)
- Phase B is more complex (extract helper class, orchestrate calls)
- Starting simple builds confidence

**Reason 2: Dependencies**
- Engine depends on context layer
- Clean context layer first → cleaner engine interface
- If context is messy, engine refactor is harder

**Reason 3: Risk Management**
- Phase A is low-risk (just moving functions)
- Phase B is higher-risk (changing control flow)
- If Phase B fails, Phase A is still valuable

**Reason 4: Psychological Win**
- Quick win with Phase A (1-2 hours) establishes momentum
- Harder Phase B feels more tractable after a success

---

## After Both Phases: What's Unblocked?

Once the foundation is clean:

1. **Model Selection** can use `RetrievalPipeline` stages for systematic evaluation
2. **Domain-Tuned Reranker** can call `score_candidates()` with alternative models
3. **Prompt Laboratory** can iterate on prompts knowing retrieval is consistent
4. **Batch Evaluation** can reuse pipeline stages without duplicating logic

---

## Implementation Guidelines

### For Cohesive Context (Phase A)

**Steps:**
1. Copy both functions to `src/context.py`
2. Update `src/context.py` imports (no need for `from src.database import ...`)
3. Update app imports:
   - `apps/answer_generator.py`: `from src.context import reconstruct_paragraph_with_hit`
   - `apps/search_explorer.py`: `from src.context import reconstruct_paragraph_with_hit`
4. Move tests from `tests/test_database.py` to `tests/test_context.py`
5. Run `make test` - all 71 should pass
6. Commit: "Refactor: Move paragraph reconstruction to context module"

**Success Criteria:**
- All tests pass
- `src/database.py` only contains `ChromaConnector` class
- `src/context.py` is the single source for context operations

---

### For Engine Workflow Simplification (Phase B)

**Steps:**
1. Create `RetrievalPipeline` helper class in `src/engine.py`
2. Extract `retrieve()`, `filter_by_distance()`, `score_candidates()` methods
3. Update `RAGEngine.retrieve_and_rerank()` to use pipeline
4. Write unit tests for each pipeline stage
5. Run `make test` - all tests (old + new) should pass
6. Commit: "Refactor: Extract retrieval pipeline stages for testability"

**Success Criteria:**
- All existing tests pass (no behavioral changes)
- New unit tests cover each stage
- `retrieve_and_rerank()` is ~10 lines of orchestration
- Batch evaluation scripts can import `RetrievalPipeline`

---

## Alternative Considered: Why Not Private Methods?

**Option 1: Private Methods in RAGEngine**
```python
class RAGEngine:
    def _retrieve(...)
    def _filter_by_distance(...)
    def _score_candidates(...)
```

**Why Separate Helper Class is Better:**
1. **Testability**: Helper class can be instantiated and tested independently
2. **Reusability**: Batch scripts can import `RetrievalPipeline` without full `RAGEngine`
3. **Clarity**: Separation makes it obvious these are a cohesive unit
4. **Growth**: Future alternative pipelines (e.g., `HybridRetrievalPipeline`) can inherit

**Decision:** Use separate helper class for cleaner architecture.

---

## Timeline Estimate

| Phase | Effort | Outcome |
|-------|--------|---------|
| Phase A: Cohesive Context | 1-2 hours | Clean separation: database = persistence, context = assembly |
| Phase B: Engine Simplification | 3-4 hours | Testable stages, reusable pipeline, readable orchestration |
| **Total** | **4-6 hours** | Foundation ready for advanced features |

---

## Next Conversation Handoff

When starting the next conversation:

1. Read this document to understand the refactoring plan
2. Check `Goals.md` for current priority (should be "Cohesive Context")
3. Propose implementation of Phase A
4. After Phase A is merged, propose Phase B
5. After both phases: ready for Model Selection, Domain-Tuned Reranker, etc.

**Key Files to Reference:**
- This document (`REFACTORING.md`) - implementation roadmap
- `Goals.md` - strategic intent and rationale
- `README.md` - current architecture
- `src/engine.py` - current `retrieve_and_rerank()` implementation
- `src/database.py` - functions to move
- `src/context.py` - destination for moved functions
