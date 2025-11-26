# Reranking: The Precision Pass

## Why Reranking Exists

The bi-encoder (MiniLM) is fast but shallow. It embeds the query and each document *independently*, then compares their vectors. It never directly "reads" the query and document together.

This causes ranking errors. Consider:

- Query: "What is the third jhāna?"
- Chunk A: "The third jhāna is characterized by equanimity..."
- Chunk B: "In the third talk, I mentioned jhāna practice..."

Both chunks contain "third" and "jhāna" and will have similar embeddings. The bi-encoder might rank them equally. But a human (or a smarter model) immediately sees that A answers the question and B doesn't.

The **cross-encoder** fixes this. It takes each (query, document) pair and scores them *together*, attending to how the words interact across both texts.

---

## Two Architectures, Two Purposes

**Bi-Encoder (MiniLM)**

'''
Query  → [Encoder] → Vector_Q ─┐
                               ├─→ Cosine Distance
Doc    → [Encoder] → Vector_D ─┘
'''

- Each text encoded once, independently
- Vectors can be pre-computed and stored (your 5,008 chunks)
- Query embedding compared against all stored vectors
- Fast: O(1) per document (just vector math)

**Cross-Encoder (MS MARCO)**

'''
[Query + Doc] → [Full Transformer] → Relevance Score
'''

- Query and document fed together as one input
- Full attention across both texts
- Must be computed fresh for each (query, doc) pair
- Slow: O(n) transformer forward passes

You can't pre-compute cross-encoder scores because they depend on the query. That's why it's used for reranking a small candidate set, not for initial retrieval.

---

## Implementation in `engine.py`

The retrieval logic is organized into two classes:

**`RetrievalPipeline`** - The staged workflow:

'''python
class RetrievalPipeline:
    """
    Staged retrieval workflow with dependency injection.
    """
    def __init__(self, collection: Collection, cross_encoder: CrossEncoder):
        self.collection = collection
        self.cross_encoder = cross_encoder

    def retrieve(self, query: str, n_results: int) -> List[Dict[str, Any]]:
        """Stage 1: Vector search."""
        ...

    def filter_by_distance(self, candidates: List[Dict], threshold: float) -> List[Dict]:
        """Stage 2: Distance filtering."""
        ...

    def score_candidates(self, query: str, candidates: List[Dict], use_reranker: bool) -> List[Dict]:
        """Stage 3: Cross-encoder scoring or distance sort."""
        ...
'''

**`RAGEngine`** - The orchestrator that owns the resources:

'''python
class RAGEngine:
    def __init__(self, env: Env):
        # Stage 1: Fast retrieval
        self.ef = get_embedding_function(env.models.embedding_model)
        self.collection = self.connector.get_collection(
            "rob_burbea_talks", 
            embedding_function=self.ef
        )
        
        # Stage 2: Precision reranking
        print(f"Loading Reranker: {env.models.reranker_model}...")
        self.cross_encoder = CrossEncoder(env.models.reranker_model)
        
        # Pipeline (uses injected dependencies)
        self.pipeline = RetrievalPipeline(self.collection, self.cross_encoder)
'''

The cross-encoder loads once at startup (~80MB model). The pipeline receives both dependencies via injection, making each stage independently testable.

---

## The Retrieval Flow

From `retrieve_and_rerank()` with defaults: `top_k=5`, `retrieval_pool_size=25`, `distance_threshold=0.75`

**Step 1: Broad vector search**

'''python
initial_k = pool_size if apply_reranker else final_top_k  # 25 if reranking

candidates = self.pipeline.retrieve(query_text, initial_k)
'''

Inside `retrieve()`, ChromaDB embeds the query and finds the 25 nearest vectors. Fast (~50ms). Results are unpacked into candidate dicts with `id`, `text`, `metadata`, and `initial_dist`.

**Step 2: Distance filtering**

'''python
candidates = self.pipeline.filter_by_distance(candidates, final_threshold)
'''

Some of the 25 might be too distant. This prunes obviously irrelevant results before the expensive reranking step.

**Step 3: Cross-encoder scoring**

'''python
candidates = self.pipeline.score_candidates(query_text, candidates, apply_reranker)
'''

Inside `score_candidates()`, when `use_reranker=True`:

'''python
pairs = [[query, c["text"]] for c in candidates]
scores = self.cross_encoder.predict(pairs)

for i, candidate in enumerate(candidates):
    candidate["score"] = scores[i]

candidates.sort(key=lambda x: x["score"], reverse=True)
'''

The cross-encoder sees each (query, chunk) pair and outputs relevance scores. Higher = more relevant. This is where the magic happens—the model attends across both texts simultaneously.

**Step 4: Take the top K**

'''python
top_hits = candidates[:final_top_k]  # Best 5
return self.context_builder.build_from_hits(top_hits)
'''

From 25 candidates, you get the 5 that the cross-encoder judged most relevant.

---

## What MS MARCO Learned

The reranker is `cross-encoder/ms-marco-MiniLM-L-6-v2`. MS MARCO is a dataset of real Bing search queries paired with relevant/irrelevant passages.

The model learned patterns like:
- "What is X?" should match passages that define X, not just mention it
- Question words matter: "how" wants process, "why" wants explanation
- Specificity: "third jhāna" should match "third jhāna", not "jhāna in general"

It doesn't know dharma, but it knows what "answering a question" looks like.

---

## The Toggle in the UI

The Search Explorer has: "Apply Reranker ☑️"

**With reranker OFF:** Results sorted by vector distance (0.34, 0.36, 0.38...)

**With reranker ON:** Results sorted by cross-encoder score. The order might change significantly - a chunk at distance 0.38 might jump to position 1 if it better answers the query.

The telemetry shows "Retrieval & Rerank: 0.56s" - that includes both vector search (~50ms) and cross-encoder scoring (~500ms for 25 candidates).

---

## Tunable Parameters

**retrieval_pool_size** (default 25)

How many candidates to fetch for reranking. Larger = more chances to find the best answer, but slower.

Trade-off: If the true best answer is ranked #30 by the bi-encoder, a pool of 25 will miss it. But pool of 50 doubles your reranking time.

**top_k** (default 5)

How many final results to return. This is what the LLM sees.

**apply_reranker** (the checkbox)

Turn it off for speed, or to debug whether reranking is helping or hurting for specific queries.

---

## When Reranking Helps Most

1. **Ambiguous queries**: "energy" could mean energy body, energy of the hindrances, or energetic effort. The bi-encoder retrieves all. The cross-encoder picks the one matching your intent.

2. **Specific jhāna questions**: "What's different about the fourth jhāna?" - the bi-encoder might rank any jhāna chunk similarly. The cross-encoder knows you want *specifically* the fourth.

3. **Conceptual vs. keyword overlap**: "letting go" vs. a passage about "release" - the bi-encoder might miss the synonym. But if it's in the top 25, the cross-encoder can rescue it.

---

## When Reranking Might Hurt

1. **Already-precise queries**: If your query is so specific that the top 5 bi-encoder results are all perfect, reranking just wastes 500ms.

2. **Domain mismatch**: MS MARCO was trained on web search. Contemplative vocabulary ("vedanā", "soulmaking", "ways of looking") wasn't in the training data. The cross-encoder might misjudge relevance for dharma-specific terms.

3. **Short chunks**: Cross-encoders work best with substantial passages. If chunks are very small, there's less signal for the model to score.

---

## Performance

On your M4 MacBook:

| Operation | Time |
|-----------|------|
| Bi-encoder query embedding | ~20ms |
| Vector similarity (25 candidates) | <1ms |
| Cross-encoder reranking (25) | ~500ms |
| **Total retrieval + rerank** | ~0.6s |
| LLM generation | ~7-15s |

The 500ms reranking is about 3-5% of your total query time. The LLM generation dominates. So the reranker is cheap relative to its benefit.

---

## A Concrete Example

Query: "How do I work with the energy body?"

**Bi-encoder top 5** (hypothetical distances):
1. "Insight Ways of Looking..." (Dist: 0.34)
2. "The Energy Body and the Whole Body Breath..." (Dist: 0.36)
3. "An Introduction to the Jhānas" (Dist: 0.38)
4. "Focusing on One Point..." (Dist: 0.40)
5. "Breathing with the Energy Body..." (Dist: 0.42)

**After cross-encoder reranking** (hypothetical scores):
1. "The Energy Body and the Whole Body Breath..." (Score: 0.89)
2. "Breathing with the Energy Body..." (Score: 0.84)
3. "Insight Ways of Looking..." (Score: 0.71)
4. "An Introduction to the Jhānas" (Score: 0.52)
5. "Focusing on One Point..." (Score: 0.48)

The chunk that was #2 by distance becomes #1 by relevance - it's from a talk specifically about energy body instructions. The cross-encoder recognizes it as a better answer even though another chunk was slightly "closer" in vector space.

---

## Future: Domain-Tuned Reranker

See `Goals.md` for the Domain-Tuned Reranker goal. The idea: collect 1,000 pairwise judgments (you comparing chunks for relevance), then fine-tune the cross-encoder on your preferences.

A reranker trained on "what Rob meant" would better distinguish nuanced contemplative concepts (pīti vs. sukha, first jhāna vs. third jhāna) than the generic MS MARCO model.

---

## Configuration

From `rb_expert.toml`:

'''toml
[models]
reranker_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

[rag]
retrieval_pool_size = 25
top_k_results = 5
similarity_threshold = 0.75
'''

To experiment: try `retrieval_pool_size = 15` (faster) or `40` (more thorough) and observe if answer quality changes.
