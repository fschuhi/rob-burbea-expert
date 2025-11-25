# Embeddings: Turning Text into Searchable Vectors

## What Are Embeddings?

At the core of semantic search is a deceptively simple idea: **text can be represented as a list of numbers** (a "vector"), and similar texts produce similar numbers.

When you ask "How do I work with the energy body?", the system converts your question into something like:

```python
[0.042, -0.183, 0.091, 0.224, ..., -0.017]  # 384 numbers
```

Every chunk of Rob's talks has been pre-converted into the same format. Finding relevant passages becomes a matter of finding which vectors are "close" to your query vector in high-dimensional space.

---

## Where Embeddings Happen in This Project

The codebase has a clean separation of concerns:

```
models.py          →  Defines HOW to embed text
indexing.py        →  Uses embeddings to BUILD the database
engine.py          →  Uses embeddings to QUERY the database
```

---

## The Embedding Function (`src/models.py`)

```python
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    """
    A wrapper for sentence-transformers to be compatible with ChromaDB.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        return embeddings.tolist()
```

This wrapper adapts the `sentence-transformers` library to ChromaDB's interface. When called with a list of strings, it returns a list of vectors.

The factory function `get_embedding_function()` lets you swap in a `FakeEmbeddingFunction` for tests - every chunk gets the same `[0.1, 0.2, 0.3]` vector, making tests deterministic but obviously useless for real retrieval.

**The model choice matters:**
- `all-MiniLM-L6-v2` produces 384-dimensional vectors
- It's trained on semantic similarity tasks - "energy body" and "working with the whole body" end up nearby even though they share few words

---

## Building the Index (`src/indexing.py`)

At indexing time, the flow is:

```
Markdown talks → split into chunks → embed each chunk → store in ChromaDB
```

The key code:

```python
# Initialize embedding function
ef = get_embedding_function(env.models.embedding_model)

# Get/create collection WITH the embedding function attached
collection = connector.get_collection(
    name="rob_burbea_talks",
    embedding_function=ef
)

# Upsert documents - ChromaDB calls ef() internally
collection.upsert(
    ids=batch_ids,
    documents=batch_texts,      # ← These get embedded automatically
    metadatas=batch_metas
)
```

When you call `upsert()` with documents, ChromaDB internally calls your embedding function on each text. You never see the vectors directly - they're computed and stored transparently.

Your 5,008 chunks each become a 384-dimensional vector stored in ChromaDB's HNSW index (that's the `data_level0.bin` and friends in `chroma_db/`).

---

## Querying (`src/engine.py`)

At query time, the same embedding function is used to convert your question into a vector:

```python
# In RAGEngine.__init__()
self.ef = get_embedding_function(env.models.embedding_model)
self.collection = self.connector.get_collection(
    "rob_burbea_talks", 
    embedding_function=self.ef  # ← Same model as indexing!
)

# In retrieve_and_rerank()
results = self.collection.query(
    query_texts=[query_text],   # ← Your question gets embedded
    n_results=initial_k
)
```

ChromaDB embeds your query with the same function, then finds the `n_results` closest vectors using cosine distance.

**Critical:** The same embedding model must be used for indexing and querying. If you indexed with MiniLM but queried with MPNet, the vector spaces wouldn't align and retrieval would be garbage.

---

## What "Distance" Means

Your config sets `similarity_threshold = 0.75`. In the code:

```python
if dists[i] <= final_threshold:
    candidates.append(...)
```

ChromaDB uses **cosine distance** (set via `metadata={"hnsw:space": "cosine"}` in `database.py`):

- Distance 0.0 = identical vectors
- Distance 1.0 = orthogonal (no relationship)  
- Distance 2.0 = opposite

A threshold of 0.75 is fairly permissive - it admits chunks that are "somewhat related." Your reranker then sorts out which are truly relevant.

Looking at retrieval results, a top hit at distance 0.34 is quite close. A hit at 0.39 is still well under threshold. These are all semantically in the ballpark of the query concept.

---

## The Bi-Encoder Architecture

```
Query  → [Encoder] → Vector_Q ─┐
                               ├─→ Cosine Distance
Doc    → [Encoder] → Vector_D ─┘
```

- Each text encoded once, independently
- Vectors can be pre-computed and stored (your 5,008 chunks)
- Query embedding compared against all stored vectors
- Fast: O(1) per document (just vector math)

This is called a "bi-encoder" because query and document are encoded separately in parallel, then compared afterward.

---

## What MiniLM Actually Learned

The model was trained on millions of sentence pairs labeled "similar" or "not similar." Through this, it learned to:

- Place synonyms nearby: "pīti" and "rapture" cluster together
- Capture semantic roles: "working with X" and "how do I use X" are close
- Ignore surface differences: "the energy body" vs "energy-body practice" → similar vectors

It doesn't understand jhānas or Rob's teaching style - it just knows general English semantic similarity. The magic of RAG is that this general-purpose similarity is often enough to surface the right passages.

---

## Dimensions and Model Size

**Why 384 dimensions?**

MiniLM is compressed for speed. Larger models (768+ dims) might capture more nuance but run slower. The dimensionality represents the model's capacity to encode different aspects of meaning.

**Common embedding models:**

| Model | Dimensions | Speed | Quality |
|-------|-----------|-------|---------|
| all-MiniLM-L6-v2 | 384 | Fast | Good |
| all-mpnet-base-v2 | 768 | Medium | Better |
| instructor-large | 768 | Slow | Best |

For local-first RAG on a 5,000 chunk corpus, MiniLM's balance of speed and quality is ideal.

---

## Embeddings and Chunking Interact

**Embedding quality depends on chunk quality.**

If a chunk is a coherent thought, the embedding captures that thought. If a chunk is half of one thought and half of another, the embedding is a blurry average that matches neither query well.

This is why the paragraph-first chunking strategy matters - it tries to keep semantic units together, giving the embedding model coherent text to encode.

---

## Questions for Exploration

1. **What happens at chunk boundaries?** If Rob says something crucial that spans two chunks, can embeddings "see" both? (Hint: this is why overlap exists)

2. **Could a dharma-specific embedding model do better?** Theoretically yes - a model fine-tuned on contemplative texts might place "energy body" and "vedanā" closer than MiniLM does.

3. **How do you know if 0.75 is the right threshold?** You don't, without evaluation. That's one of the tuning knobs you can experiment with as a domain expert.

4. **Why not just use keyword search?** Embeddings understand "breath practice" and "working with the breath" are related. Keywords would miss the match.

---

## Configuration

From `rb_expert.toml`:

```toml
[models]
embedding_model = "all-MiniLM-L6-v2"

[rag]
similarity_threshold = 0.75
```

To experiment with different embedding models, change the `embedding_model` value and re-run `make ingest-pilot` to rebuild the index.
