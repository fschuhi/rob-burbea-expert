# Chunking: Breaking Text into Semantic Units

## The Fundamental Problem

Embedding models have a context window - MiniLM handles about 256 tokens effectively (roughly 200-300 words). Feed it a 45-minute dharma talk transcript and it either truncates or produces a mushy "average" vector that captures nothing specific.

So we chunk: break the talk into pieces small enough to embed meaningfully, but large enough to preserve coherent thoughts.

**The tension:** Small chunks = precise retrieval but lost context. Large chunks = preserved context but diluted relevance signal.

---

## The Strategy in `data_prep.py`

This project uses a **semantic-first, two-phase approach** that respects Shannon's careful transcription work:

```
Phase 1: Split on paragraph boundaries (double newlines)
Phase 2: If a paragraph exceeds chunk_size, split it with overlap
Phase 3: Track which chunk came from which paragraph
```

This is smarter than naive character splitting because paragraphs tend to be semantic units - one thought, one instruction, one story.

---

## Phase 1: Respecting Paragraph Structure

```python
# Normalize line endings
clean_content = text_content.replace('\r\n', '\n').replace('\r', '\n')
clean_content = re.sub(r'\n{3,}', '\n\n', clean_content)

# Split on paragraphs
paragraphs = clean_content.split('\n\n')
```

Rob's transcripts use `\n\n` to separate paragraphs. This first pass respects that structure. A passage about pīti stays together; the transition to the next topic gets its own chunk.

Shannon and Rob spent time in quality assurance making sure paragraph breaks align with semantic shifts. The chunker honors that work.

---

## Phase 2: Handling Oversized Paragraphs

```python
for paragraph in paragraphs:
    para_stripped = paragraph.strip()
    
    if not para_stripped:
        continue
    
    # Skip the "## Transcription" header
    if para_stripped == "## Transcription":
        continue
    
    # Split if needed
    if len(para_stripped) <= rag_config.chunk_size:
        para_chunks = [para_stripped]
    else:
        para_chunks = _split_text_with_overlap(
            para_stripped,
            rag_config.chunk_size,
            rag_config.chunk_overlap,
            ["\n", " ", ""]  # Separator hierarchy
        )
```

Most paragraphs fit within 500 characters and stay intact. But when Rob goes on a longer riff, the splitter kicks in.

---

## The Overlap Splitter

The `separators` list `["\n", " ", ""]` is a **hierarchy of preferences**:

1. First, try to split on line breaks within the paragraph
2. If that doesn't work, split on word boundaries (spaces)
3. Last resort: split mid-word (character level)

This matters because "energy body" shouldn't become "energy bo" and "dy practice."

**The overlap** (default 50 characters) means consecutive chunks share some text:

```
Chunk 1: "...the texture of it, the vibration, the tone of it, the feel"
Chunk 2: "the tone of it, the feel, the energy of it. It shrinks; you..."
```

This overlap creates redundancy that helps retrieval. If someone searches for "tone and feel of the energy body," both chunks might score well, and reconstruction can show the full context.

---

## Phase 3: Metadata Tracking

```python
for chunk_position, chunk_text in enumerate(para_chunks):
    if chunk_text.strip():
        chunks.append(Document(
            page_content=chunk_text.strip(),
            metadata={
                "source": str(source_path),
                "paragraph_index": paragraph_index,
                "chunk_position": chunk_position,
                "total_chunks_in_para": total_chunks
            }
        ))

paragraph_index += 1
```

Each chunk knows:
- **source**: Which talk it came from
- **paragraph_index**: Which paragraph in that talk
- **chunk_position**: If the paragraph was split, which piece is this (0, 1, 2...)
- **total_chunks_in_para**: How many pieces the paragraph became

This metadata enables the paragraph reconstruction you see in the UI - when a search hit lands on chunk 2 of 4, the system can fetch chunks 0, 1, 2, 3 and show the full paragraph with the hit highlighted.

---

## Reconstruction in Action

In `database.py`, the `get_paragraph_chunks` function uses this metadata:

```python
def get_paragraph_chunks(collection, source, paragraph_index):
    results = collection.get(
        where={"$and": [
            {"source": source}, 
            {"paragraph_index": paragraph_index}
        ]}
    )
    # ... sort by chunk_position ...
    return chunks
```

And `reconstruct_paragraph_with_hit` reassembles them:

```python
full_text = " ".join(chunk["text"] for chunk in chunks)
# Plus: wrap the hit chunk in <hit>...</hit> tags for highlighting
```

This is why the Search Explorer can show full paragraphs with yellow highlighting on the matched portion.

---

## Current Configuration

From `rb_expert.toml`:

```toml
[rag]
chunk_size = 500
chunk_overlap = 50
```

With 5,008 total chunks across 32 talks, that's roughly 156 chunks per talk. Given that each talk is maybe 30-60 paragraphs, many paragraphs are staying intact while longer ones get split into 2-4 pieces.

---

## The Tradeoffs

**chunk_size = 500** (characters, roughly 80-100 words)

- Pro: Fine-grained retrieval - a specific instruction about breath can be found without pulling in unrelated material
- Con: Some of Rob's teachings unfold over multiple sentences; a 500-char chunk might catch the instruction but miss the "why"

**chunk_overlap = 50** (10% overlap)

- Pro: Some continuity at boundaries
- Con: Minimal - a search term spanning the boundary might still get split

**Paragraph-first splitting**

- Pro: Respects Rob's (and Shannon's) natural pacing and structure
- Con: Paragraphs in transcripts are somewhat arbitrary, though carefully considered

---

## What Could Go Wrong

**Scenario 1: The split thought**

Rob says: "So there's this movement, this kind of igniting of the energy body. And when I say 'igniting,' I mean..."

If the chunk boundary falls after "energy body," the explanation of "igniting" lands in a different chunk. Someone searching "what does igniting mean" might get the definition chunk but lose the context that it's about the energy body.

**Scenario 2: The Q&A asymmetry**

In Q&A talks, a questioner's brief question and Rob's long answer might be separate paragraphs. The answer chunk loses the question context.

---

## Experiments to Try

As the domain expert, you're positioned to evaluate these qualitatively:

1. **Spot-check split paragraphs**: Find a talk you know well, examine paragraphs that got split. Do the boundaries feel natural or jarring?

2. **Try different chunk_size values**: 300 (more granular) vs 800 (more context). Re-index with `make ingest-pilot`, run the same queries, compare results.

3. **Examine overlap behavior**: With overlap=50, search for a phrase you know spans a chunk boundary. Does it get found?

---

## The Chunking → Embedding Connection

**Embedding quality depends on chunk quality.**

If a chunk is a coherent thought, the embedding captures that thought. If a chunk is half of one thought and half of another, the embedding is a blurry average that matches neither query well.

The paragraph-first approach is already smarter than naive splitting. The metadata tracking enables reconstruction. The overlap provides some safety net.

The question is whether 500 characters is the right granularity for Rob's teaching style - and that's something only a domain expert can evaluate through systematic testing.

---

## Future: Sentence-Based Splitting

See `Goals.md` for "Semantic Surgery" - the idea of using sentence-aware splitters. This would ensure chunks never break mid-sentence, which is linguistically cleaner.

But it requires a sentence tokenizer that handles Rob's speaking style (contractions, em-dashes, Pāli terms). The current paragraph-first approach works well enough that sentence splitting is lower priority than tuning the existing parameters.
