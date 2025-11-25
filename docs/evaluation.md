# Evaluation: Knowing Whether It's Working

## The Fundamental Problem

You've built a pipeline with many knobs:
- Chunk size, overlap
- Embedding model
- Distance threshold
- Retrieval pool size
- Reranker on/off
- Top K
- LLM model
- System prompt

Turning any knob might make things better or worse. Without measurement, you're tuning by vibes.

But measurement requires knowing what "good" looks like. And that's hard.

---

## Three Levels of Evaluation

**Level 1: Retrieval** — Did we find the right chunks?

**Level 2: Ranking** — Are the best chunks at the top?

**Level 3: Answer** — Is the final response correct and faithful?

Each level has different methods and different difficulties.

---

## Level 1: Retrieval Evaluation

The question: Given a query, did the retrieved chunks contain the information needed to answer?

**If you had labeled data:**

A test set of (question, relevant_chunk_ids) pairs would let you compute:

| Metric | Meaning |
|--------|---------|
| Recall@K | Of all relevant chunks, what % did we retrieve in top K? |
| Precision@K | Of the K chunks we retrieved, what % were relevant? |
| MRR (Mean Reciprocal Rank) | How high is the first relevant chunk? (1/rank, averaged) |

Example: Query has 3 truly relevant chunks in the corpus. You retrieve 5 chunks, 2 of which are relevant.
- Recall@5 = 2/3 = 0.67
- Precision@5 = 2/5 = 0.40

**You don't have labeled data yet.**

Creating it is work: you'd need to write 50-100 questions and manually identify which chunks answer each one. Time-consuming, but doable for a 32-talk corpus you know well.

See `evaluation/probe_questions.xml` for a starting point - 10 questions from NotebookLM.

---

## Level 2: Ranking Evaluation

The question: Does the reranker improve ordering compared to vector distance alone?

**Comparative approach:**

For the same query, capture:
- Ranking by bi-encoder distance
- Ranking by cross-encoder score

Then judge: which order is better?

You can do this qualitatively in your Search Explorer. Toggle "Apply Reranker" and compare. For queries where you know the answer, does reranking surface the right chunk?

**If you had labeled data:**

You could compute the improvement in MRR or NDCG (Normalized Discounted Cumulative Gain) between the two rankings.

---

## Level 3: Answer Evaluation

The question: Is the LLM's response actually good?

This is the hardest level because "good" is multidimensional:

| Criterion | Question |
|-----------|----------|
| **Correctness** | Is the information accurate? |
| **Faithfulness** | Does it reflect what Rob actually said? |
| **Completeness** | Did it address the full question? |
| **Citation accuracy** | Do the [1], [2] references point to supporting passages? |
| **Hallucination** | Did it invent things not in the sources? |

**Automated metrics exist but are weak:**

- BLEU, ROUGE: Compare to reference answer. You don't have reference answers yet.
- BERTScore: Semantic similarity to reference. Same problem.
- LLM-as-judge: Ask GPT-4 "Is this answer good?" Works, but expensive and circular.

**Your advantage: domain expertise.**

You've studied Rob's teachings deeply. You can read an answer and immediately sense whether it's faithful or slightly off. This qualitative judgment is more valuable than any metric.

---

## Practical Evaluation Without Labeled Data

### 1. The Probe Set

Create 10-20 questions where you know what a good answer looks like:

```
- "What is the first jhāna?"
- "How does Rob describe pīti?"
- "What's the relationship between mettā and the energy body?"
- "What are the hindrances?"
- "How do I work with restlessness?"
```

Run these periodically. When you change a parameter, re-run and compare. You become the evaluation function.

Start with `evaluation/probe_questions.xml` - it has 10 questions with reference answers from NotebookLM.

### 2. Failure Collection

When you use the Expert and get a bad answer, save it:

```markdown
## Failure Log

### 2025-01-15
Query: "What does Rob say about the third jhāna and sukha?"
Problem: Retrieved chunks about first jhāna instead
Hypothesis: "sukha" appears more in first jhāna discussions
```

Over time, patterns emerge. Maybe certain query types consistently fail. That tells you where to focus.

### 3. A/B Snapshots

Before changing a parameter:
1. Run your probe set, save results
2. Change the parameter
3. Re-run, compare

```bash
# Pseudo-workflow
make probe-set > results_baseline.md
# Edit rb_expert.toml: chunk_size = 400
make ingest-pilot
make probe-set > results_chunk400.md
diff results_baseline.md results_chunk400.md
```

### 4. Citation Audit

For a sample of answers, verify the citations:
- Click the [1], [2] references
- Does the source text actually support the claim?
- Is the LLM hallucinating connections?

This catches a specific failure mode: the retrieval was fine, but the LLM misrepresented the sources.

---

## Building Toward Systematic Evaluation

If you wanted to invest in evaluation infrastructure:

**Phase 1: Probe Set (manual, quick)**

A markdown file with 20 questions. Run manually, judge by eye.

**Phase 2: Golden Answers (more work)**

For each probe question, write what a good answer should include:

```yaml
- question: "What is the first jhāna?"
  must_mention:
    - pīti (rapture)
    - sukha (happiness/pleasure)
    - one-pointedness or unification
  must_not_mention:
    - equanimity (that's third/fourth jhāna)
  relevant_talks:
    - "2019-12-22-the-first-jhana-and-playing..."
```

Now you can semi-automate: check if the answer contains the must_mention terms.

**Phase 3: Retrieval Ground Truth (significant work)**

For each probe question, identify the 3-5 chunks that should be retrieved. Now you can compute recall/precision and track it over time.

---

## The Domain Expert Advantage

Most RAG developers are building systems for domains they don't deeply understand. They *need* metrics because they can't judge quality directly.

You're building an expert system for teachings you've studied for years under a teacher you knew personally. You can read an answer about the energy body and immediately feel whether it captures Rob's meaning or flattens it.

This is more powerful than any metric. The metrics are proxies for "does a human expert think this is good?" You *are* the human expert.

The discipline is just to do it systematically: keep a probe set, log failures, compare before/after when you change things.

---

## A Minimal Starting Point

If you wanted to start evaluation today with 30 minutes of work:

1. Use `evaluation/probe_questions.xml` (10 questions already generated)
2. Run each through the Answer Generator
3. For each, note: Good / Okay / Bad, and why
4. Save as `evaluation/baseline_results.md`

Next time you change something significant (model, chunk size, prompt), repeat and compare.

---

## Using NotebookLM as Oracle

Upload the 32 talks to NotebookLM. It has access to everything simultaneously - no chunking, no retrieval errors, no top-K cutoff.

Ask it your probe questions. Its answers become reference answers - the ceiling your RAG system aspires to.

The comparison isn't exact-match - it's semantic. Does your system's answer cover the same ground? Miss key points? Add hallucinations?

One subtlety: NotebookLM has its own biases and might occasionally get things wrong. Your domain expertise is still the final arbiter. But as a scalable way to generate plausible reference answers, it's clever.

---

## Pairwise Comparison

See `Goals.md` for the Domain-Tuned Reranker goal, which uses pairwise comparison to collect training data.

The same approach works for evaluation:

```
┌─────────────────────────────────────────────────┐
│  Query: "How do I work with the energy body?"  │
├────────────────────┬────────────────────────────┤
│      Answer A      │         Answer B           │
│  (Reranker ON)     │  (Reranker OFF)            │
│  [text...]         │  [text...]                 │
├────────────────────┴────────────────────────────┤
│   [ A is better ]  [ Tie ]  [ B is better ]     │
└─────────────────────────────────────────────────┘
```

Why pairwise is easier than absolute scoring:

| Absolute | Pairwise |
|----------|----------|
| "Rate this answer 1-10" | "Which is better, A or B?" |
| Requires calibrated scale | Just comparison |
| "Is 7 good or bad?" | Binary choice |
| Cognitive load: high | Cognitive load: low |

After 50-100 judgments, you have statistically meaningful preferences. You can compute win rates, even Elo ratings.

---

## What You'd Learn

After systematic evaluation:

- "Reranker ON beats Reranker OFF 73% of the time"
- "Chunk 500 vs Chunk 400 is basically a tie (52%)"
- "dolphin-mistral beats gemma3n 64% of the time for factual queries"
- "My system beats NotebookLM reference 41% of the time"

That last one is interesting - if your RAG system sometimes beats the oracle, it might be surfacing specific passages that NotebookLM's synthesis glossed over.

---

## Configuration

No configuration needed - evaluation is process, not code.

But you might add to `Makefile`:

```bash
probe-set: ## Run probe questions through the system
	# Script to batch-process probe_questions.xml
```

---

## Next Steps

1. Run the 10 questions from `evaluation/probe_questions.xml`
2. Score each answer qualitatively
3. Document as baseline
4. Start collecting failures as you use the system
5. When you change a parameter, re-run the probe set

Evaluation is the feedback loop that turns random changes into informed improvements.
