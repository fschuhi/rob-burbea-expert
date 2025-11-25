# The LLM Layer: Choosing and Tuning the Synthesis Engine

## What the LLM Does in Your Pipeline

By the time the LLM sees anything, all the hard work is done:
- Chunks selected (embeddings)
- Best ones surfaced (reranking)
- Context formatted (context builder)
- Instructions set (system prompt)

The LLM's job: read the context, understand the question, synthesize a coherent answer, cite sources correctly.

This is *constrained generation* - much easier than open-ended chat. The model doesn't need world knowledge about jhānas; it needs to faithfully compress and rephrase the passages you've given it.

---

## Model Characteristics That Matter for RAG

**1. Instruction following**

Does the model do what the prompt says? Some models drift, ignore citation formats, or add unrequested material. For RAG, tight instruction following is essential.

**2. Faithfulness / Grounding**

Does it stick to the provided context, or hallucinate? A model that invents plausible-sounding dharma that Rob never said is worse than useless.

**3. Citation accuracy**

When it writes [1], does the claim actually come from Reference [1]? Some models cite randomly or cite the wrong source.

**4. Synthesis ability**

Can it weave multiple passages into a coherent answer? Or does it just quote chunks sequentially?

**5. Tone / Voice**

Does it sound like a helpful assistant? A Wikipedia article? A lecturer? For Rob's teachings, a warm, precise, practitioner-to-practitioner tone might be ideal.

**6. Speed**

You're running locally. 7B models are interactive; 13B models are slower; 30B+ models test your patience.

---

## Your Current Candidates

From your config and testing:

| Model | Size | Character |
|-------|------|-----------|
| dolphin-mistral:7b | 7B | Uncensored Mistral finetune, good instruction following |
| gemma3n-abliterated:e2b-fp16 | ~4B | Google's efficient model, alignment removed |
| qwen2.5 variants | 7B-14B | Strong reasoning, minimal filtering |

"Abliterated" / "uncensored" models have had their refusal training removed. For dharma content this mostly doesn't matter - Rob wasn't discussing anything models would refuse. But it does mean less hand-holding and hedging in responses.

---

## Speed vs. Quality Tradeoff

Rough expectations on your M4 24GB:

| Model Size | Speed (words/sec) | Quality |
|------------|-------------------|---------|
| 3B (phi, gemma-small) | ~25-30 | Basic, may miss nuance |
| 7B (mistral, qwen-7b) | ~15 | Good balance |
| 13B (qwen-14b) | ~8-10 | Better reasoning |
| 30B+ | ~3-5 | Best quality, but slow |

For interactive use, 7B is the sweet spot. For batch evaluation or when you can wait, larger models might be worth testing.

Your current ~15 words/sec with dolphin-mistral:7b is solidly usable - not instant, but comfortable for contemplative work.

---

## Quantization

Models come in different precisions:

- **fp16**: Full precision, largest, highest quality
- **q8**: 8-bit quantized, ~half the size, minimal quality loss
- **q4**: 4-bit quantized, quarter size, some quality loss

On your 24GB M4, you can run:
- 7B fp16 comfortably
- 13B q8 comfortably  
- 13B fp16 tight but possible
- 30B q4 possible but slow

For RAG specifically, q8 quantization is usually fine - the quality loss is small and the speed gain meaningful.

---

## What "Uncensored" Actually Changes

Standard models are trained to:
- Refuse harmful requests
- Add safety caveats
- Hedge on sensitive topics

"Abliterated" models have this training reversed. For your use case:

**Advantages:**
- No unexpected refusals on contemplative content
- Less hedging ("I should note that..." preambles)
- More direct responses

**Disadvantages:**
- Less guardrails if you ever expand scope
- Sometimes *too* direct (no "consult a teacher" caveats where appropriate)

For a personal dharma study tool, uncensored is probably fine. For something you'd share publicly, more nuanced.

---

## Inference Parameters

From `src/llm.py`, you can add:

```python
response = self.client.chat(
    model=self.model_name,
    messages=messages,
    options={
        "temperature": 0.2,       # Controls randomness
        "top_p": 0.9,             # Nucleus sampling
        "top_k": 40,              # Token selection limit
        "repeat_penalty": 1.1,    # Discourages loops
    },
    stream=True,
)
```

**temperature** (default often 0.7)

Controls randomness. Lower = more deterministic, higher = more creative.

For RAG grounded in source material: **low temperature (0.1-0.3)** is usually better. You want faithful synthesis, not creative riffing.

**top_p** (nucleus sampling)

Alternative to temperature. Limits token selection to the most probable tokens summing to p probability.

**top_k**

Only consider the k most likely next tokens.

**repeat_penalty**

Discourages repetition. Useful if the model gets stuck in loops.

---

## Model Selection Methodology

For your **Model Selection** goal in `Goals.md`, here's a concrete approach:

**Step 1: Define the probe set**

10-20 questions spanning:
- Factual: "What is the first jhāna?"
- Procedural: "How do I work with restlessness?"
- Comparative: "What's the difference between pīti and sukha?"
- Conceptual: "Why does Rob emphasize the energy body?"
- Edge cases: Questions where the answer isn't clearly in the corpus

Use `evaluation/probe_questions.xml` as your starting point.

**Step 2: Run each model**

Same retrieval, same prompt, different models. Save outputs.

```bash
# Pseudo-code
for model in dolphin-mistral:7b gemma3n-abliterated qwen2.5:7b; do
  # Switch model in config
  # Run probe set
  # Save to evaluation/results_${model}.md
done
```

**Step 3: Evaluate on criteria**

For each answer, score (Good/Okay/Bad) on:
- Correctness: Does it match what Rob taught?
- Faithfulness: Did it stick to the sources?
- Citations: Are [1], [2] used correctly?
- Completeness: Did it address the question fully?
- Tone: Does it sound right?

**Step 4: Aggregate and decide**

Which model wins most often? Are there patterns? ("Model X is better for procedural questions, Model Y for conceptual ones")

---

## Your M4 in Context

For local LLM inference: Upper tier of consumer hardware.

| Machine | dolphin-mistral:7b Speed |
|---------|-------------------------|
| Your M4 24GB | ~15 words/sec |
| M1 16GB | ~8-10 words/sec |
| M3 Max 64GB | ~18-22 words/sec |
| Gaming PC (RTX 4090) | ~60-80 words/sec |

Your hardware is not the limiting factor. You're in the sweet spot where local-first is viable but not instant.

---

## The Retrieval Bottleneck

Your 15-second query time breaks down roughly:

- Retrieval + rerank: ~0.6 seconds (4%)
- LLM processing prompt: ~7 seconds (47%)
- LLM generating answer: ~7 seconds (47%)

The LLM dominates. If you want faster responses, the levers are:

1. **Smaller model**: dolphin-phi (3B) would be 2× faster, but less capable
2. **Quantization**: 4-bit models run faster, slight quality loss
3. **Shorter context**: Fewer retrieved chunks = less for the LLM to read

But speed isn't your bottleneck - quality is. 15 seconds for a thoughtful answer is acceptable.

---

## Concrete Recommendation for Evaluation

For your Model Selection sprint:

1. **Start with dolphin-mistral:7b** - your current default, solid baseline
2. **Compare against gemma3n-abliterated** - different architecture, potentially different strengths
3. **Try qwen2.5:7b** - known for strong instruction following
4. **If time permits, try a 13B** - see if the quality jump is worth the speed cost

Document which wins on which criteria. The notes become the `notes` field in your Prompt Laboratory templates.

---

## Model-Prompt Interaction

Different models respond differently to the same prompt:

- **dolphin-mistral**: Good instruction following, might be overly direct
- **gemma3n**: Can be chatty, might need tighter constraints  
- **qwen**: Strong reasoning, handles complex prompts well

This is why Model Selection and Prompt Laboratory are adjacent goals in `Goals.md` - they interact. The best model-prompt pairing might not be your default model with your default prompt.

---

## Current Configuration

From `rb_expert.toml`:

```toml
[models]
default_llm_model = "dolphin-mistral:7b"

[ollama]
base_url = "http://localhost:11434"
timeout = 60
```

With Live Wire (dynamic model fetching) and The Menu (model selector UI), you'll be able to switch models without editing config files.

---

## Long-Context Models

Some newer models support 128K+ context:

- `qwen2.5-1m-abliterated:14b` - ~1M token context

Could you just dump all 32 talks into the prompt and skip the RAG pipeline?

**No, for several reasons:**

1. **Dilution**: The model's attention spreads thin across 1M tokens. Specific details get lost.
2. **Cost**: Processing 1M tokens on your M4 would take 5-10 minutes just to read the context.
3. **Quality**: RAG retrieval focuses the model on the 5 most relevant chunks. That focus is a feature, not a bug.

Long-context models are useful for single-document analysis (reading one full talk), not corpus search.

---

## Comparison to NotebookLM

NotebookLM has access to all 32 talks simultaneously in its large context. It represents the quality ceiling.

Your RAG system retrieves only top-K chunks. The tradeoff:
- NotebookLM: Complete context, slower, proprietary
- Your Expert: Focused context, faster, local, private

Sometimes your system will beat NotebookLM by surfacing a specific passage that the larger model glossed over. That's the power of retrieval.

---

## Next Steps

1. Review the Model Selection goal in `Goals.md`
2. Create your evaluation framework (probe set, scoring rubric)
3. Run dolphin-mistral as baseline
4. Test 2-3 alternatives
5. Document findings
6. Choose default based on your priorities (speed vs. quality vs. tone)

The model choice shapes everything downstream, including prompt design. Get this right and the rest follows.
