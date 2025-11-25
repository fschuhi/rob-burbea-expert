# Prompt Engineering: Shaping the Synthesis

## The Last Mile

We've covered how chunks get retrieved (embeddings), how they're prepared (chunking), and how they're sorted (reranking). Now: what happens when the LLM sees them?

The system prompt is the last mile. Everything upstream can be perfect, but a bad prompt produces bad answers.

---

## Your Current Prompt

From `src/llm.py`:

```python
system_prompt = (
    "You are a helpful assistant specializing in the teachings of Rob Burbea. "
    "Answer the user's question based *only* on the provided reference material. "
    "Cite your sources using the reference IDs in square brackets, e.g., [1], [2]. "
    "If the answer is not in the material, say so.\n\n"
    "### EXAMPLE ###\n"
    "Context: \n"
    "### Reference [1]: talk.md\n"
    "Breathing is good.\n"
    "\n"
    "Assistant: You should practice breathing **[1]**."
)
```

And the user message:

```python
{
    'role': 'user',
    'content': f"Reference Material:\n{context}\n\nQuestion:\n{query}"
}
```

---

## What This Prompt Does Well

1. **Role anchoring**: "specializing in the teachings of Rob Burbea" - primes the model for contemplative vocabulary

2. **Grounding constraint**: "based *only* on the provided reference material" - discourages hallucination

3. **Citation format**: Explicit instruction with example - the model knows to output [1], [2]

4. **Graceful failure**: "If the answer is not in the material, say so" - prevents confident fabrication

5. **Few-shot example**: Shows the expected format concretely

---

## What Could Be Improved

**1. The example is too simple**

"Breathing is good" → "You should practice breathing [1]"

Real chunks are dense contemplative material. The example doesn't demonstrate synthesizing across multiple sources or handling nuance.

**2. No guidance on tone**

Rob had a distinctive voice - warm, precise, exploratory, often circling back to qualify his statements. The prompt doesn't encourage this.

**3. No handling of apparent contradictions**

Rob sometimes said seemingly opposite things in different contexts (e.g., "let go of effort" vs. "apply more effort"). The prompt doesn't tell the LLM how to handle this.

**4. Citation granularity unclear**

Should the model cite after every sentence? Every claim? Every paragraph? Overcitation clutters; undercitation loses traceability.

---

## A More Nuanced Prompt

Here's one direction (not prescriptive - you'd tune this):

```python
system_prompt = """You are an assistant helping practitioners study Rob Burbea's teachings.

TASK: Answer the question using ONLY the provided reference passages. 

GUIDELINES:
- Synthesize across passages when they address the question from different angles
- Preserve nuance: Rob often qualified his statements ("it depends", "for some practitioners")
- If passages seem to conflict, acknowledge the tension rather than forcing resolution
- Cite sources with [1], [2] etc. after the relevant claim, not at end of paragraphs
- If the passages don't address the question, say so directly

TONE:
- Clear and practical, as if explaining to a fellow practitioner
- Avoid over-summarizing; Rob's precision matters

EXAMPLE:
Reference [1]: "Pīti can arise as a pleasant tingling, a sense of vibration..."
Reference [2]: "Sometimes pīti is intense, almost too much. Other times it's subtle."

Question: What does pīti feel like?

Answer: Pīti often manifests as pleasant tingling or vibration in the body [1]. 
Its intensity varies widely - sometimes it's almost overwhelming, other times barely 
noticeable [2]. Rob emphasizes this variability, suggesting you shouldn't expect 
a fixed experience.
"""
```

---

## The Context Window Budget

Your prompt consumes tokens. Let's count:

| Component | Approximate Tokens |
|-----------|-------------------|
| System prompt | ~150-300 |
| Retrieved chunks (5 × ~100 words) | ~400-600 |
| Query | ~20-50 |
| **Total input** | ~600-900 |
| Generated answer | ~200-400 |

With dolphin-mistral's 8K context, you have plenty of headroom. But if you later scale to more chunks or longer context, the prompt might need trimming.

---

## Prompt Variables to Experiment With

**Strictness of grounding:**

```
Strict:  "Use ONLY the provided passages. Do not add outside knowledge."
Loose:   "Primarily use the provided passages, supplementing with general context if needed."
```

For Rob's teachings, strict is probably right - you want the Expert to reflect *his* formulations, not generic dharma.

**Citation density:**

```
Dense:   "Cite after every factual claim."
Sparse:  "Cite at natural paragraph breaks."
Inline:  "Weave citations naturally: 'As Rob notes [1], pīti can...'"
```

**Handling uncertainty:**

```
Cautious:  "If uncertain, say 'The passages suggest...' rather than stating definitively."
Direct:    "State the answer clearly, then note any caveats."
```

**Length guidance:**

```
Concise:   "Answer in 2-3 sentences unless the question requires more."
Thorough:  "Provide a complete answer, typically 1-2 paragraphs."
Adaptive:  "Match your answer length to the question's complexity."
```

---

## The System Prompt Is a Lever

Unlike embeddings (baked into the model) or chunking (baked into the index), the system prompt changes instantly. You can A/B test prompts in a single session.

This makes it ideal for your pairwise comparison approach: same retrieval, same chunks, different prompts. Which produces better answers?

---

## Domain-Specific Considerations

Rob's teaching style has patterns a prompt could encourage:

1. **Phenomenological precision**: "What do you actually experience?" The prompt could encourage the LLM to foreground experiential description.

2. **Multiple valid approaches**: Rob often offered alternatives. The prompt could discourage false singularity ("The way to do this is...").

3. **Relationship with concepts**: Rob treated concepts as tools, not truths. The prompt could encourage holding ideas lightly.

4. **Poetic register**: Rob sometimes shifted into poetic language. Should the LLM mirror this when the source material does?

These are aesthetic and pedagogical choices. The prompt is where you encode them.

---

## Practical Experimentation

If you wanted to experiment with prompts:

1. **Baseline capture**: Run your probe set with the current prompt, save answers
2. **Draft 2-3 variants**: Stricter? Warmer? More citation guidance?
3. **Compare**: Same questions, different prompts, which do you prefer?
4. **Iterate**: Refine the winner

The system prompt is one of the highest-leverage, lowest-cost changes you can make.

---

## Temperature and Other Parameters

Beyond the prompt text, you have inference parameters:

```python
response = self.client.chat(
    model=self.model_name,
    messages=messages,
    options={
        "temperature": 0.2,      # Lower = more deterministic
        "top_p": 0.9,            # Nucleus sampling
    },
    stream=True,
)
```

**For RAG grounded in source material:** Low temperature (0.1-0.3) is usually better. You want faithful synthesis, not creative riffing.

---

## Prompt Laboratory

See `Goals.md` for the Prompt Laboratory goal - making prompts editable and switchable from the UI.

The idea:
- Store prompts in `prompts/` directory as TOML files
- Each includes: name, description, notes field, template
- Selector dropdown in Answer Generator sidebar
- In-app editor for live tuning

This would turn the Answer Generator into a prompt experimentation workbench.

---

## Example Prompt Variants

**Strict Grounding:**
```
You are a precise assistant. Answer ONLY from the provided passages. 
No outside knowledge. Cite heavily with [1], [2] after each claim.
```

**Conversational:**
```
You're helping a fellow practitioner explore Rob's teachings. 
Use the passages as your guide, but speak naturally as if in dialogue.
Cite key points with [1], [2] but don't overdo it.
```

**Phenomenological:**
```
Focus on experiential descriptions. When Rob describes a practice or 
state, emphasize what the practitioner might actually feel, sense, 
or notice. Ground everything in the body and direct experience [1], [2].
```

Each serves a different use case. The Prompt Laboratory lets you switch between them depending on what you're exploring.

---

## Configuration

Currently hardcoded in `src/llm.py`. With Prompt Laboratory, it would move to:

```
prompts/
  default.toml
  strict_grounding.toml
  conversational.toml
```

And `rb_expert.toml` would track:

```toml
[prompts]
active_prompt = "default"
```

---

## The Prompt-Model Interaction

Different models respond differently to the same prompt:
- dolphin-mistral: Good instruction following, might be overly direct
- gemma3n: Can be chatty, might need tighter constraints
- qwen: Strong reasoning, handles complex prompts well

The `notes` field in the Prompt Laboratory TOML would document these discoveries:

```toml
notes = """
Works well with dolphin-mistral for factual queries.
Tends to over-cite with gemma3n. Consider 'Conversational' prompt for that model.
"""
```

This is why Model Selection and Prompt Laboratory are adjacent goals - they interact.

---

## Next Steps

1. Read through your current prompt in `src/llm.py`
2. Draft 1-2 variants that address the improvements discussed above
3. Test them on your probe set
4. Document which works better for which query types
5. This learning feeds into the Prompt Laboratory design
