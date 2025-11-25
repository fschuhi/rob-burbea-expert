# Streamlit: Building Interactive UIs for RAG

## What Streamlit Is

Streamlit turns Python scripts into interactive web apps with minimal boilerplate. No HTML, no CSS (unless you want it), no JavaScript. Just write Python top-to-bottom, and Streamlit handles the rest.

**The promise:** Focus on your logic (retrieval, ranking, generation), not on web framework plumbing.

**The reality:** It delivers on that promise, with some quirks you need to understand.

Your two apps demonstrate the full range:
- **Search Explorer**: Classic interactive tool - inputs, outputs, settings
- **Answer Generator**: Advanced chat interface with streaming and state management

---

## The Mental Model: Top-to-Bottom Execution

**Every interaction reruns your entire script from top to bottom.**

This is the key mental shift from traditional web frameworks.

```python
# search_explorer.py - simplified
import streamlit as st

st.title("Search Explorer")
query = st.text_input("Search:")  # Widget

if query:
    results = search_database(query)  # Your logic
    st.write(results)                 # Display
```

**What happens:**
1. User loads page → Script runs top-to-bottom
2. User types in text box → **Script reruns top-to-bottom**
3. User clicks button → **Script reruns top-to-bottom**

Every widget interaction triggers a full rerun. Streamlit preserves widget state between reruns, but your script code executes fresh each time.

---

## Caching: The Performance Escape Hatch

If everything reruns, how do you avoid reloading the database on every keystroke?

**Answer:** `@st.cache_resource`

From Search Explorer:

```python
@st.cache_resource
def get_engine():
    """Initialize the RAG Engine (cached)."""
    env = load_env()
    return RAGEngine(env)
```

This decorator says: "Run this function once, cache the result, reuse it on subsequent reruns."

**Use for:**
- Loading models (embeddings, cross-encoders, LLMs)
- Opening database connections
- Reading large config files

**Don't use for:**
- User input (it needs to change each run)
- Query results (they depend on the query)

The cache persists as long as the Streamlit server runs. Restart the server → cache clears → models reload.

---

## Layout: Page Configuration

From Answer Generator:

```python
st.set_page_config(
    page_title="Rob Burbea Expert - AI Chat",
    page_icon="🧘",
    layout="wide"
)
```

**Must be first Streamlit command** (before any other `st.*` calls).

Options:
- `layout="wide"`: Use full browser width (vs centered narrow column)
- `page_icon`: Shows in browser tab
- `initial_sidebar_state`: "expanded" or "collapsed"

Then constrain with CSS (Answer Generator does this):

```python
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1000px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
```

Pattern: Use `layout="wide"` for control, then CSS to set comfortable reading width.

---

## The Sidebar

Both apps use `with st.sidebar:` to create the left panel.

From Search Explorer:

```python
with st.sidebar:
    st.header("📊 Database")
    st.metric("Total Chunks", collection.count())
    
    st.markdown("---")  # Horizontal divider
    st.header("⚙️ Tuning")
    
    n_results = st.slider(
        "Retrieval Pool Size",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
        help="How many chunks to fetch from the Vector Database."
    )
```

**Patterns:**
- `st.metric()`: Big number displays
- `st.markdown("---")`: Horizontal line separators
- `st.slider()`: Interactive controls
- `help=` parameter: Hover tooltips

Everything inside `with st.sidebar:` goes to the left panel. Everything outside goes to the main area.

---

## Input Widgets

**Text input** (Search Explorer):

```python
query = st.text_input(
    "🔍 Search:",
    value=st.session_state.search_query,  # Preserve across reruns
    placeholder="e.g., 'energy body practice'"
)
```

**Sliders** (both apps):

```python
top_k = st.slider(
    "Final Context Chunks",
    1, 15, 
    engine.env.rag.top_k_results,  # Default value
    help="How many chunks to send to the LLM."
)
```

**Checkboxes** (Search Explorer):

```python
use_reranker = st.checkbox(
    "Apply Reranker",
    value=False,
    help="Sort results by Semantic Relevance"
)
```

**Selectbox** (Answer Generator - model selection):

```python
selected_model = st.selectbox(
    "Active LLM",
    options=model_names,
    index=default_idx,
    help="Select which Ollama model to use"
)
```

**All of these return values immediately.** No form submission needed. Change the slider → value changes → script reruns with new value.

---

## Displaying Results

**Simple text:**

```python
st.write("Hello")
st.markdown("**Bold** text")
st.caption("Small grey text")
```

**Metrics:**

```python
st.metric("Speed", "15 words/s")
```

**Code blocks:**

```python
st.code("print('hello')", language="python")
```

**JSON:**

```python
st.json({"distance": 0.34})
```

**HTML** (Search Explorer uses this for custom cards):

```python
st.markdown(f'<div class="result-card">{html_content}</div>', unsafe_allow_html=True)
```

The `unsafe_allow_html=True` flag lets you inject custom HTML/CSS. Use sparingly - it's an escape hatch when Streamlit's built-ins don't suffice.

---

## Layout Patterns: Columns and Expanders

**Columns** (Answer Generator suggestion buttons):

```python
col1, col2, col3 = st.columns(3)
if col1.button("How do I work with the energy body?"):
    st.session_state.example_input = "..."
if col2.button("What is the role of pīti?"):
    st.session_state.example_input = "..."
```

**Expanders** (collapsible sections):

From Answer Generator:

```python
with st.expander(f"📚 Sources ({len(citation_ids)})"):
    for ref_id in citation_ids:
        st.markdown(f"**[{ref_id}] {source_name}**")
        st.markdown(reconstruction)
        st.divider()
```

Default: collapsed. Add `expanded=True` to start open.

---

## Advanced: Session State

Streamlit reruns your script on every interaction. How do you preserve data between reruns?

**Session state** - a persistent dictionary:

From Answer Generator:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# Later: append to it
st.session_state.messages.append({
    "role": "assistant",
    "content": full_response,
    "citations": used_ids
})

# Even later: read from it
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).markdown(msg["content"])
```

**Pattern:** Check if key exists, initialize if not, then read/write freely.

Session state persists as long as the browser tab stays open. Close tab → state clears. Server restart → state clears.

---

## Advanced: Chat Interface

Answer Generator uses Streamlit's chat components:

```python
# Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box (sticky at bottom)
prompt = st.chat_input("Ask a question...")

if prompt:
    # Display user message
    st.chat_message("user").markdown(prompt)
    
    # Display assistant message
    with st.chat_message("assistant"):
        st.markdown("Thinking...")
```

`st.chat_message("user")` gives you the user icon/styling. `st.chat_message("assistant")` gives you the assistant styling.

The `st.chat_input()` widget is sticky at the bottom, like ChatGPT.

---

## Advanced: Streaming Output

The hardest pattern in Answer Generator - showing text as it generates:

```python
with st.chat_message("assistant"):
    answer_box = st.empty()  # Placeholder
    
    full_response = ""
    for chunk in stream:
        full_response += chunk
        answer_box.markdown(full_response + "▌")  # Blinking cursor
    
    answer_box.markdown(full_response)  # Final (no cursor)
```

**Key concept: `st.empty()`** - a placeholder you can update repeatedly.

Without `st.empty()`, each `st.markdown()` call would append a new element (100 markdown boxes). With `st.empty()`, you overwrite the same spot.

The `+ "▌"` adds a blinking cursor effect during generation.

---

## Advanced: Multiple Placeholders

Answer Generator uses TWO placeholders simultaneously:

```python
telemetry_box = st.empty()
answer_box = st.empty()

# Update telemetry
render_telemetry_box(telemetry_box, telemetry, final=False)

# Stream answer
for chunk in stream:
    full_response += chunk
    answer_box.markdown(full_response + "▌")

# Final telemetry update
render_telemetry_box(telemetry_box, telemetry, final=True)
```

This creates the effect of:
1. Telemetry showing "Processing..."
2. Answer streaming in
3. Telemetry updating to "Finished in 15s"

All without the page jumping around.

---

## Spinners and Status Indicators

**Spinner** (Search Explorer):

```python
with st.spinner("Searching..."):
    results = collection.query(query_texts=[query], n_results=n_results)
```

Shows a rotating spinner with your message while the code block runs.

**Status boxes** (Answer Generator sidebar):

```python
if apply_reranker:
    st.info("Reranker is ACTIVE.")
else:
    st.warning("Reranker is OFF.")
```

Colors: `st.info()` (blue), `st.success()` (green), `st.warning()` (orange), `st.error()` (red).

---

## Custom CSS

Both apps inject custom CSS to override Streamlit's defaults.

From Search Explorer:

```python
st.markdown(
    """
    <style>
    /* Custom hit highlighting */
    hit {
        color: #7CFC00 !important;
        background-color: rgba(124, 252, 0, 0.1);
        border-radius: 3px;
        padding: 0 2px;
    }
    
    /* Tighter spacing */
    hr {
        margin: 0.15rem 0 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
```

Then in your HTML:

```python
html_content = markdown.markdown(display_text)
st.markdown(html_content, unsafe_allow_html=True)
```

The `<hit>` tag gets styled by your CSS. Streamlit doesn't sanitize HTML when `unsafe_allow_html=True`, so you have full control.

---

## Buttons and Callbacks

**Simple button** (Search Explorer):

```python
if st.button("🔍 energy body meditation", key="btn_0"):
    st.session_state.search_query = "energy body meditation"
    st.rerun()
```

Button press returns `True` once, then `False` on subsequent reruns. The pattern: detect the press, update session state, force a rerun to show the new state.

**`st.rerun()`** - programmatically triggers a full script rerun. Use after updating session state to immediately reflect changes.

The `key=` parameter gives the button a unique identifier. Required when you have multiple buttons in a loop.

---

## Debugging Streamlit Apps

**Problem:** Everything reruns constantly. How do you debug?

**Techniques:**

1. **Print statements:**
   ```python
   st.write(f"Debug: query = {query}")
   ```
   They show in the app UI, not console.

2. **Session state inspection:**
   ```python
   with st.sidebar:
       with st.expander("🐛 Debug"):
           st.write(st.session_state)
   ```

3. **Conditional debug panels:**
   ```python
   if show_chunk_debug:
       st.json(result["metadata"])
   ```
   Toggle with a checkbox.

4. **Server logs:** The terminal running `streamlit run` shows Python exceptions.

---

## Common Pitfalls

**1. Expensive operations in the main script**

❌ **Bad:**
```python
model = load_huge_model()  # Runs on EVERY widget interaction
query = st.text_input("Search:")
```

✅ **Good:**
```python
@st.cache_resource
def get_model():
    return load_huge_model()

model = get_model()  # Cached
query = st.text_input("Search:")
```

**2. Stale session state**

If you modify `st.session_state.foo` but don't see changes, you forgot to `st.rerun()`.

**3. Widget key conflicts**

If you dynamically create widgets in a loop, each needs a unique `key=`:

```python
for idx, example in enumerate(examples):
    if st.button(example, key=f"btn_{idx}"):
        ...
```

**4. CSS specificity wars**

Your custom CSS might not override Streamlit's. Use `!important`:

```python
strong { color: #FFD700 !important; }
```

---

## When Streamlit Fits (and When It Doesn't)

**Streamlit is great for:**
- Internal tools (your Rob Burbea Expert)
- Prototyping ML/data apps quickly
- Demos and dashboards
- When Python is your strength and web dev isn't

**Streamlit is awkward for:**
- High-traffic public apps (it's not optimized for scale)
- Complex multi-page flows (routing is basic)
- Real-time collaboration (state is per-session)
- Fine-grained UX control (you're working within Streamlit's patterns)

For your use case (local-first, single-user RAG exploration), Streamlit is perfect.

---

## Performance Considerations

Your apps are both efficient:

**Search Explorer:**
- Loads engine once (`@st.cache_resource`)
- Query execution is fast (~0.6s including reranking)
- Reconstruction happens only when showing full paragraphs

**Answer Generator:**
- Loads engine once
- Streaming gives perceived speed (first token in ~8s)
- Session state keeps history without re-retrieving

**Bottleneck:** Not Streamlit, but the LLM generation (7-15s). That's hardware, not framework.

---

## The Search Explorer Pattern

Let's trace a full interaction:

```python
# 1. Setup (runs once per rerun)
engine = get_engine()  # Cached

# 2. Sidebar (renders every time)
with st.sidebar:
    n_results = st.slider("Retrieval Pool Size", 5, 100, 20)
    use_reranker = st.checkbox("Apply Reranker", value=False)

# 3. Input
query = st.text_input("🔍 Search:", placeholder="...")

# 4. Execution (only if query exists)
if query:
    with st.spinner("Searching..."):
        results = collection.query(query_texts=[query], n_results=n_results)
    
    # Optional reranking
    if use_reranker:
        scores = engine.cross_encoder.predict(pairs)
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)
    
    # 5. Display
    for result in candidates:
        st.markdown(f"#{idx} · {source}")
        st.markdown(display_text, unsafe_allow_html=True)
```

Simple, linear flow. No routing, no complex state management. Just: get input → process → display.

---

## The Answer Generator Pattern

More complex due to chat history:

```python
# 1. Setup
engine = get_engine()  # Cached

if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Display history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander("📚 Sources"):
                # Show citations

# 3. New input
prompt = st.chat_input("Ask a question...")

if prompt:
    # 4. Save user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 5. Generate response with streaming
    with st.chat_message("assistant"):
        answer_box = st.empty()
        
        stream = engine.llm_client.stream_answer(prompt, context)
        full_response = ""
        
        for chunk in stream:
            full_response += chunk
            answer_box.markdown(full_response + "▌")
        
        answer_box.markdown(full_response)
    
    # 6. Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # 7. Force rerun to show updated history
    st.rerun()
```

The session state accumulates history. Each rerun displays the full history, then adds new messages at the bottom.

---

## Customizing the Look

**Answer Generator's citation styling:**

```python
st.markdown(
    """
    <style>
    strong { color: #FFD700 !important; font-weight: 900 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
```

Then in the text:

```python
def ensure_bold_citations(text: str) -> str:
    """Wraps citation tags [N] in bold markers if not already bold."""
    return re.sub(r"(?<!\*\*)\[(\d+)\](?!\*\*)", r"**[\1]**", text)
```

Result: `**[1]**` renders in gold, making citations pop visually.

**Search Explorer's hit highlighting:**

```python
<style>
hit {
    color: #7CFC00 !important;
    background-color: rgba(124, 252, 0, 0.1);
}
</style>
```

Then `<hit>energy body</hit>` in the HTML renders in lime green with subtle background.

---

## Configuration

No explicit Streamlit config file. Settings in the code:

- Page config: `st.set_page_config()`
- CSS: `st.markdown("<style>...</style>", unsafe_allow_html=True)`
- Widget defaults: `value=` parameters

To change themes (light/dark), users configure Streamlit globally:

```bash
# ~/.streamlit/config.toml
[theme]
primaryColor = "#FFD700"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

But for your local-first use case, inline CSS is simpler.

---

## Testing Streamlit Apps

Streamlit apps are harder to unit test because they're tightly coupled to the UI.

**Strategies:**

1. **Extract business logic:**
   ```python
   # Good: Testable
   def format_source(source_path: str) -> str:
       return Path(source_path).stem.replace("-", " ").title()
   
   # In the app:
   formatted = format_source(result["metadata"]["source"])
   st.markdown(formatted)
   ```

2. **Mock Streamlit:**
   You can mock `st.*` calls in tests, but it's brittle.

3. **Manual testing:**
   Run the app, click around. Your apps are simple enough that this works.

Your pattern of keeping logic in `src/` and UI in `apps/` is good - it makes the logic testable independently.

---

## Next Steps

1. **Read both apps top-to-bottom** - now that you understand the patterns, the code should be much clearer

2. **Experiment with widgets** - try adding a date filter or text area, see how reruns work

3. **Debug with session state inspection** - add a debug expander that shows `st.session_state` to see what persists

4. **Explore Streamlit docs** - now that you have the mental model, their docs will make more sense: https://docs.streamlit.io

5. **Consider Alternative UI Prototype** - from `Goals.md`. After deep Streamlit experience, you might want to try a different framework (CLI, web, etc.) to see what Streamlit's constraints are hiding.

---

## The 80/20 Takeaway

**Streamlit's value:** Turns Python scripts into interactive apps with minimal overhead. Perfect for data/ML tooling where Python is your strength.

**The rerun model:** Everything reruns on every interaction. Cache expensive operations. Use session state for persistence. Use `st.empty()` for dynamic updates.

**Your apps demonstrate:** Search Explorer shows the basics cleanly. Answer Generator shows advanced patterns (streaming, chat, state) that push Streamlit's boundaries.

**When it fits:** Internal tools, prototypes, demos. Not for high-scale production web apps.

You built two sophisticated RAG interfaces in ~3 days, including the entire backend pipeline. That's the Streamlit value proposition - rapid development without drowning in web framework complexity.
