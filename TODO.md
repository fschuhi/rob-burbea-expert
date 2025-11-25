# TODO

Tactical scratchpad for quick fixes, small refactors, and tasks under ~1 hour. For strategic work with intent/rationale/definition of done, see [`Goals.md`](Goals.md).

---

### --- Data & Indexing ---

- **Noise Reduction** - Filter out timestamps (e.g., `[28:02]`) and non-content markers (e.g., `[laughter]`, `[inaudible]`) during chunking in `data_prep.py` to prevent low-value "island" chunks.
- **Complete Rob** - Prepare the full Rob Burbea dataset (all retreats) alongside the Practicing the Jhānas pilot.

### --- Features ---

- **Data Liberation** - Add export functionality to save search results to markdown/JSON.
- **Etched in Stone** - Implement a "Save as Default" feature that writes current slider values back to `rb_expert.toml`.
- **Slice & Dice** - Implement retreat/date range filters in Search Explorer sidebar.

### --- UI Polish ---

- **Branding Beacon** - Replace the text title "Rob Burbea Talks" in the sidebar with a configurable image (e.g., Rob's photo or a logo) in both apps.

### --- Code Quality ---

- **Silencing the Linter** - Fix "Type of '__init__' is incompatible with 'EmbeddingFunction'" warnings in `src/models.py` (PyCharm strict type checking issue).
- **Zero Warnings** - Sweep through all code and fix remaining PyCharm warnings.
