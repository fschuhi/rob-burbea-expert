# Rob Burbea Expert

Local-first Retrieval Augmented Generation (RAG) for exploring Rob Burbea’s dharma talks with uncensored Ollama-hosted LLMs—no cloud services required.

---

## Vision

Enable deep, citation-backed study of Rob Burbea’s teachings by delivering:

- **Semantic search** across hundreds of talks
- **Question answering** powered by local LLMs
- **Comparative insights** between retreats and time periods
- **Traceable references** that point to specific talks and timestamps

_Current pilot scope: 30 talks from the 2019 “Practicing the Jhanas” retreat._

---

## Architecture Overview

```
User Query ──► Embedding ──► ChromaDB Retrieval ──► Context Builder ──► Ollama LLM ──► Response
```

| Component                  | Role                                                                 |
|----------------------------|----------------------------------------------------------------------|
| **sentence-transformers**  | Produces embeddings locally (MiniLM by default)                      |
| **ChromaDB**               | Vector store with deterministic metadata + ID handling               |
| **Custom splitter**        | Two-phase, semantic-first chunking strategy                          |
| **Ollama**                 | Runs local LLMs (tested: `dolphin-mistral`, `gemma3n-abliterated`)   |
| **Streamlit UI (planned)** | Lightweight local dashboard for search + QA                         |

---

## Text Splitting Strategy

1. **Phase 1** – Split on paragraph boundaries (`\n\n`) to preserve semantic units.
2. **Phase 2** – Oversized paragraphs get overlap-aware sub-chunks (word/line boundary aware).

Fallback: LangChain’s `RecursiveCharacterTextSplitter` (toggle via `rag.use_langchain_splitter`).

---

## Setup

### Requirements

- Python **3.11+** (uses `tomllib`)
- [Ollama](https://ollama.ai) installed with desired models downloaded
- ~24 GB RAM recommended for heavier models
- macOS/Linux (Windows via WSL works)

### Installation

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Configuration (`rb_expert.toml`)

```
[paths]
data_dir = "data"
raw_talks_dir = "data/raw_talks"
chroma_db_dir = "data/chroma_db"
metadata_path = "data/raw_talks/metadata.json"

[rag]
chunk_size = 500
chunk_overlap = 50
use_langchain_splitter = false
```

Reference template: `tests/fixtures/test_env.toml`.

---

## Project Structure

```
rob-burbea-expert/
├── src/
│   ├── env.py          # Config loading (TOML)
│   ├── data_prep.py    # Text ingestion + splitting
│   ├── models.py       # Embedding factory (real + fake)
│   └── indexing.py     # ChromaDB indexing pipeline
├── tests/
│   ├── test_env.py
│   ├── test_data_prep.py
│   ├── test_models.py
│   └── test_indexing.py
├── tools/              # Utility scripts
├── data/               # Runtime artifacts (gitignored)
└── notebooks/          # Exploratory analysis
```

---

## Development Approach

1. **Small, representative dataset** (3 talks) for fast iteration and fixtures.
2. **Jhana retreat subset** (30 talks) for intermediate validation.
3. **Full corpus** (548 talks) once indexing + metadata pipeline are battle-tested.
4. **Streamlit UI** after backend stabilizes.

_Principle:_ every component should be independently testable and explainable.

---

## Current Status

| Feature                                   | Status |
|-------------------------------------------|--------|
| Project scaffolding & config              | ✅     |
| Semantic-first splitter + LangChain check | ✅     |
| Fake + real embedding factory (list-safe) | ✅     |
| ChromaDB indexing (tests: 21/21 passing)  | ✅     |
| Excel metadata extraction tool            | ⬜     |
| Full embedding generation pipeline        | ⬜     |
| RAG retrieval orchestration               | ⬜     |
| Ollama UI integration                     | ⬜     |

---

## Testing

```
pytest tests/              # Full suite
pytest -q                  # Quiet
pytest -v --durations=10   # Highlight slowest tests
```

Typical runtime: ~3 s on dev machine (manual splitter path).

---

## Related Work

- Hermes Amara Foundation: https://hermesamara.org
- Talk transcriptions: 548 high-quality Markdown files
- Metadata: Excel spreadsheet with retreat info, dates, series

---

## Notes

This is a private, local-only system: all processing happens on-device with no external API calls.

