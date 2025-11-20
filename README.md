# Rob Burbea Expert

Local-first Retrieval Augmented Generation (RAG) for exploring Rob Burbea's dharma talks with uncensored Ollama-hosted LLMs—no cloud services required.

---

## Vision

Enable deep, citation-backed study of Rob Burbea's teachings by delivering:

- **Semantic search** across hundreds of talks
- **Question answering** powered by local LLMs
- **Comparative insights** between retreats and time periods
- **Traceable references** that point to specific talks and timestamps

_Current pilot scope: 32 talks from the 2019-2020 "Practising the Jhānas" retreat._

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
| **Streamlit UI (planned)** | Lightweight local dashboard for search + QA                          |

---

## Text Splitting Strategy

1. **Phase 1** – Split on paragraph boundaries (`\n\n`) to preserve semantic units.
2. **Phase 2** – Oversized paragraphs get overlap-aware sub-chunks (word/line boundary aware).

Fallback: LangChain's `RecursiveCharacterTextSplitter` (toggle via `rag.use_langchain_splitter`).

---

## Setup

### Requirements

- Python **3.11+** (uses `tomllib`)
- [Ollama](https://ollama.ai) installed with desired models downloaded
- ~24 GB RAM recommended for heavier models
- macOS/Linux (Windows via WSL works)

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Configuration (`rb_expert.toml`)

```toml
[paths]
data_dir = "data"
raw_talks_dir = "data/raw_talks"
chroma_db_dir = "data/chroma_db"
metadata_path = "data/raw_talks/metadata.json"

[models]
embedding_model = "all-MiniLM-L6-v2"
default_llm_model = "dolphin-mistral:7b"

[rag]
chunk_size = 500
chunk_overlap = 50
use_langchain_splitter = false
top_k_results = 5
similarity_threshold = 0.7

[ollama]
base_url = "http://localhost:11434"
timeout = 60
```

Reference template: `tests/fixtures/test_env.toml`.

---

## Project Structure

```
rob-burbea-expert/
├── src/
│   ├── env.py          # Config loading (TOML) with Pydantic validation
│   ├── models.py       # Embedding factory (real + fake for testing)
│   ├── data_prep.py    # Text ingestion + chunking
│   ├── database.py     # ChromaDB connector wrapper
│   └── indexing.py     # Full indexing pipeline
├── tests/
│   ├── test_env.py         # Configuration tests
│   ├── test_models.py      # Embedding function tests
│   ├── test_data_prep.py   # Data loading and chunking tests
│   ├── test_database.py    # ChromaDB operations tests
│   ├── test_indexing.py    # Full indexing pipeline tests
│   └── test_retrieval.py   # Retrieval validation and performance tests
├── tests/fixtures/
│   └── data/
│       ├── raw_talks/      # 32 markdown transcripts for testing
│       └── metadata.json   # Talk metadata
├── tmp/                    # Test databases (gitignored, for inspection)
│   ├── chroma_db_database/
│   ├── chroma_db_indexing/
│   └── chroma_db_retrieval/
├── tools/                  # Utility scripts
└── data/                   # Runtime artifacts (gitignored)
```

---

## Development Approach

1. **Small, representative dataset** (2-3 talks) for fast iteration and unit tests.
2. **Jhana retreat subset** (32 talks, ~4,900 chunks) for validation and integration testing.
3. **Full corpus** (548 talks) once indexing + retrieval pipeline are battle-tested.
4. **Streamlit UI** after backend stabilizes.

_Principle:_ every component should be independently testable and explainable.

---

## Current Status

| Feature                                   | Status |
|-------------------------------------------|--------|
| Project scaffolding & config              | ✅     |
| Semantic-first splitter + LangChain check | ✅     |
| Fake + real embedding factory (list-safe) | ✅     |
| ChromaDB indexing with deterministic IDs  | ✅     |
| Full embedding generation pipeline        | ✅     |
| RAG retrieval validation & testing        | ✅     |
| Query performance benchmarking            | ✅     |
| Ollama LLM integration                    | ⬜     |
| Context building for LLM prompts          | ⬜     |
| CLI interface                             | ⬜     |
| Streamlit UI                              | ⬜     |

**All 32 tests passing** ✅

---

## Testing

```bash
# Run all tests
pytest

# Quiet mode (summary only)
pytest -q

# Verbose with print statements
pytest -v -s

# Run specific test module
pytest tests/test_retrieval.py -v -s

# Show slowest tests
pytest -v --durations=10
```

### Test Coverage

- **Configuration**: TOML parsing, path validation, model settings
- **Embeddings**: Determinism, dimension validation, mock vs real
- **Data Processing**: Markdown loading, chunking, metadata extraction
- **Database**: ChromaDB operations, collection management
- **Indexing**: Full pipeline with ~4,900 chunks from 32 talks
- **Retrieval**: Semantic search, relevance validation, performance benchmarking

Typical runtime: ~25 seconds on dev machine (includes model download on first run).

---

## Database Inspection

Test runs persist ChromaDB files in `tmp/` for inspection:

```bash
# View directory structure
ls -lh tmp/chroma_db_retrieval/

# The database is SQLite-based
sqlite3 tmp/chroma_db_retrieval/chroma.sqlite3 .schema
```

Each test module uses its own isolated database to prevent dimension conflicts.

---

## Key Implementation Details

### Deterministic Chunk IDs

Every chunk gets a stable ID: `{filename_stem}_{index}`

Example: `2019-12-17-orienting-to-this-jhana-retreat_0`

This enables:
- Idempotent re-indexing (upsert semantics)
- Consistent cross-run references
- Traceable citations back to source talks

### Embedding Validation

`test_retrieval.py` validates:
- Same text → same embedding (determinism)
- Different texts → different embeddings (sensitivity)
- Query results are ordered by similarity
- Metadata (source file) is preserved
- Performance benchmarks (~10-50ms per query)

### Test Database Isolation

Each test suite uses a dedicated ChromaDB directory:
- `test_database.py` → `tmp/chroma_db_database/`
- `test_indexing.py` → `tmp/chroma_db_indexing/`
- `test_retrieval.py` → `tmp/chroma_db_retrieval/`

This prevents dimension mismatches between mock (3D) and real (384D) embeddings.

---

## Next Steps

1. **CLI Query Interface**: Simple command-line tool to query the indexed talks
2. **Context Builder**: Assemble relevant chunks + metadata for LLM prompts
3. **Ollama Integration**: Send constructed prompts to local LLM
4. **Citation Formatter**: Link responses back to specific talks and timestamps
5. **Streamlit Dashboard**: Visual interface for exploration

---

## Related Work

- Hermes Amara Foundation: https://hermesamara.org
- Talk transcriptions: 548 high-quality Markdown files with YAML frontmatter

---

## Notes

This is a private, local-only system: all processing happens on-device with no external API calls.
