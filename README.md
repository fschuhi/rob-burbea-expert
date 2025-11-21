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
| **sentence-transformers** | Produces embeddings locally (MiniLM by default)                      |
| **ChromaDB** | Vector store with deterministic metadata + ID handling               |
| **Custom splitter** | Two-phase, semantic-first chunking strategy                          |
| **Context Builder** | Assembles full paragraphs with XML metadata for the LLM              |
| **Ollama** | Runs local LLMs (tested: `dolphin-mistral`, `gemma3n-abliterated`)   |
| **Streamlit UI** | Lightweight local dashboard for search with full paragraph context   |

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

## Makefile Targets

```bash
# Setup and dependencies
make setup          # Install dependencies and project

# Testing
make test           # Run all tests (quiet mode)
make test-verbose   # Run tests with full output
make test-fast      # Skip slow indexing tests

# Application
make app            # Launch Search Explorer Streamlit app
make chat           # Launch Answer Generator AI app

# Data Management
make ingest-pilot   # Copy pilot data to data/ and index it (PRODUCTION DB)
make index          # Index pilot data to tmp/ (TEST DB only)

# Ollama
make startollama    # Safely start Ollama in the background
make killollama     # Stop the running Ollama process

# Utilities
make clean          # Remove venv, caches, and tmp databases
make showtree       # Display project structure
make gentree        # Save project tree to project-tree.txt
make filesdump      # Concatenate files for LLM context
```

---

## Project Structure

```
rob-burbea-expert/
├── src/
│   ├── engine.py       # RAG Orchestrator (RAGEngine)
│   ├── context.py      # RAG context assembly (ContextBuilder)
│   ├── llm.py          # Ollama API Client
│   ├── data_prep.py    # Text ingestion + chunking with paragraph tracking
│   ├── database.py     # ChromaDB connector + paragraph reconstruction
│   ├── env.py          # Config loading (TOML) with Pydantic validation
│   ├── indexing.py     # Full indexing pipeline
│   └── models.py       # Embedding factory (real + fake for testing)
├── apps/
│   ├── answer_generator.py # Streamlit AI Chat interface
│   └── search_explorer.py  # Streamlit semantic search interface
├── tests/
│   ├── conftest.py         # Shared fixtures (real DB integration)
│   ├── test_engine.py      # RAG Engine integration tests
│   ├── test_llm.py         # Ollama Client tests
│   ├── test_context.py     # Context builder tests
│   ├── test_data_prep.py   # Data loading and chunking tests
│   ├── test_database.py    # ChromaDB operations + paragraph reconstruction tests
│   ├── test_env.py         # Configuration tests
│   ├── test_indexing.py    # Full indexing pipeline tests
│   ├── test_models.py      # Embedding function tests
│   └── test_retrieval.py   # Retrieval validation and performance tests
├── tests/fixtures/
│   └── data/
│       ├── raw_talks/      # 32 markdown transcripts for testing
│       └── metadata.json   # Talk metadata
├── tmp/                    # Test databases (gitignored, for inspection)
│   ├── chroma_db_database/
│   ├── chroma_db_indexing/
│   ├── chroma_db_retrieval/
├── tools/                  # Utility scripts
│   └── concat_files.py     # File concatenation for LLM context
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
| Paragraph metadata tracking               | ✅     |
| Fake + real embedding factory (list-safe) | ✅     |
| ChromaDB indexing with deterministic IDs  | ✅     |
| Full embedding generation pipeline        | ✅     |
| RAG retrieval validation & testing        | ✅     |
| Query performance benchmarking            | ✅     |
| Search Explorer Streamlit app             | ✅     |
| Paragraph reconstruction with highlighting| ✅     |
| Context building for LLM prompts          | ✅     |
| Ollama LLM integration                    | ✅     |
| Answer Generator Streamlit app            | ✅     |
| CLI interface                             | ⬜     |

**All 42 tests passing** ✅

**Legend:** ✅ Complete | 🚧 In Progress | ⬜ Planned

---

## Testing

```bash
# Run all tests
pytest

# Quiet mode (summary only)
pytest -q

# Verbose with print statements
pytest -v -s
```

### Test Coverage

- **Configuration**: TOML parsing, path validation
- **Embeddings**: Determinism, mock vs real
- **Data Processing**: Markdown loading, chunking, metadata extraction
- **Database**: ChromaDB operations, collection management
- **Context Building**: Formatting, distance filtering, metadata injection
- **Indexing**: Full pipeline with ~4,900 chunks from 32 talks
- **Retrieval**: Semantic search, relevance validation
- **RAG Engine**: Full integration tests (DB -> Context -> LLM Mock)

**Note:** We use `conftest.py` to spin up temporary, isolated ChromaDB environments for robust integration testing without relying on fragile mocks for the database layer.

---

## Search Explorer App

Interactive Streamlit interface for exploring semantic search results:

```bash
# Run the app (recommended)
make app
```

**Features:**
- Semantic search across 32 indexed talks (~4,800 chunks)
- **Full paragraph reconstruction** with matched chunk highlighting (light green)
- **Dense UI Layout** optimized for rapid scanning of search results
- Adjustable distance threshold for result filtering
- Distance metrics displayed prominently on each result (bold)
- Optional chunk debug information (paragraph index, position, total chunks)
- Real-time query performance

---

## Answer Generator App

Chat-based RAG interface for asking questions:

```bash
# Run the app
make chat
```

**Features:**
- **Conversational Interface**: Ask natural language questions.
- **Live Streaming**: Watch the answer type out in real-time.
- **Source Transparency**: Every answer includes an expandable "View Context Used" section showing exactly what paragraphs (with highlighting) were sent to the LLM.
- **Model-Agnostic**: Uses whichever model is configured in `rb_expert.toml` (e.g., Dolphin Mistral).

---

## Key Implementation Details

### Deterministic Chunk IDs

Every chunk gets a stable ID: `{filename_stem}_{index}`
Example: `2019-12-17-orienting-to-this-jhana-retreat_0`

This enables idempotent re-indexing and traceable citations.

### Paragraph Metadata Tracking

Each chunk includes metadata to reconstruct its original context:
```python
{
    "source": "path/to/talk.md",
    "paragraph_index": 12,
    "chunk_position": 1,
    "total_chunks_in_para": 3
}
```

### Paragraph Reconstruction & Context Building

The system uses two layers of reconstruction:

1.  **UI Layer (`database.py`)**: Reconstructs full paragraphs for the Streamlit app, highlighting the specific chunk that triggered the search hit.
2.  **LLM Layer (`context.py`)**: The `ContextBuilder` assembles retrieval results into a structured system prompt. It:
    * Filters hits by distance threshold.
    * Sorts by relevance.
    * Retrieves full paragraphs via `get_paragraph_chunks`.
    * Applies `<hit distance="0.25">...</hit>` XML tags so the LLM can distinguish specific evidence from surrounding context.

---

## Next Steps

1.  **CLI Query Interface**: Simple command-line tool to query the indexed talks.
2.  **Citation Formatter**: Link responses back to specific talks and timestamps.
3.  **Advanced Search Features**: Filters by retreat, date range, or speaker.

---

## Notes

This is a private, local-only system: all processing happens on-device with no external API calls.
