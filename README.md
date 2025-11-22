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
| **Context Builder** | Assembles paragraphs and maps them to Reference IDs `[1]`, `[2]`     |
| **Ollama** | Runs local LLMs (tested: `dolphin-mistral`, `gemma3n-abliterated`)   |
| **Search Explorer** | Streamlit app for deep semantic search and exploration               |
| **Answer Generator** | Chat interface for Q&A with strictly cited evidence                  |

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

Reference template: `tests/fixtures/test_env.toml`.

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
top_k_results = 5
similarity_threshold = 0.7

[ollama]
base_url = "http://localhost:11434"
timeout = 60
```

---

## Makefile Targets

```bash
# Setup and dependencies
make setup          # Install dependencies and project

# Testing
make test           # Run all tests (quiet mode)
make test-verbose   # Run tests with full output
make test-fast      # Skip slow indexing tests

# Application (Foreground - Blocks Terminal)
make app            # Launch Search Explorer (Port 8501)
make chat           # Launch Answer Generator (Port 8502)

# Application (Background - Daemon Mode)
make startapp       # Start Search Explorer in background
make killapp        # Stop Search Explorer
make startchat      # Start Answer Generator in background
make killchat       # Stop Answer Generator

# Data Management
make ingest-pilot   # Copy pilot data to data/ and index it (PRODUCTION DB)
make index          # Index pilot data to tmp/ (TEST DB only)

# Ollama Management
make startollama    # Safely start Ollama in the background
make killollama     # Stop the running Ollama process

# Utilities
make clean          # Remove venv, caches, and tmp databases
make showtree       # Display project structure
make gentree        # Save project tree to project-tree.txt
```

---

## Project Structure

```
rob-burbea-expert/
├── src/
│   ├── engine.py       # RAG Orchestrator (RAGEngine)
│   ├── context.py      # Context assembly & ID mapping
│   ├── llm.py          # Ollama API Client & Prompt Engineering
│   ├── data_prep.py    # Text ingestion + chunking
│   ├── database.py     # ChromaDB connector + paragraph reconstruction
│   ├── env.py          # Config loading (TOML)
│   ├── indexing.py     # Full indexing pipeline
│   └── models.py       # Embedding factory (real + fake)
├── apps/
│   ├── answer_generator.py # Chat interface with footnote citations
│   └── search_explorer.py  # Search interface with green highlighting
├── tests/
│   ├── conftest.py         # Shared fixtures (real DB integration)
│   ├── test_engine.py      # Integration tests
│   └── ... (unit tests for all modules)
├── data/                   # Runtime artifacts (gitignored)
└── tools/                  # Utility scripts
```

---

## Current Status

| Feature                                   | Status |
|-------------------------------------------|--------|
| Project scaffolding & config              | ✅     |
| Semantic-first splitter + LangChain check | ✅     |
| Paragraph metadata tracking               | ✅     |
| ChromaDB indexing with deterministic IDs  | ✅     |
| RAG retrieval validation & testing        | ✅     |
| Search Explorer Streamlit app             | ✅     |
| Paragraph reconstruction with highlighting| ✅     |
| Context building & ID Mapping             | ✅     |
| Ollama LLM integration                    | ✅     |
| Answer Generator Streamlit app            | ✅     |
| Footnote-style Citations                  | ✅     |
| CLI interface                             | ⬜     |

**All 43 tests passing** ✅

**Legend:** ✅ Complete | 🚧 In Progress | ⬜ Planned

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
- **Footnote Citations**: Claims are cited with `[1]`, `[2]` markers.
- **Interactive References**: Expandable reference list at the bottom showing the source text for every citation.
- **Model-Agnostic**: Uses whichever model is configured in `rb_expert.toml`.
- **Live Tuning**: Adjust `Top K` and `Distance Threshold` in the sidebar.

---

## Key Implementation Details

### Reference ID Mapping
To prevent LLM hallucinations of long filenames, the system uses an ID mapping layer:
1.  **Retrieval**: Engine gets Top-K chunks.
2.  **Mapping**: `ContextBuilder` assigns integer IDs (`[1]`, `[2]`) to chunks.
3.  **Prompting**: LLM is instructed to cite using *only* the integer ID.
4.  **Resolution**: The UI maps `[1]` back to the full filename and text for display.

### Paragraph Reconstruction
1.  **UI Layer (`database.py`)**: Reconstructs full paragraphs for the Streamlit app, highlighting the specific chunk that triggered the search hit.
2.  **LLM Layer (`context.py`)**: Assembles retrieval results into a structured system prompt with distinct headers for each source.

---

## Notes

This is a private, local-only system: all processing happens on-device with no external API calls.
