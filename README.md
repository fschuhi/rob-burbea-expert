# Rob Burbea Expert

A local RAG (Retrieval Augmented Generation) system for exploring Rob Burbea's dharma talks using uncensored local LLMs via Ollama.

## Project Vision

This system enables deep engagement with Rob Burbea's teachings through:
- **Semantic search** across 300+ dharma talks
- **Question answering** powered by local LLMs (no cloud dependencies)
- **Comparative analysis** between retreats and teaching periods
- **Citation-backed responses** linking to specific talks

Initial focus: 30 talks from "Practicing the Jhanas" retreat (2019)

## Architecture

```
Query → Embedding → ChromaDB Retrieval → Context + LLM → Response
```

**Components:**
- **ChromaDB**: Vector database for semantic search
- **sentence-transformers**: Local embedding generation
- **Ollama**: Local LLM inference (tested models: dolphin-mistral, gemma3n-abliterated)
- **Streamlit**: Web UI (future)

## Text Splitting Strategy

**Semantic-first, two-phase approach:**
1. **Phase 1**: Split on paragraph boundaries (`\n\n`) to preserve semantic units
2. **Phase 2**: For oversized paragraphs, sub-split with overlap respecting word/line boundaries

This ensures embeddings capture complete thoughts rather than arbitrary text fragments.

**Implementation:**
- **Manual splitter** (default): Fast, zero ML dependencies for testing
- **Langchain fallback**: Battle-tested `RecursiveCharacterTextSplitter` for validation
- Configurable via `use_langchain_splitter` in `rag` section

## Setup

### Prerequisites
- Python 3.11+ (uses `tomllib`)
- [Ollama](https://ollama.ai) installed with models downloaded
- 24GB RAM recommended (for larger models)

### Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Configuration

Create a `rb_expert.toml` configuration file (see `tests/fixtures/test_env.toml` for example):

```toml
[paths]
data_dir = "data"
raw_talks_dir = "data/raw_talks"
chroma_db_dir = "data/chroma_db"
metadata_path = "data/raw_talks/metadata.json"

[rag]
chunk_size = 500
chunk_overlap = 50
use_langchain_splitter = false  # Use fast manual splitter
```

### Project Structure

```
rob-burbea-expert/
├── src/              # Core application code
│   ├── env.py        # Configuration management
│   └── data_prep.py  # Text loading and splitting
├── tests/            # Pytest tests with fixtures
│   ├── test_env.py
│   └── test_data_prep.py
├── tools/            # Utility scripts
├── data/             # Runtime data (gitignored)
│   ├── chroma_db/    # Vector database
│   └── raw_talks/    # Full talk collection
└── notebooks/        # Exploration notebooks
```

## Development Approach

**Incremental, test-driven development:**
1. Build core functionality with tests using small dataset (3 talks)
2. Validate with full Jhana retreat (30 talks)
3. Scale to complete corpus (548 talks)
4. Add web UI

**Philosophy:** Each component should be testable and documentable independently. Code should be comprehensible to future conversations.

## Current Status

- [x] Project structure created
- [x] Configuration system (`env.py` with TOML support)
- [x] Text loading and semantic-first splitting (`data_prep.py`)
- [x] Comprehensive test suite (13 tests passing)
- [ ] Excel metadata extraction tool
- [ ] Embedding generation
- [ ] ChromaDB indexing
- [ ] RAG retrieval logic
- [ ] Ollama integration
- [ ] Web UI

## Testing

```bash
# Run all tests
pytest tests/

# Quick run with minimal output
pytest -q

# Verbose with durations
pytest -v --durations=10
```

**Performance:**
- Most tests use fast manual splitter (~0.1s)
- Langchain validation tests include ML library import (~3s)
- Total test suite: ~3s

## Related Work

- Hermes Amara Foundation: https://hermesamara.org
- Talk transcriptions: 548 high-quality markdown files
- Metadata: Excel spreadsheet with retreat info, dates, series

## Notes

This is a private, local-only system. All processing happens on-device with no external API calls.
