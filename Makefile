# --- Variables ---
VENV_DIR = .venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate
ACTIVATE = . $(VENV_ACTIVATE)
PIP = $(ACTIVATE) && pip
# RUN_WITH_PATH sets PYTHONPATH to find the 'src' directory
RUN_WITH_PATH = $(ACTIVATE) && PYTHONPATH=.
# The sentinel file to check if setup is complete
SETUP_STAMP = $(VENV_DIR)/.setup_stamp

# --- Phony targets ---
.PHONY: all setup test test-verbose test-fast index app chat clean showtree gentree filesdump startollama killollama ingest-pilot

all: setup

# --- Setup ---
$(VENV_ACTIVATE):
	python3 -m venv $(VENV_DIR)

$(SETUP_STAMP): $(VENV_ACTIVATE) requirements.txt pyproject.toml
	@echo "--- Installing dependencies ---"
	$(PIP) install -r requirements.txt
	@echo "--- Installing project in editable mode ---"
	$(PIP) install -e .
	@# Auto-create config if missing
	@if [ ! -f rb_expert.toml ]; then \
		echo "--- Creating default configuration (rb_expert.toml) ---"; \
		cp rb_expert.example.toml rb_expert.toml; \
	fi
	@echo "--- Setup complete ---"
	@touch $(SETUP_STAMP)

setup: $(SETUP_STAMP)

# --- Testing ---
test: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -q

test-verbose: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -v -s

test-fast: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -q --ignore=tests/test_retrieval.py

# --- Application ---
app: $(SETUP_STAMP)
	$(ACTIVATE) && PYTHONPATH=. streamlit run apps/search_explorer.py

chat: $(SETUP_STAMP)
	$(ACTIVATE) && PYTHONPATH=. streamlit run apps/answer_generator.py

# --- Data Management ---

# Manually index the TEST database (tmp/)
index: $(SETUP_STAMP)
	@echo "--- Indexing 32 talks to tmp/chroma_db_retrieval ---"
	@rm -rf tmp/chroma_db_retrieval
	$(RUN_WITH_PATH) python -c "from pathlib import Path; \
		from src.env import Env, Paths, RAG, IO, Models; \
		from src.indexing import run_indexer; \
		env = Env( \
			paths=Paths( \
				data_dir=Path('tests/fixtures/data'), \
				raw_talks_dir=Path('tests/fixtures/data/raw_talks'), \
				chroma_db_dir=Path('tmp/chroma_db_retrieval'), \
				metadata_path=Path('tests/fixtures/data/metadata.json') \
			), \
			rag=RAG(chunk_size=500, chunk_overlap=0), \
			io=IO(create_missing_dirs=True), \
			models=Models(embedding_model='all-MiniLM-L6-v2') \
		); \
		run_indexer(env)"

# NEW: Ingest the pilot data into the PRODUCTION database (data/)
ingest-pilot: $(SETUP_STAMP)
	@echo "--- Copying pilot data to data/raw_talks ---"
	@mkdir -p data/raw_talks
	@cp tests/fixtures/data/raw_talks/*.md data/raw_talks/
	@cp tests/fixtures/data/metadata.json data/raw_talks/
	@echo "--- Indexing pilot data to production DB (data/chroma_db) ---"
	@# This uses rb_expert.toml configuration automatically
	$(RUN_WITH_PATH) python src/indexing.py

# --- Ollama Management ---
startollama:
	@if lsof -i :11434 > /dev/null; then \
		echo "✅ Ollama is already running."; \
	else \
		echo "🚀 Starting Ollama in background (logs in tmp/ollama.log)..."; \
		mkdir -p tmp; \
		ollama serve > tmp/ollama.log 2>&1 & \
		echo "Waiting for startup..."; \
		sleep 3; \
		if lsof -i :11434 > /dev/null; then echo "✅ Ollama started successfully."; else echo "❌ Failed to start."; fi \
	fi

killollama:
	@if lsof -i :11434 > /dev/null; then \
		echo "🛑 Stopping Ollama..."; \
		lsof -ti :11434 | xargs kill; \
		echo "✅ Ollama stopped."; \
	else \
		echo "Ollama is not running."; \
	fi

# --- Utilities ---
filesdump: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python tools/concat_files.py files.lst > tmp/filesdump.txt

clean:
	rm -rf $(VENV_DIR) .pytest_cache tmp
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +

showtree:
	tree -I ".venv|__pycache__|.idea|.pytest_cache|*egg-info|tmp"

gentree:
	tree -I ".venv|__pycache__|.idea|.pytest_cache|*egg-info|tmp" > project-tree.txt
