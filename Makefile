# --- Variables ---
VENV_DIR = .venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate
ACTIVATE = . $(VENV_ACTIVATE)
PIP = $(ACTIVATE) && pip
# RUN_WITH_PATH sets PYTHONPATH to find the 'src' directory
RUN_WITH_PATH = $(ACTIVATE) && PYTHONPATH=src
# The sentinel file to check if setup is complete
SETUP_STAMP = $(VENV_DIR)/.setup_stamp

# --- Phony targets (commands that don't produce files) ---
.PHONY: all setup test test-verbose test-fast index app clean showtree gentree filesdump

# Default target runs 'setup'
all: setup

# --- Virtual Environment Setup ---
# This recipe will only run if the 'activate' file does not exist.
$(VENV_ACTIVATE):
	python3 -m venv $(VENV_DIR)

# Smart 'setup' target - only runs if dependencies changed
$(SETUP_STAMP): $(VENV_ACTIVATE) requirements.txt pyproject.toml
	@echo "--- Installing dependencies ---"
	$(PIP) install -r requirements.txt
	@echo "--- Installing project in editable mode ---"
	$(PIP) install -e .
	@echo "--- Setup complete ---"
	@touch $(SETUP_STAMP)

# 'setup' is a friendly alias for the stamp file
setup: $(SETUP_STAMP)

# --- Testing Targets ---

# Run all tests (quiet mode)
test: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -q

# Run tests with verbose output and print statements
test-verbose: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -v -s

# Run fast tests only (skip retrieval tests that do full indexing)
test-fast: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -q --ignore=tests/test_retrieval.py

# --- Indexing Targets ---

# Index the 32 test talks into the retrieval database
# This is useful for manually refreshing the database used by the app
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

# --- Application Targets ---

# Run the search explorer Streamlit app
app: $(SETUP_STAMP)
	$(ACTIVATE) && PYTHONPATH=. streamlit run apps/search_explorer.py

# --- Utility Targets ---

# Concatenate files for LLM context
filesdump: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python tools/concat_files.py files.lst > tmp/filesdump.txt

# Clean build/test artifacts, venv, and databases
clean:
	rm -rf $(VENV_DIR) .pytest_cache tmp
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	find . -name "*.egg-info" -type d -prune -exec rm -rf {} +

# Show project tree (excluding common noise)
showtree:
	tree -I ".venv|__pycache__|.idea|.pytest_cache|*egg-info|tmp"

# Save a tree snapshot
gentree:
	tree -I ".venv|__pycache__|.idea|.pytest_cache|*egg-info|tmp" > project-tree.txt
