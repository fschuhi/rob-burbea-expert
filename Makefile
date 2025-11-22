# --- Variables ---
VENV_DIR = .venv
VENV_ACTIVATE = $(VENV_DIR)/bin/activate
ACTIVATE = . $(VENV_ACTIVATE)
PIP = $(ACTIVATE) && pip
# RUN_WITH_PATH sets PYTHONPATH to find the 'src' directory
RUN_WITH_PATH = $(ACTIVATE) && PYTHONPATH=.
# The sentinel file to check if setup is complete
SETUP_STAMP = $(VENV_DIR)/.setup_stamp

# Ports for the applications
APP_PORT = 8501
CHAT_PORT = 8502

# --- Phony targets ---
.PHONY: all setup test test-verbose test-fast index app chat clean showtree gentree filesdump startollama killollama startapp killapp startchat killchat ingest-pilot

# Default target runs 'setup'
all: setup

# --- Virtual Environment Setup ---
$(VENV_ACTIVATE):
	python3 -m venv $(VENV_DIR)

# Smart 'setup' target
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

# --- Testing Targets ---
test: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -q

test-verbose: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -v -s

test-fast: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -q --ignore=tests/test_retrieval.py

# --- Application Targets (Foreground) ---

# Run Search Explorer (Foreground)
app: $(SETUP_STAMP)
	$(ACTIVATE) && PYTHONPATH=. streamlit run apps/search_explorer.py --server.port $(APP_PORT)

# Run Answer Generator (Foreground)
chat: $(SETUP_STAMP)
	$(ACTIVATE) && PYTHONPATH=. streamlit run apps/answer_generator.py --server.port $(CHAT_PORT)

# --- Application Targets (Background) ---

# Start Search Explorer in Background
startapp: $(SETUP_STAMP)
	@if lsof -i :$(APP_PORT) > /dev/null; then \
		echo "✅ Search Explorer is already running on port $(APP_PORT)."; \
	else \
		echo "🚀 Starting Search Explorer (port $(APP_PORT))..."; \
		echo "📝 Logs: tmp/app.log"; \
		mkdir -p tmp; \
		nohup bash -c "$(ACTIVATE) && PYTHONPATH=. streamlit run apps/search_explorer.py --server.port $(APP_PORT) --server.headless true" > tmp/app.log 2>&1 & \
		echo "⏳ Waiting for startup..."; \
		sleep 3; \
		if lsof -i :$(APP_PORT) > /dev/null; then \
			echo "✅ App running at http://localhost:$(APP_PORT)"; \
		else \
			echo "❌ Failed to start. Check tmp/app.log"; \
		fi \
	fi

# Stop Search Explorer
killapp:
	@if lsof -i :$(APP_PORT) > /dev/null; then \
		echo "🛑 Stopping Search Explorer..."; \
		lsof -ti :$(APP_PORT) | xargs kill; \
		echo "✅ Search Explorer stopped."; \
	else \
		echo "Search Explorer is not running."; \
	fi

# Start Answer Generator in Background
startchat: $(SETUP_STAMP)
	@if lsof -i :$(CHAT_PORT) > /dev/null; then \
		echo "✅ Answer Generator is already running on port $(CHAT_PORT)."; \
	else \
		echo "🚀 Starting Answer Generator (port $(CHAT_PORT))..."; \
		echo "📝 Logs: tmp/chat.log"; \
		mkdir -p tmp; \
		nohup bash -c "$(ACTIVATE) && PYTHONPATH=. streamlit run apps/answer_generator.py --server.port $(CHAT_PORT) --server.headless true" > tmp/chat.log 2>&1 & \
		echo "⏳ Waiting for startup..."; \
		sleep 3; \
		if lsof -i :$(CHAT_PORT) > /dev/null; then \
			echo "✅ Chat running at http://localhost:$(CHAT_PORT)"; \
		else \
			echo "❌ Failed to start. Check tmp/chat.log"; \
		fi \
	fi

# Stop Answer Generator
killchat:
	@if lsof -i :$(CHAT_PORT) > /dev/null; then \
		echo "🛑 Stopping Answer Generator..."; \
		lsof -ti :$(CHAT_PORT) | xargs kill; \
		echo "✅ Answer Generator stopped."; \
	else \
		echo "Answer Generator is not running."; \
	fi

# --- Data Management Targets ---

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

# Ingest pilot data safely (does not overwrite existing metadata)
ingest-pilot: $(SETUP_STAMP)
	@mkdir -p data/raw_talks
	@if [ -f data/raw_talks/metadata.json ]; then \
		echo "⚠️  Metadata file already exists. Skipping copy to prevent overwriting."; \
	else \
		echo "--- Copying pilot metadata ---"; \
		cp tests/fixtures/data/metadata.json data/raw_talks/; \
	fi
	@echo "--- Copying pilot talks (skipping existing) ---"
	@# cp -n does not overwrite existing files
	@cp -n tests/fixtures/data/raw_talks/*.md data/raw_talks/ || true
	@echo "--- Indexing production DB (data/chroma_db) ---"
	$(RUN_WITH_PATH) python -m src.indexing

# --- Ollama Management Targets ---

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

# --- Utility Targets ---

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
