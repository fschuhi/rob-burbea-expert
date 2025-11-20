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
.PHONY: all setup test run extract streamline clean showtree gentree filesdump discover-pdfs hash

# Default target runs 'setup'
all: setup

# --- FIX: Target is now the 'activate' file itself ---
# This recipe will only run if the 'activate' file does not exist.
$(VENV_ACTIVATE):
	python3 -m venv $(VENV_DIR)

# --- FIX: Smart 'setup' target ---
# This target now depends on the venv *existing* (via the activate file)
# and our config files. It will only run if the stamp file is missing,
# or if requirements.txt or pyproject.toml have been modified.
$(SETUP_STAMP): $(VENV_ACTIVATE) requirements.txt pyproject.toml
	@echo "--- Installing dependencies ---"
	$(PIP) install -r requirements.txt
	@echo "--- Installing project in editable mode ---"
	$(PIP) install -e .
	@echo "--- Setup complete ---"
	@touch $(SETUP_STAMP)

# 'setup' is a friendly alias for the stamp file
setup: $(SETUP_STAMP)

# --- Lightweight 'run' target ---
# Depends on setup being complete.
# Runs the main sync module directly via python -m.
# Pass arguments like: make run ARGS="-c myconfig.toml"
run: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python -m pdf_annot.sync $(ARGS)

# --- NEW: 'hash' target ---
# Runs the print_hashes.py tool
# Pass arguments like: make hash ARGS="(Albini 2013) Title.pdf"
hash: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python tools/print_hashes.py $(ARGS)

# Run tests
test: $(SETUP_STAMP)
	$(RUN_WITH_PATH) pytest -q

# Extract annotations
extract: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python -m pdf_annot.extract -p tests/fixtures/pdf_to_markdown_e2e/input.pdf

# Streamline annotations
streamline: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python -m pdf_annot.streamline_annotations -i tests/fixtures/pdf_to_markdown_e2e/expected_raw.ndjson -o /tmp/final_streamlined.ndjson

# Discover PDFs
discover-pdfs: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python tools/discover_pdfs.py --env tests/fixtures/env/test_pdf_annot.toml --relative-to .

# Concatenate files
filesdump: $(SETUP_STAMP)
	$(RUN_WITH_PATH) python tools/concat_files.py files.lst > tmp/filesdump.txt

# --- Utility targets ---

# Clean build/test artifacts and venv
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

