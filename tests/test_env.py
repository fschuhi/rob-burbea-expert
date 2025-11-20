from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Mapping

import pytest

# NOTE: We assume the user has moved src/old/env.py to src/env.py
from src.env import Env, load_env


# =============================================================================
# Test Configuration Fixture
# =============================================================================

# This fixture represents the contents of the test_env.toml file
# provided earlier, but as a Python mapping for easy loading.
def get_test_config_mapping(tmp_path: Path) -> Mapping[str, Any]:
    """Provides a complete configuration mapping using temporary paths."""
    # Ensure the required directories/files exist for validation checks
    # The setup of these paths is crucial for the Env validator to pass.
    fixtures_dir = tmp_path / "fixtures"
    raw_talks_dir = fixtures_dir / "talks"
    metadata_path = fixtures_dir / "metadata.json"

    raw_talks_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.touch(exist_ok=True)

    return {
        "paths": {
            # data_dir is now explicitly added to the mapping to satisfy the requirement
            "data_dir": str(fixtures_dir / "data"),
            # chroma_db_dir will be created by the Env validator if missing
            "chroma_db_dir": str(fixtures_dir / "data" / "chroma_db"),
            # raw_talks_dir MUST exist before loading Env
            "raw_talks_dir": str(raw_talks_dir),
            # metadata_path MUST exist before loading Env
            "metadata_path": str(metadata_path),
        },
        "models": {
            "embedding_model": "all-MiniLM-L6-v2",
            "default_llm_model": "dolphin-mistral:7b",
            "available_llm_models": [
                "dolphin-mistral:7b",
                "gemma3n-abliterated:e2b-fp16",
            ],
        },
        "rag": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "top_k_results": 3,
            "similarity_threshold": 0.7,
        },
        "ollama": {
            "base_url": "http://localhost:11434",
            "timeout": 60,
        },
        "io": {
            "create_missing_dirs": True,
            "atomic_writes": True,
        },
    }


# =============================================================================
# Unit tests for Env loading and validation
# =============================================================================

def test_load_from_mapping_creates_chroma_db_dir(tmp_path: Path):
    """Test loading from a mapping creates the chroma_db_dir if it's missing."""
    data = get_test_config_mapping(tmp_path)
    # Ensure the chroma_db_dir is missing before load
    chroma_path = Path(data["paths"]["chroma_db_dir"])
    if chroma_path.exists():
        chroma_path.rmdir()

    env = load_env(data)
    assert isinstance(env, Env)
    # Check that it created the directory
    assert env.paths.chroma_db_dir.exists()
    assert env.rag.chunk_size == 500  # Check a RAG default
    assert env.models.embedding_model == "all-MiniLM-L6-v2"  # Check a Model default


def test_load_from_mapping_validates_raw_talks_dir(tmp_path: Path):
    """Test loading fails if raw_talks_dir does not exist."""
    data = get_test_config_mapping(tmp_path)

    # Intentionally point to a non-existent directory
    missing_dir = tmp_path / "does_not_exist_talks"
    data["paths"]["raw_talks_dir"] = str(missing_dir)

    with pytest.raises(ValueError) as exc:
        load_env(data)

    assert "raw_talks_dir not found or not a directory" in str(exc.value)


def test_load_from_mapping_validates_metadata_path(tmp_path: Path):
    """Test loading fails if metadata_path does not exist."""
    data = get_test_config_mapping(tmp_path)

    # Intentionally point to a non-existent file
    missing_file = tmp_path / "missing_metadata.json"
    data["paths"]["metadata_path"] = str(missing_file)

    with pytest.raises(ValueError) as exc:
        load_env(data)

    assert "metadata_path not found or not a file" in str(exc.value)


def test_load_from_file_flat_and_grouped(tmp_path: Path):
    """Test loading from a TOML file using mixed flat/grouped sections."""
    # We create the required directories/files for a successful load.
    raw_talks = tmp_path / "talks"
    raw_talks.mkdir()
    metadata = tmp_path / "meta.json"
    metadata.touch()

    # --- FIX: Added data_dir as a required field ---
    data_dir_path = tmp_path / "data"

    toml = textwrap.dedent(
        f"""
        # Flat keys will be grouped by _build_env_from_data
        data_dir = "{str(data_dir_path)}"
        raw_talks_dir = "{str(raw_talks)}"
        metadata_path = "{str(metadata)}"
        chunk_size = 600

        [paths]
        # Grouped key takes precedence
        chroma_db_dir = "{str(tmp_path / "db")}"

        [io]
        create_missing_dirs = true

        [models]
        # Override default embedding model
        embedding_model = "e5-large-v2"
        """
    )

    cfg = tmp_path / "rb_expert.toml"
    cfg.write_text(toml, encoding="utf-8")

    env = load_env(cfg)
    assert env.paths.chroma_db_dir.exists()  # Should create 'db'
    assert env.paths.raw_talks_dir == raw_talks.resolve()
    assert env.rag.chunk_size == 600  # Flat key loaded
    assert env.models.embedding_model == "e5-large-v2"  # Grouped key override


def test_env_var_resolution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that the custom environment variable RB_EXPERT_ENV_PATH works."""
    # We create the required directories/files for a successful load.
    (tmp_path / "talks").mkdir()
    (tmp_path / "meta.json").touch()

    # --- FIX: Added data_dir as a required field ---
    data_dir_path = tmp_path / "data_env"

    cfg = tmp_path / "env.toml"
    cfg.write_text(
        f"""
        [paths]
        data_dir = "{str(data_dir_path)}"
        chroma_db_dir = "{str(tmp_path / "db_env")}"
        raw_talks_dir = "{str(tmp_path / "talks")}"
        metadata_path = "{str(tmp_path / "meta.json")}"
        [io]
        create_missing_dirs = true
        """,
        encoding="utf-8",
    )

    # NOTE: Set the new environment variable name
    monkeypatch.setenv("RB_EXPERT_ENV_PATH", str(cfg))
    env = load_env()
    assert env.paths.chroma_db_dir.exists()
    assert env.io.create_missing_dirs is True
