from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

try:
    import tomllib  # Python 3.11+
except Exception as e:  # pragma: no cover
    raise RuntimeError("tomllib is required (Python 3.11+). You're on an unsupported interpreter.") from e

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


def _expand_path(value: Any) -> Optional[Path]:
    """Helper to resolve path strings, expanding environment variables and user home directory."""
    if value is None or value == "":
        return None
    return Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()


class Paths(BaseModel):
    """
    Filesystem locations used by the Rob Burbea Expert RAG tool.
    """

    data_dir: Path = Field(...,
                           description="Root directory for all project data (e.g., 'data' or 'tests/fixtures/data').")
    raw_talks_dir: Path = Field(..., description="Directory containing the raw Markdown talk transcripts.")
    chroma_db_dir: Path = Field(..., description="Directory where the ChromaDB vector store will be persisted.")
    metadata_path: Path = Field(..., description="Path to the JSON/CSV/Excel file containing talk metadata.")

    # Removed: notes_root, pdf_dirs, backup_dir, temp_dir from old Env

    @field_validator("*", mode="before")
    @classmethod
    def _norm_path_fields(cls, v: Any) -> Any:
        return _expand_path(v)


class Models(BaseModel):
    """
    Configuration for the Embedding and LLM models.
    """
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name of the sentence-transformers model for embeddings.",
    )
    default_llm_model: str = Field(
        default="dolphin-mistral:7b",
        description="The default Ollama model to use for RAG queries.",
    )
    available_llm_models: list[str] = Field(
        default_factory=list,
        description="List of all available Ollama models (used by Streamlit frontend).",
    )


class RAG(BaseModel):
    """
    Settings related to the Retrieval Augmented Generation (RAG) process.
    """
    chunk_size: int = Field(default=500, description="Size of text chunks (in characters) for splitting documents.")
    chunk_overlap: int = Field(default=50, description="Overlap between consecutive chunks.")
    top_k_results: int = Field(default=5, description="Number of context documents to retrieve from ChromaDB.")
    similarity_threshold: float = Field(
        default=0.75, description="Minimum similarity score for document retrieval (0.0 to 1.0)."
    )
    use_langchain_splitter: bool = Field(
        default=False,
        description="Use langchain RecursiveCharacterTextSplitter (slower, battle-tested) vs manual splitter (fast)."
    )


class Ollama(BaseModel):
    """
    Configuration for the local Ollama LLM server.
    """
    base_url: str = Field(default="http://localhost:11434", description="The base URL for the Ollama server.")
    timeout: int = Field(default=60, description="Request timeout in seconds.")


class IO(BaseModel):
    """
    I/O behavior flags.
    """
    atomic_writes: bool = Field(default=True, description="Write files atomically where possible.")
    create_missing_dirs: bool = Field(
        default=True, description="Create configured directories if they do not exist (e.g., chroma_db_dir)."
    )


class CLI(BaseModel):
    """
    CLI-related defaults.
    """
    default_env: Optional[str] = Field(
        default=None,
        description="Optional profile name; useful if you add multiple env profiles later.",
    )


class Env(BaseModel):
    """
    Top-level configuration object for the Rob Burbea Expert RAG system.
    """
    paths: Paths
    models: Models = Field(default_factory=Models)
    rag: RAG = Field(default_factory=RAG)
    ollama: Ollama = Field(default_factory=Ollama)
    io: IO = Field(default_factory=IO)
    cli: CLI = Field(default_factory=CLI)
    # Removed: frontmatter, annotations

    model_config = {"frozen": True}  # make it effectively immutable after creation

    @model_validator(mode="after")
    def _validate_directories(self) -> "Env":
        # 1. Ensure chroma_db_dir exists (create if allowed)
        if not self.paths.chroma_db_dir.exists():
            if self.io.create_missing_dirs:
                self.paths.chroma_db_dir.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"chroma_db_dir does not exist: {self.paths.chroma_db_dir}"
                )

        # 2. Ensure raw_talks_dir exists
        if not self.paths.raw_talks_dir.exists() or not self.paths.raw_talks_dir.is_dir():
            raise ValueError(
                f"raw_talks_dir not found or not a directory: {self.paths.raw_talks_dir}"
            )

        # 3. Ensure metadata_path exists
        if not self.paths.metadata_path.exists() or not self.paths.metadata_path.is_file():
            raise ValueError(
                f"metadata_path not found or not a file: {self.paths.metadata_path}"
            )

        return self


def load_env(
        source: Optional[Path | str | Mapping[str, Any]] = None,
        profile: Optional[str] = None,
        # --- CHANGE: Updated env_var and default_filenames for this project ---
        env_var: str = "RB_EXPERT_ENV_PATH",
        default_filenames: tuple[str, ...] = ("rb_expert.toml", "rob-burbea-expert.toml"),
) -> Env:
    """
    Load an Env from a TOML file, a mapping, or defaults.

    Resolution order:
      1) Mapping passed directly.
      2) Explicit path passed in.
      3) Path from RB_EXPERT_ENV_PATH env var.
      4) First existing file among default_filenames in CWD.
    """
    if isinstance(source, Mapping):
        data = _mapping_to_data(source)
        return _build_env_from_data(data, profile=profile)

    path = None

    if isinstance(source, (str, Path)):
        path = Path(str(source))
    elif source is None:
        env_path = os.environ.get(env_var)
        if env_path:
            path = Path(env_path)
        else:
            cwd = Path.cwd()
            for name in default_filenames:
                candidate = cwd / name
                if candidate.exists():
                    path = candidate
                    break

    if path is None:
        # --- CHANGE: Updated message ---
        raise FileNotFoundError(
            "No configuration source found. Provide a mapping, set RB_EXPERT_ENV_PATH, "
            f"or create one of {default_filenames} in the current directory."
        )

    with path.open("rb") as f:
        toml_data = tomllib.load(f)

    return _build_env_from_data(toml_data, profile=profile)


def _mapping_to_data(mapping: Mapping[str, Any]) -> dict[str, Any]:
    return dict(mapping)


def _build_env_from_data(data: Mapping[str, Any], profile: Optional[str]) -> Env:
    if profile:
        envs = data.get("envs")
        if not isinstance(envs, Mapping) or profile not in envs:
            raise KeyError(f"Profile '{profile}' not found under [envs] in configuration.")
        data = envs[profile]

    grouped = {
        "paths": {},
        "models": {},
        "rag": {},
        "ollama": {},
        "io": {},
        "cli": {},
        # Removed: frontmatter, annotations
    }

    # Copy grouped keys if present
    for section in grouped.keys():
        if isinstance(data.get(section), Mapping):
            grouped[section] = dict(data[section])

    # Allow flat keys too (simplified for the new project, focusing on core RAG needs)
    flat_to_group = {
        "data_dir": ("paths", "data_dir"),
        "raw_talks_dir": ("paths", "raw_talks_dir"),
        "chroma_db_dir": ("paths", "chroma_db_dir"),
        "metadata_path": ("paths", "metadata_path"),

        "embedding_model": ("models", "embedding_model"),
        "default_llm_model": ("models", "default_llm_model"),

        "chunk_size": ("rag", "chunk_size"),
        "chunk_overlap": ("rag", "chunk_overlap"),
        "top_k_results": ("rag", "top_k_results"),
        "similarity_threshold": ("rag", "similarity_threshold"),
        "use_langchain_splitter": ("rag", "use_langchain_splitter"),

        "base_url": ("ollama", "base_url"),
        "timeout": ("ollama", "timeout"),

        "atomic_writes": ("io", "atomic_writes"),
        "create_missing_dirs": ("io", "create_missing_dirs"),
        "default_env": ("cli", "default_env"),
    }

    for k, v in data.items():
        if k in flat_to_group and not grouped[flat_to_group[k][0]].get(flat_to_group[k][1]):
            group, dest = flat_to_group[k]
            grouped[group][dest] = v

    try:
        return Env(
            paths=Paths(**grouped["paths"]),
            models=Models(**grouped["models"]),
            rag=RAG(**grouped["rag"]),
            ollama=Ollama(**grouped["ollama"]),
            io=IO(**grouped["io"]),
            cli=CLI(**grouped["cli"]),
        )
    except ValidationError as ve:
        raise ValueError(f"Invalid configuration: {ve}") from ve
