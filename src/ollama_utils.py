"""
Ollama utility functions for model discovery and management.
"""

from __future__ import annotations

from typing import Any
import ollama


def list_ollama_models(
    base_url: str = "http://localhost:11434",
    timeout: float = 5.0,
) -> dict[str, Any]:
    """
    List available Ollama models by querying the Ollama API.

    Args:
        base_url: Ollama server URL (default: http://localhost:11434)
        timeout: Request timeout in seconds (default: 5.0)

    Returns:
        Dictionary with structure:
        {
            "available": bool,     # Is Ollama reachable?
            "models": [
                {
                    "name": str,              # e.g., "dolphin-mistral:7b"
                    "size_gb": float,         # Size in gigabytes
                    "family": str,            # e.g., "llama", "unknown"
                    "parameter_size": str,    # e.g., "7B"
                    "quantization": str,      # e.g., "Q4_0"
                    "modified_at": str,       # ISO timestamp
                },
                ...
            ],
            "error": str | None    # Error message if something went wrong
        }

    Examples:
        >>> result = list_ollama_models()
        >>> if result["available"]:
        ...     for model in result["models"]:
        ...         print(f"{model['name']} ({model['size_gb']} GB)")
    """
    try:
        client = ollama.Client(host=base_url, timeout=timeout)
        response = client.list()

        models = []
        for model in response.models:
            # Handle None case for families
            family = model.details.family if model.details.family else "unknown"

            # Convert size from bytes to GB
            size_gb = round(model.size / (1024**3), 2)

            # Format modified_at as ISO string
            modified_at = model.modified_at.isoformat() if model.modified_at else None

            models.append(
                {
                    "name": model.model,
                    "size_gb": size_gb,
                    "family": family,
                    "parameter_size": model.details.parameter_size,
                    "quantization": model.details.quantization_level,
                    "modified_at": modified_at,
                }
            )

        return {
            "available": True,
            "models": models,
            "error": None,
        }

    except ollama.ResponseError as e:
        # Ollama returned an error response
        return {
            "available": False,
            "models": [],
            "error": f"Ollama API error: {e}",
        }

    except Exception as e:
        # Connection error, Ollama not running, etc.
        error_msg = str(e)
        if "Connection" in error_msg or "refused" in error_msg:
            error_msg = "Ollama not reachable. Is it running?"

        return {
            "available": False,
            "models": [],
            "error": error_msg,
        }
