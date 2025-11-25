"""
Tests for ollama_utils module.
"""

from datetime import datetime, timezone
from unittest.mock import Mock, patch
import pytest
import ollama

from src.ollama_utils import list_ollama_models


def create_mock_model(
    name: str,
    size: int,
    family: str | None,
    parameter_size: str,
    quantization: str,
    modified_at: datetime | None = None,
):
    """Helper to create a mock Ollama Model object."""
    mock_model = Mock()
    mock_model.model = name
    mock_model.size = size
    mock_model.modified_at = modified_at or datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

    mock_details = Mock()
    mock_details.family = family
    mock_details.parameter_size = parameter_size
    mock_details.quantization_level = quantization

    mock_model.details = mock_details
    return mock_model


def test_list_ollama_models_success():
    """Test successful model listing."""
    # Arrange
    mock_response = Mock()
    mock_response.models = [
        create_mock_model(
            name="dolphin-mistral:7b",
            size=4108940323,  # ~3.83 GB
            family="llama",
            parameter_size="7B",
            quantization="Q4_0",
        ),
        create_mock_model(
            name="llama2:latest",
            size=3825819449,  # ~3.56 GB
            family="llama",
            parameter_size="7B",
            quantization="Q4_0",
        ),
    ]

    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.list.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = list_ollama_models()

        # Assert
        assert result["available"] is True
        assert result["error"] is None
        assert len(result["models"]) == 2

        # Check first model
        model1 = result["models"][0]
        assert model1["name"] == "dolphin-mistral:7b"
        assert model1["size_gb"] == 3.83
        assert model1["family"] == "llama"
        assert model1["parameter_size"] == "7B"
        assert model1["quantization"] == "Q4_0"
        assert model1["modified_at"] == "2024-01-01T12:00:00+00:00"

        # Verify client was created with correct parameters
        mock_client_class.assert_called_once_with(host="http://localhost:11434", timeout=5.0)


def test_list_ollama_models_with_custom_params():
    """Test model listing with custom base_url and timeout."""
    mock_response = Mock()
    mock_response.models = []

    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.list.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = list_ollama_models(base_url="http://custom:8080", timeout=10.0)

        # Assert
        assert result["available"] is True
        mock_client_class.assert_called_once_with(host="http://custom:8080", timeout=10.0)


def test_list_ollama_models_handles_none_family():
    """Test that None families are handled gracefully."""
    # Arrange
    mock_response = Mock()
    mock_response.models = [
        create_mock_model(
            name="wizard-vicuna-uncensored:13b",
            size=7365835082,
            family=None,  # This is the edge case from your data
            parameter_size="13B",
            quantization="Q4_0",
        ),
    ]

    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.list.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = list_ollama_models()

        # Assert
        assert result["available"] is True
        assert result["models"][0]["family"] == "unknown"


def test_list_ollama_models_empty_list():
    """Test when Ollama has no models installed."""
    # Arrange
    mock_response = Mock()
    mock_response.models = []

    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.list.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Act
        result = list_ollama_models()

        # Assert
        assert result["available"] is True
        assert result["models"] == []
        assert result["error"] is None


def test_list_ollama_models_connection_error():
    """Test when Ollama is not running (connection refused)."""
    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.list.side_effect = ConnectionRefusedError("Connection refused")
        mock_client_class.return_value = mock_client

        # Act
        result = list_ollama_models()

        # Assert
        assert result["available"] is False
        assert result["models"] == []
        assert "Ollama not reachable" in result["error"]


def test_list_ollama_models_api_error():
    """Test when Ollama API returns an error."""
    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.list.side_effect = ollama.ResponseError("API Error")
        mock_client_class.return_value = mock_client

        # Act
        result = list_ollama_models()

        # Assert
        assert result["available"] is False
        assert result["models"] == []
        assert "Ollama API error" in result["error"]


def test_list_ollama_models_generic_error():
    """Test handling of unexpected errors."""
    with patch("ollama.Client") as mock_client_class:
        mock_client = Mock()
        mock_client.list.side_effect = ValueError("Unexpected error")
        mock_client_class.return_value = mock_client

        # Act
        result = list_ollama_models()

        # Assert
        assert result["available"] is False
        assert result["models"] == []
        assert "Unexpected error" in result["error"]
