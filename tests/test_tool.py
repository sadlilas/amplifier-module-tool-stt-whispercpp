"""Tests for the TranscribeTool."""

import pytest

from amplifier_module_tool_stt_whispercpp import (
    AVAILABLE_MODELS,
    TranscribeTool,
    get_model_path,
)


def test_available_models():
    """Test that AVAILABLE_MODELS is populated."""
    assert len(AVAILABLE_MODELS) > 0
    assert "small" in AVAILABLE_MODELS
    assert "tiny" in AVAILABLE_MODELS


def test_model_info():
    """Test model info structure."""
    model = AVAILABLE_MODELS["small"]
    assert hasattr(model, "name")
    assert hasattr(model, "size_mb")
    assert model.size_mb > 0


def test_get_model_path_not_downloaded():
    """Test get_model_path returns None for non-downloaded model."""
    path = get_model_path("large-v3")
    assert path is None or path.exists()


def test_tool_creation():
    """Test TranscribeTool can be instantiated."""
    tool = TranscribeTool(config={"model": "small"})
    assert tool.name == "transcribe"
    assert tool.description is not None


def test_tool_get_schema():
    """Test the tool has proper input schema."""
    tool = TranscribeTool(config={"model": "small"})
    schema = tool.get_schema()
    assert "audio_data" in schema["properties"]
    assert "audio_format" in schema["properties"]
