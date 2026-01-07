"""
Model management for whisper.cpp GGML models.

Handles model information, downloading, and path resolution.
Models are downloaded from Hugging Face on first use.
"""

import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)

# Hugging Face model repository
HF_MODEL_REPO = "ggerganov/whisper.cpp"
HF_BASE_URL = f"https://huggingface.co/{HF_MODEL_REPO}/resolve/main"


@dataclass
class ModelInfo:
    """Information about a whisper.cpp model."""

    id: str
    name: str
    filename: str
    size_mb: int
    description: str
    multilingual: bool = True


# Available models with their metadata
AVAILABLE_MODELS: dict[str, ModelInfo] = {
    "tiny": ModelInfo(
        id="tiny",
        name="Tiny",
        filename="ggml-tiny.bin",
        size_mb=75,
        description="Fastest model, lower accuracy. Good for testing.",
        multilingual=True,
    ),
    "tiny.en": ModelInfo(
        id="tiny.en",
        name="Tiny (English)",
        filename="ggml-tiny.en.bin",
        size_mb=75,
        description="English-only tiny model, slightly better for English.",
        multilingual=False,
    ),
    "base": ModelInfo(
        id="base",
        name="Base",
        filename="ggml-base.bin",
        size_mb=142,
        description="Good balance of speed and accuracy.",
        multilingual=True,
    ),
    "base.en": ModelInfo(
        id="base.en",
        name="Base (English)",
        filename="ggml-base.en.bin",
        size_mb=142,
        description="English-only base model.",
        multilingual=False,
    ),
    "small": ModelInfo(
        id="small",
        name="Small",
        filename="ggml-small.bin",
        size_mb=466,
        description="Better accuracy, reasonable speed. Recommended default.",
        multilingual=True,
    ),
    "small.en": ModelInfo(
        id="small.en",
        name="Small (English)",
        filename="ggml-small.en.bin",
        size_mb=466,
        description="English-only small model.",
        multilingual=False,
    ),
    "medium": ModelInfo(
        id="medium",
        name="Medium",
        filename="ggml-medium.bin",
        size_mb=1500,
        description="High accuracy, slower. Good for important transcriptions.",
        multilingual=True,
    ),
    "medium.en": ModelInfo(
        id="medium.en",
        name="Medium (English)",
        filename="ggml-medium.en.bin",
        size_mb=1500,
        description="English-only medium model.",
        multilingual=False,
    ),
    "large-v3": ModelInfo(
        id="large-v3",
        name="Large V3",
        filename="ggml-large-v3.bin",
        size_mb=3000,
        description="Best accuracy, slowest. For critical transcriptions.",
        multilingual=True,
    ),
    "large-v3-turbo": ModelInfo(
        id="large-v3-turbo",
        name="Large V3 Turbo",
        filename="ggml-large-v3-turbo.bin",
        size_mb=1500,
        description="Fast + accurate. Best balance for production use.",
        multilingual=True,
    ),
}

# Default models shown in UI (subset of all available)
DEFAULT_MODEL_OPTIONS = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "large-v3-turbo",
]

DEFAULT_MODEL = "small"


def get_models_dir() -> Path:
    """Get the directory where models are stored."""
    # Use XDG_DATA_HOME or default to ~/.local/share
    if os.name == "nt":  # Windows
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:  # Unix-like
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))

    models_dir = base / "whisper-cpp" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir


def get_model_path(model_id: str) -> Path | None:
    """
    Get the path to a downloaded model.

    Args:
        model_id: Model identifier (e.g., "small", "base.en")

    Returns:
        Path to model file if it exists, None otherwise
    """
    if model_id not in AVAILABLE_MODELS:
        logger.warning(f"Unknown model: {model_id}")
        return None

    model_info = AVAILABLE_MODELS[model_id]
    model_path = get_models_dir() / model_info.filename

    if model_path.exists():
        return model_path
    return None


def is_model_downloaded(model_id: str) -> bool:
    """Check if a model is already downloaded."""
    return get_model_path(model_id) is not None


async def download_model(
    model_id: str,
    progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Download a model from Hugging Face.

    Args:
        model_id: Model identifier (e.g., "small", "base.en")
        progress_callback: Optional callback(downloaded_bytes, total_bytes)

    Returns:
        Path to the downloaded model file

    Raises:
        ValueError: If model_id is not recognized
        RuntimeError: If download fails
    """
    import httpx

    if model_id not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_id}. "
            f"Available: {', '.join(AVAILABLE_MODELS.keys())}"
        )

    model_info = AVAILABLE_MODELS[model_id]
    model_path = get_models_dir() / model_info.filename

    # Check if already downloaded
    if model_path.exists():
        logger.info(f"Model {model_id} already downloaded at {model_path}")
        return model_path

    # Download URL
    url = f"{HF_BASE_URL}/{model_info.filename}"
    logger.info(f"Downloading model {model_id} from {url}")

    # Download with progress
    temp_path = model_path.with_suffix(".tmp")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                downloaded = 0

                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(downloaded, total)

        # Verify temp file was written
        if not temp_path.exists():
            raise RuntimeError(f"Download failed: temp file not created at {temp_path}")

        # Use shutil.move() for cross-platform reliability.
        # Path.rename() can silently fail on Windows, especially in PyInstaller bundles on ARM.
        shutil.move(str(temp_path), str(model_path))

        # Verify the final file exists after move
        if not model_path.exists():
            raise RuntimeError(f"Download failed: file not present at {model_path} after move")

        logger.info(f"Model {model_id} downloaded successfully to {model_path}")
        return model_path

    except Exception as e:
        # Clean up temp file on error
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download model {model_id}: {e}") from e


def list_downloaded_models() -> list[str]:
    """List all downloaded models."""
    downloaded = []
    for model_id in AVAILABLE_MODELS:
        if is_model_downloaded(model_id):
            downloaded.append(model_id)
    return downloaded


def get_model_info(model_id: str) -> ModelInfo | None:
    """Get information about a model."""
    return AVAILABLE_MODELS.get(model_id)


def get_recommended_model() -> str:
    """Get the recommended default model."""
    return DEFAULT_MODEL
