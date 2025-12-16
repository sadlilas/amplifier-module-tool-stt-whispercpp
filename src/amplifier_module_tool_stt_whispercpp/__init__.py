"""
Amplifier Tool Module: Speech-to-Text with whisper.cpp

Local speech-to-text transcription using whisper.cpp.
Models are downloaded on first use from Hugging Face.

Supported models:
- tiny: ~75MB, fastest
- base: ~142MB, good balance
- small: ~466MB, recommended default
- medium: ~1.5GB, high accuracy
- large-v3: ~3GB, best accuracy
- large-v3-turbo: ~1.5GB, fast and accurate

Usage:
    tools:
      - module: tool-stt-whispercpp
        config:
          model: "small"
"""

from typing import Any

__amplifier_module_type__ = "tool"
__version__ = "0.1.0"

from .tool import TranscribeTool
from .models import AVAILABLE_MODELS, download_model, get_model_path

__all__ = [
    "TranscribeTool",
    "AVAILABLE_MODELS",
    "download_model",
    "get_model_path",
    "mount",
]


async def mount(coordinator: Any, config: dict[str, Any] | None = None) -> Any:
    """
    Mount the TranscribeTool as an Amplifier module.

    Args:
        coordinator: ModuleCoordinator for registration
        config: Configuration from mount plan
            - model: Model to use (tiny, base, small, medium, large-v3, large-v3-turbo)

    Returns:
        The tool instance
    """
    config = config or {}
    tool_config = config.get("config", config)

    tool = TranscribeTool(config=tool_config)

    await coordinator.mount("tools", tool, name="transcribe")

    coordinator.register_contributor(
        "observability.events",
        "amplifier-module-tool-stt-whispercpp",
        lambda: [
            "tool:transcribe:start",
            "tool:transcribe:complete",
            "tool:transcribe:error",
        ],
    )

    return tool
