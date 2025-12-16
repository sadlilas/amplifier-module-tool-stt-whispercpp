"""
TranscribeTool - Amplifier Tool for local speech-to-text using whisper.cpp.
"""

import asyncio
import base64
import io
import logging
import tempfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import av
from amplifier_core import ToolResult

from .models import AVAILABLE_MODELS, DEFAULT_MODEL, download_model, get_model_path

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio."""
    text: str
    start: float
    end: float


@dataclass
class TranscriptionResult:
    """Result of transcription."""
    text: str
    language: str = "en"
    duration: float = 0.0
    segments: list[TranscriptionSegment] = field(default_factory=list)


def convert_audio_to_wav(audio_data: bytes, input_format: str) -> bytes:
    """
    Convert audio to 16kHz mono WAV format required by whisper.cpp.

    Uses PyAV which bundles FFmpeg - no external dependencies.
    """
    input_buffer = io.BytesIO(audio_data)
    output_buffer = io.BytesIO()

    with av.open(input_buffer, format=input_format) as input_container:
        input_stream = input_container.streams.audio[0]

        with av.open(output_buffer, mode="w", format="wav") as output_container:
            output_stream = output_container.add_stream("pcm_s16le", rate=WHISPER_SAMPLE_RATE)
            output_stream.layout = "mono"

            resampler = av.AudioResampler(
                format="s16",
                layout="mono",
                rate=WHISPER_SAMPLE_RATE,
            )

            for frame in input_container.decode(input_stream):
                for resampled_frame in resampler.resample(frame):
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)

            for packet in output_stream.encode(None):
                output_container.mux(packet)

    return output_buffer.getvalue()


class TranscribeTool:
    """
    Amplifier Tool for local audio transcription using whisper.cpp.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the TranscribeTool.

        Args:
            config: Tool configuration
                - model: Model to use (tiny, base, small, medium, large-v3, large-v3-turbo)
        """
        self._config = config or {}
        self._model_id = self._config.get("model", DEFAULT_MODEL)
        self._model = None

    @property
    def name(self) -> str:
        return "transcribe"

    @property
    def description(self) -> str:
        return "Transcribe audio to text using local whisper.cpp."

    def get_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_data": {
                    "type": "string",
                    "description": "Base64-encoded audio data",
                },
                "audio_format": {
                    "type": "string",
                    "description": "Audio format (wav, mp3, webm, etc.)",
                    "default": "wav",
                },
                "language": {
                    "type": "string",
                    "description": "Language code (e.g., 'en') or null for auto-detect",
                },
            },
            "required": ["audio_data"],
        }

    async def execute(self, input: dict[str, Any]) -> ToolResult:
        """Transcribe audio to text."""
        audio_data_b64 = input.get("audio_data")
        if not audio_data_b64:
            return ToolResult(
                success=False,
                output=None,
                error={"message": "Missing required field: audio_data"},
            )

        try:
            audio_data = base64.b64decode(audio_data_b64)
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error={"message": f"Invalid base64 audio data: {e}"},
            )

        audio_format = input.get("audio_format", "wav")
        language = input.get("language")

        try:
            result = await self._transcribe(audio_data, audio_format, language)
            return ToolResult(success=True, output=asdict(result))
        except Exception as e:
            logger.exception("Transcription failed")
            return ToolResult(
                success=False,
                output=None,
                error={"message": str(e)},
            )

    async def _transcribe(
        self,
        audio_data: bytes,
        audio_format: str,
        language: str | None,
    ) -> TranscriptionResult:
        """Transcribe audio using whisper.cpp."""
        from pywhispercpp.model import Model

        # Ensure model is downloaded
        model_path = get_model_path(self._model_id)
        if model_path is None:
            model_path = await download_model(self._model_id)

        # Load model if needed
        if self._model is None:
            import os
            n_threads = min(os.cpu_count() or 4, 8)
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None, lambda: Model(str(model_path), n_threads=n_threads)
            )

        # Convert audio to 16kHz mono WAV
        loop = asyncio.get_event_loop()
        wav_data = await loop.run_in_executor(
            None, lambda: convert_audio_to_wav(audio_data, audio_format)
        )

        # Write to temp file (whisper.cpp needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_data)
            wav_path = Path(f.name)

        try:
            # Transcribe
            segments_raw = await loop.run_in_executor(
                None, lambda: self._model.transcribe(str(wav_path), language=language)
            )

            # Convert to result format
            segments = []
            full_text_parts = []

            for seg in segments_raw:
                start = seg.t0 / 100.0
                end = seg.t1 / 100.0
                text = seg.text.strip()

                if text:
                    segments.append(TranscriptionSegment(text=text, start=start, end=end))
                    full_text_parts.append(text)

            duration = segments[-1].end if segments else 0.0

            return TranscriptionResult(
                text=" ".join(full_text_parts),
                language=language or "auto",
                duration=duration,
                segments=segments,
            )
        finally:
            wav_path.unlink(missing_ok=True)

    def get_supported_formats(self) -> list[str]:
        return ["wav", "mp3", "m4a", "ogg", "webm", "flac", "aac", "aiff"]

    def get_available_models(self) -> list[dict]:
        return [
            {"id": k, "name": v.name, "size_mb": v.size_mb}
            for k, v in AVAILABLE_MODELS.items()
        ]
