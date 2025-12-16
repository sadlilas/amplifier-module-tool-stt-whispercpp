# amplifier-module-tool-stt-whispercpp

Amplifier tool module for local speech-to-text using [whisper.cpp](https://github.com/ggml-org/whisper.cpp).

## Features

- Local transcription using whisper.cpp (no cloud, fully private)
- No PyTorch required (~3MB vs ~2GB)
- Models downloaded on first use from Hugging Face
- Multiple model sizes: tiny, base, small, medium, large-v3, large-v3-turbo
- Self-contained audio conversion via PyAV (bundles FFmpeg internally)
- Accepts common audio formats: wav, mp3, m4a, ogg, webm, flac, aac

## Installation

```bash
pip install amplifier-module-tool-stt-whispercpp
```

Or from GitHub:

```bash
pip install git+https://github.com/sadlilas/amplifier-module-tool-stt-whispercpp
```

## Usage

### In Amplifier Mount Plan

Add to your Amplifier session's mount plan:

```python
tools = [
    {"module": "tool-stt-whispercpp", "config": {"model": "small"}},
]
```

### Programmatic Usage

```python
from amplifier_module_tool_stt_whispercpp import TranscribeTool
import base64

tool = TranscribeTool(config={"model": "small"})

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

result = await tool.execute({
    "audio_data": audio_b64,
    "audio_format": "wav",
})

print(result.output["text"])
```

### Model Management

The module exports functions for managing models:

```python
from amplifier_module_tool_stt_whispercpp import (
    AVAILABLE_MODELS,
    download_model,
    get_model_path,
)

# List available models
for model_id, info in AVAILABLE_MODELS.items():
    print(f"{model_id}: {info.name} ({info.size_mb} MB)")

# Check if a model is downloaded
path = get_model_path("small")
if path is None:
    print("Model not downloaded")

# Download a model
await download_model("small")
```

## Models

| Model | Size | Notes |
|-------|------|-------|
| tiny | 75 MB | Fastest, lower accuracy |
| tiny.en | 75 MB | English-only tiny |
| base | 142 MB | Good balance |
| base.en | 142 MB | English-only base |
| small | 466 MB | Recommended default |
| small.en | 466 MB | English-only small |
| medium | 1.5 GB | High accuracy |
| medium.en | 1.5 GB | English-only medium |
| large-v3 | 3 GB | Best accuracy |
| large-v3-turbo | 1.5 GB | Fast and accurate |

Models are stored in:
- Linux/macOS: `~/.local/share/whisper-cpp/models/`
- Windows: `%LOCALAPPDATA%/whisper-cpp/models/`

## Bundling with PyInstaller

This module works with PyInstaller for creating standalone executables. Add these hidden imports to your `.spec` file:

```python
hiddenimports = [
    # The module itself
    "amplifier_module_tool_stt_whispercpp",
    "amplifier_module_tool_stt_whispercpp.tool",
    "amplifier_module_tool_stt_whispercpp.models",
    # Native bindings
    "pywhispercpp",
    "pywhispercpp.model",
    # Audio conversion (bundles FFmpeg)
    "av",
    "av.audio",
    "av.audio.resampler",
]
```

PyInstaller hooks for `av` (PyAV) are included in `pyinstaller-hooks-contrib`, which handles bundling the FFmpeg libraries automatically.

**Note:** The whisper.cpp models are NOT bundled in the executable. They are downloaded on first use to the user's local data directory. This keeps the executable small and allows users to choose which models to download.

## How It Works

1. **Audio Conversion**: PyAV (which bundles FFmpeg internally) converts any input audio to 16kHz mono WAV format required by whisper.cpp
2. **Model Loading**: pywhispercpp loads the GGML model file
3. **Transcription**: whisper.cpp processes the audio and returns segments with timestamps
4. **Result**: Returns full text and individual segments with timing

## Dependencies

- `amplifier-core` - Amplifier framework
- `pywhispercpp` - Python bindings for whisper.cpp
- `av` - PyAV for audio conversion (bundles FFmpeg)
- `httpx` - For downloading models

## License

MIT
