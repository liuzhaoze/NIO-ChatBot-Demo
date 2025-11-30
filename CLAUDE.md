# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based real-time AI audio processing demonstration project that showcases real-time Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) capabilities using Alibaba DashScope's Qwen3 models.

## Common Development Commands

### Environment Setup
```bash
# Create and activate virtual environment (Python 3.13+)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies using UV
uv sync

# Alternative: Install with pip
pip install -e .
```

### Running the Demos
```bash
# Run real-time ASR demo (speech-to-text)
python qwen3_asr_flash_realtime_demo.py

# Run real-time TTS demo (text-to-speech)
python qwen3_tts_flash_realtime_demo.py

# Run basic entry point
python main.py
```

### Setting API Key
```bash
# Set DashScope API key (required)
export DASHSCOPE_API_KEY="your-api-key-here"
```

## Architecture Overview

### Core Components

**Real-time ASR Demo** (`qwen3_asr_flash_realtime_demo.py`):
- WebSocket-based real-time audio streaming for speech recognition
- 16kHz microphone input with live transcription
- Uses DashScope's Qwen3 ASR Flash model via `wss://dashscope.aliyuncs.com/api-ws/v1/realtime`
- Implements speech start/stop detection and audio recording to PCM files

**Real-time TTS Demo** (`qwen3_tts_flash_realtime_demo.py`):
- WebSocket-based text-to-speech synthesis
- Multiple text chunk processing with 24kHz PCM audio output
- Integrates with B64PCMPlayer for real-time audio playback
- Uses Qwen3 TTS Flash model with customizable voice options

**Audio Player Utility** (`B64PCMPlayer.py`):
- Multi-threaded Base64 PCM audio player
- Handles real-time audio streaming with configurable buffer sizes
- Thread-safe audio queuing and optional file output
- Supports various sample rates (16kHz, 24kHz)

### Dependencies and Technology Stack

- **Python 3.13+** (modern Python features)
- **DashScope SDK** (`dashscope>=1.25.2`) - Alibaba Cloud AI services
- **PyAudio** (`pyaudio>=0.2.14`) - Real-time audio I/O
- **Dify Client** - Local integration from `../dify/sdks/python-client`
- **UV** package manager with Tsinghua mirror for Chinese users

### Key Technical Patterns

**WebSocket Communication:**
- Both demos use WebSocket connections to DashScope's real-time API
- Callback-based event handling for session management
- Streaming audio data in chunks with base64 encoding

**Audio Processing Pipeline:**
- ASR: Microphone → PyAudio → Base64 encoding → WebSocket → Text transcription
- TTS: Text input → WebSocket → Base64 audio → B64PCMPlayer → Speaker output

**Real-time Considerations:**
- Chunk-based processing to minimize latency
- Multi-threading for concurrent audio operations
- Configurable buffer sizes (100ms chunks in TTS demo)

## Development Notes

- The project uses **UV** as the package manager for faster dependency resolution
- API keys are loaded from environment variables (never hardcoded)
- Audio files are stored in PCM format for direct compatibility with PyAudio
- Comprehensive debug logging is configured for development troubleshooting
- The codebase includes example PCM audio files for testing purposes

## Use Cases

This project serves as a reference implementation for:
- Real-time voice assistants and conversational AI
- Interactive audio applications requiring low latency
- Integration testing with Chinese language AI services
- Audio processing pipeline development and benchmarking
- Voice-enabled applications using Alibaba's DashScope platform