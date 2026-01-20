# Voice Chatbot with Speaker Identification

A real-time voice chatbot that supports speech recognition, natural language understanding, text-to-speech, and speaker identification with personalized responses. This application is written with the assistance of Claude. It runs fully locally with limited support on multiple languages.

## Features

- 🎤 **Speech Recognition (ASR)**: Convert audio to text using OpenAI Whisper
- 🤖 **Dialogue Generation**: Natural responses using Ollama (Llama 3.2)
- 🔊 **Text-to-Speech (TTS)**: High-quality voice synthesis using Coqui TTS
- 👤 **Speaker Identification**: Recognizes returning users by voice
- 💬 **Conversation Memory**: Maintains 5-turn conversation history per speaker
- ⚡ **Async Processing**: Non-blocking operations for better performance
- 🌐 **Web Interface**: Beautiful frontend with microphone recording

## Architecture

### Components

1. **ASR (Automatic Speech Recognition)**: Whisper small model
2. **LLM (Large Language Model)**: Ollama with Llama 3.2
3. **TTS (Text-to-Speech)**: Coqui TTS (local) with gTTS fallback
4. **Speaker Identification**: Whisper-based voice embeddings with cosine similarity
5. **Web Server**: FastAPI with CORS support

### Why Not CosyVoice?

Initially, we planned to use CosyVoice for TTS, but switched to **Coqui TTS** for the following reasons:

1. **Model Availability**: CosyVoice requires downloading large pretrained models manually from the repository
2. **Setup Complexity**: CosyVoice has complex dependencies and setup requirements
3. **Accessibility**: Coqui TTS automatically downloads models on first use
4. **Reliability**: Coqui TTS is production-ready with better error handling
5. **Quality**: Both provide high-quality neural TTS, but Coqui is easier to deploy

**Fallback Chain**: Coqui TTS (primary) → gTTS (fallback if offline)

## Installation

### Prerequisites

- Python 3.8+
- Ollama installed and running
- macOS, Linux, or Windows

### Step 1: Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows
# Download from https://ollama.com/download
```

Start Ollama and pull the model:

```bash
ollama serve
ollama pull llama3.2
```

### Step 2: Install Python Dependencies

```bash
cd class3
pip install -r requirements.txt
```

### Step 3: Run the Server

```bash
uvicorn hw3:app --reload
```

The server will start at `http://127.0.0.1:8000`

## Usage

### Option 1: Web Interface (Recommended)

1. Open your browser to `http://127.0.0.1:8000`
2. Click "Start Recording" to record your voice
3. Click "Stop Recording" to send audio to the bot
4. Listen to the bot's voice response

### Option 2: Command Line with cURL

```bash
curl -X POST "http://127.0.0.1:8000/chat/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav" \
  --output response.wav
```

### Option 3: Python Script

```python
import requests

with open("your_audio.wav", "rb") as f:
    response = requests.post(
        "http://127.0.0.1:8000/chat/",
        files={"file": f}
    )

with open("bot_response.wav", "wb") as f:
    f.write(response.content)
```

## Speaker Identification

The bot can recognize users by their voice!

### First Time Usage

Say: **"Hello, my name is [Your Name]"**

The bot will register your voice and remember you in future conversations.

### Returning Users

Just start talking! The bot will:
- Recognize your voice automatically
- Greet you by name
- Recall your previous conversations (last 3 interactions)
- Provide personalized responses

### How It Works

1. Extracts voice embeddings using Whisper's mel spectrogram
2. Compares with stored speaker database using cosine similarity
3. Threshold: 0.75 for speaker recognition
4. Stores voice profile, name, and conversation history in `speakers_db.pkl`

## API Documentation

### Endpoints

#### `GET /`
Serves the web interface

#### `POST /chat/`
Main voice chat endpoint

**Request:**
- Content-Type: `multipart/form-data`
- Field: `file` (audio file, supports .wav, .webm, .mp3, etc.)

**Response:**
- Content-Type: `audio/wav` or `audio/mpeg`
- Body: Audio file with bot's voice response

**Processing Pipeline:**
1. Speaker identification (~0.5s)
2. Speech-to-text transcription (~1-3s)
3. LLM response generation (~2-5s)
4. Text-to-speech synthesis (~1-2s)

**Total latency**: ~5-10 seconds

## Configuration

### Modify LLM Model

Edit `hw3.py`:

```python
OLLAMA_MODEL = "llama3.2"  # Change to "llama2", "mistral", etc.
```

### Adjust Whisper Model Size

```python
asr_model = whisper.load_model("small")  # Options: tiny, base, small, medium, large
```

### Change TTS Voice/Speed

Coqui TTS supports multiple models:

```python
# Fast model
coqui_tts = TTS(model_name="tts_models/en/ljspeech/glow-tts")

# Multi-voice model
coqui_tts = TTS(model_name="tts_models/en/vctk/vits")
```

### Speaker Recognition Threshold

```python
SPEAKER_THRESHOLD = 0.75  # Range: 0.0-1.0 (higher = stricter)
```

## File Structure

```
class3/hw
├── hw3.py                 # Main application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── speakers_db.pkl       # Speaker database (auto-generated)
```

## Performance Tips

1. **GPU Acceleration**: Install CUDA for faster Whisper/TTS processing
2. **Model Size**: Use smaller models for faster response (trade-off: accuracy)
3. **Concurrent Requests**: Server handles multiple users simultaneously
4. **Memory**: Expect ~4-6GB RAM usage with all models loaded

## Troubleshooting

### Ollama Not Running
```
Error: I'm having trouble connecting. Please make sure Ollama is running.
```
**Solution**: Run `ollama serve` in a separate terminal

### Microphone Permission Denied
**Solution**: Enable microphone access in browser settings

### Slow Response Times
**Solutions**:
- Use smaller Whisper model (`tiny` or `base`)
- Reduce LLM `num_predict` tokens
- Check system resources (CPU/RAM)

### Speaker Not Recognized
**Solutions**:
- Ensure consistent audio quality
- Lower `SPEAKER_THRESHOLD` (e.g., 0.65)
- Re-register by saying your name again


## License

This project is for educational purposes as part of MLE Homework Class 3.

## Acknowledgments

- **OpenAI Whisper**: Speech recognition
- **Ollama**: LLM inference
- **Coqui TTS**: Text-to-speech synthesis
- **FastAPI**: Web framework
