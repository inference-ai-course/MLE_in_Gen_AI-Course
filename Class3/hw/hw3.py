import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import whisper
import sys
from pathlib import Path
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import tempfile
import time

# Thread pool for CPU-bound operations
executor = ThreadPoolExecutor(max_workers=3)


app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Step 2: Load ASR model
asr_model = whisper.load_model("small")

# Step 3: Load LLM
# Using Ollama for better responses
OLLAMA_API = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # or "llama2", "mistral", etc.

conversation_history = []

# Speaker identification system
SPEAKERS_DB_FILE = "speakers_db.pkl"
SPEAKER_THRESHOLD = 0.75  # Similarity threshold for speaker recognition

class SpeakerDatabase:
    def __init__(self):
        self.speakers = {}  # {speaker_id: {"name": str, "embedding": np.array, "preferences": dict, "history": list}}
        self.load_database()

    def load_database(self):
        """Load speaker database from file"""
        if os.path.exists(SPEAKERS_DB_FILE):
            try:
                with open(SPEAKERS_DB_FILE, 'rb') as f:
                    self.speakers = pickle.load(f)
                print(f"Loaded {len(self.speakers)} speakers from database")
            except Exception as e:
                print(f"Error loading speaker database: {e}")

    def save_database(self):
        """Save speaker database to file"""
        try:
            with open(SPEAKERS_DB_FILE, 'wb') as f:
                pickle.dump(self.speakers, f)
        except Exception as e:
            print(f"Error saving speaker database: {e}")

    def extract_voice_embedding(self, audio_bytes):
        """Extract voice embedding from audio using Whisper encoder"""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            temp_path = f.name

        try:
            audio = whisper.load_audio(temp_path)
            audio = whisper.pad_or_trim(audio)
            mel = whisper.log_mel_spectrogram(audio).to(asr_model.device)
            embedding = mel.mean(dim=1).cpu().numpy()
            return embedding
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def identify_speaker(self, audio_bytes):
        """Identify speaker from voice embedding"""
        current_embedding = self.extract_voice_embedding(audio_bytes)

        if not self.speakers:
            return None, current_embedding

        # Compare with existing speakers
        max_similarity = 0
        best_match = None

        for speaker_id, speaker_data in self.speakers.items():
            similarity = cosine_similarity(
                current_embedding.reshape(1, -1),
                speaker_data["embedding"].reshape(1, -1)
            )[0][0]

            if similarity > max_similarity:
                max_similarity = similarity
                best_match = speaker_id

        # Return match if above threshold
        if max_similarity >= SPEAKER_THRESHOLD:
            return best_match, current_embedding
        else:
            return None, current_embedding

    def register_speaker(self, speaker_id, name, embedding, preferences=None):
        """Register new speaker"""
        self.speakers[speaker_id] = {
            "name": name,
            "embedding": embedding,
            "preferences": preferences or {},
            "history": []
        }
        self.save_database()

    def update_speaker_history(self, speaker_id, user_text, bot_response):
        """Update speaker conversation history"""
        if speaker_id in self.speakers:
            self.speakers[speaker_id]["history"].append({
                "user": user_text,
                "bot": bot_response
            })
            # Keep only last 20 conversations
            if len(self.speakers[speaker_id]["history"]) > 20:
                self.speakers[speaker_id]["history"] = self.speakers[speaker_id]["history"][-20:]
            self.save_database()

speaker_db = SpeakerDatabase()

# Step 4: Load TTS model (Coqui TTS - local, high quality)
coqui_tts = None
try:
    from TTS.api import TTS
    print("Loading Coqui TTS model...")
    coqui_tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
    print("Coqui TTS loaded successfully")
except Exception as e:
    print(f"Warning: Coqui TTS not available: {e}. Will use gTTS as fallback.")

# Step 2: ASR function (async wrapper)
async def transcribe_audio_async(audio_bytes):
    """Async wrapper for Whisper transcription"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, transcribe_audio_sync, audio_bytes)
    return result

def transcribe_audio_sync(audio_bytes):
    """Synchronous transcription function"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as f:
        f.write(audio_bytes)
        temp_path = f.name

    try:
        result = asr_model.transcribe(temp_path)
        return result["text"]
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Step 3: LLM response generation with Ollama (async) - with speaker personalization
async def generate_response_async(user_text, speaker_id=None, speaker_name=None):
    """Async LLM response generation using Ollama with speaker personalization"""
    conversation_history.append({"role": "user", "text": user_text})

    # Build conversation context with personalization
    context = ""

    # Add speaker context if known
    if speaker_id and speaker_name:
        speaker_data = speaker_db.speakers.get(speaker_id)
        if speaker_data:
            context += f"You are talking to {speaker_name}. "
            # Add recent history context
            recent_history = speaker_data.get("history", [])[-3:]
            if recent_history:
                context += "Previous conversations with this person:\n"
                for conv in recent_history:
                    context += f"User: {conv['user']}\nAssistant: {conv['bot']}\n"
            context += "\nCurrent conversation:\n"

    # Add current conversation context (last 5 turns)
    for turn in conversation_history[-5:]:
        if turn['role'] == 'user':
            context += f"User: {turn['text']}\n"
        else:
            context += f"Assistant: {turn['text']}\n"

    prompt = f"{context}Assistant:"

    try:
        # Call Ollama API asynchronously
        async with aiohttp.ClientSession() as session:
            async with session.post(
                OLLAMA_API,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "num_predict": 100
                    }
                },
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    bot_response = data["response"].strip()
                else:
                    bot_response = "I'm having trouble thinking right now. Could you try again?"

    except Exception as e:
        print(f"Ollama error: {e}")
        bot_response = "I'm having trouble connecting. Please make sure Ollama is running."

    conversation_history.append({"role": "assistant", "text": bot_response})

    # Keep only last 10 messages (5 turns)
    if len(conversation_history) > 10:
        conversation_history[:] = conversation_history[-10:]

    return bot_response

# Step 4: TTS function (async wrapper)
async def synthesize_speech_async(bot_text):
    """Async wrapper for TTS"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(executor, synthesize_speech_sync, bot_text)
    return result

def synthesize_speech_sync(bot_text):
    """Synchronous TTS function"""
    if coqui_tts is not None:
        print("Using Coqui TTS for speech synthesis")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
                output_path = f.name

            coqui_tts.tts_to_file(text=bot_text, file_path=output_path)
            print(f"Coqui TTS generated: {output_path}")
            return output_path

        except Exception as e:
            print(f"Coqui TTS error: {e}. Falling back to gTTS...")

    # Fallback to gTTS (requires internet)
    print("Using gTTS (requires internet)")
    from gtts import gTTS

    tts = gTTS(text=bot_text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as f:
        output_path = f.name
    tts.save(output_path)
    return output_path

# Step 5: Fully integrated endpoint with async processing and speaker identification
@app.post("/chat/")
async def chat_endpoint(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        # Read audio file
        audio_bytes = await file.read()
        print(f"Received audio file: {len(audio_bytes)} bytes")

        # Speaker identification (async)
        speaker_start = time.time()
        loop = asyncio.get_event_loop()
        speaker_id, embedding = await loop.run_in_executor(
            executor, speaker_db.identify_speaker, audio_bytes
        )
        print(f"Speaker identification took {time.time() - speaker_start:.2f}s")

        speaker_name = None
        if speaker_id:
            speaker_name = speaker_db.speakers[speaker_id]["name"]
            print(f"Identified speaker: {speaker_name} (ID: {speaker_id})")
        else:
            # New speaker detected
            print("New speaker detected. Checking if they introduce themselves...")

        # Step 1: Transcribe audio to text (async)
        asr_start = time.time()
        user_text = await transcribe_audio_async(audio_bytes)
        print(f"Transcribed text: {user_text} (took {time.time() - asr_start:.2f}s)")

        # Register new speaker if they introduce themselves
        if not speaker_id:
            # Check if user introduces themselves (simple pattern matching)
            lower_text = user_text.lower()
            if "my name is" in lower_text or "i am" in lower_text or "i'm" in lower_text:
                # Extract name (simplified)
                words = user_text.split()
                if "my name is" in lower_text:
                    idx = lower_text.split().index("is")
                    if idx + 1 < len(words):
                        speaker_name = words[idx + 1].strip('.,!?')
                elif "i am" in lower_text or "i'm" in lower_text:
                    # Find the name after "I am" or "I'm"
                    for i, word in enumerate(words):
                        if word.lower() in ["am", "i'm"]:
                            if i + 1 < len(words):
                                speaker_name = words[i + 1].strip('.,!?')
                                break

                if speaker_name:
                    speaker_id = f"speaker_{len(speaker_db.speakers) + 1}"
                    speaker_db.register_speaker(speaker_id, speaker_name, embedding)
                    print(f"Registered new speaker: {speaker_name} (ID: {speaker_id})")

        # Step 2: Generate response with LLM (with personalization)
        llm_start = time.time()
        bot_text = await generate_response_async(user_text, speaker_id, speaker_name)
        print(f"Generated response: {bot_text} (took {time.time() - llm_start:.2f}s)")

        # Update speaker history
        if speaker_id:
            speaker_db.update_speaker_history(speaker_id, user_text, bot_text)

        # Step 3: Convert text to speech (async)
        tts_start = time.time()
        audio_path = await synthesize_speech_async(bot_text)
        print(f"Generated audio: {audio_path} (took {time.time() - tts_start:.2f}s)")

        total_time = time.time() - start_time
        print(f"Total processing time: {total_time:.2f}s")

        # Determine media type based on file extension
        media_type = "audio/mpeg" if audio_path.endswith('.mp3') else "audio/wav"
        return FileResponse(audio_path, media_type=media_type)

    except Exception as e:
        print(f"Error in chat_endpoint: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))