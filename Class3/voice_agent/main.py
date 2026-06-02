import os
import shutil
from urllib.parse import quote
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from asr import transcribe_audio
from llm import generate_response, get_history, clear_history
from tts import synthesize_speech

app = FastAPI(title="Voice Agent", description="5-turn voice chatbot: ASR → LLM → TTS")

app.mount("/static", StaticFiles(directory="static"), name="static")


def _cleanup(path: str):
    parent = os.path.dirname(path)
    if os.path.isdir(parent):
        shutil.rmtree(parent, ignore_errors=True)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    with open("static/index.html") as f:
        return f.read()


@app.post("/chat/", summary="Send audio, get audio reply")
async def chat_endpoint(
    file: UploadFile = File(..., description="Audio file (wav/mp3/webm)"),
    background_tasks: BackgroundTasks = None,
):
    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")

    user_text = transcribe_audio(audio_bytes)
    print(f"[User]      {user_text}")

    bot_text = generate_response(user_text)
    print(f"[Assistant] {bot_text}")

    wav_path = synthesize_speech(bot_text)
    if background_tasks:
        background_tasks.add_task(_cleanup, wav_path)

    return FileResponse(
        wav_path,
        media_type="audio/wav",
        headers={
            # URL-encode so non-ASCII characters survive HTTP headers
            "X-User-Text": quote(user_text),
            "X-Bot-Text":  quote(bot_text),
            "Access-Control-Expose-Headers": "X-User-Text, X-Bot-Text",
        },
    )


@app.get("/history/", summary="Get conversation history")
async def history_endpoint():
    return {"turns": len(get_history()) // 2, "history": get_history()}


@app.delete("/history/", summary="Clear conversation history")
async def clear_history_endpoint():
    clear_history()
    return {"message": "Conversation history cleared"}


@app.get("/health/")
async def health():
    return {"status": "ok"}
