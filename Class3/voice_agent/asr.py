import whisper
import os
import tempfile

_model = None


def _get_model():
    global _model
    if _model is None:
        print("[ASR] Loading Whisper medium model...")
        _model = whisper.load_model("medium")
        print("[ASR] Model loaded.")
    return _model


def transcribe_audio(audio_bytes: bytes) -> str:
    model = _get_model()
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name
    try:
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    finally:
        os.remove(tmp_path)
