#!/usr/bin/env python3
"""Download short YouTube talks and transcribe them with Whisper.

Edit TALKS below, then run:
    conda run -n llm_course_env python hw_output/whisper_transcription/transcribe_youtube_talks.py
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import whisper

OUTPUT_DIR = Path(__file__).resolve().parent
AUDIO_DIR = OUTPUT_DIR / "audio"
TRANSCRIPTS_PATH = OUTPUT_DIR / "talks_transcripts.jsonl"
WHISPER_MODEL = "base"
LANGUAGE = "en"

TALKS = [
    {
        "talk_id": "pilot_001",
        "title": "Pilot NLP conference talk",
        "url": "https://www.youtube.com/watch?v=PFmVF93_f54",
        "start_seconds": None,
        "end_seconds": None,
    },
]


def run_command(command: list[str]) -> None:
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError("Command failed: " + " ".join(command) + "\n" + result.stderr)


def download_audio(talk: dict[str, Any]) -> Path:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    output_template = AUDIO_DIR / f"{talk['talk_id']}.%(ext)s"
    run_command(
        [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "m4a",
            "--output",
            str(output_template),
            talk["url"],
        ]
    )
    matches = sorted(AUDIO_DIR.glob(f"{talk['talk_id']}.*"))
    if not matches:
        raise FileNotFoundError(f"No audio file was created for {talk['talk_id']}")
    return matches[0]


def clip_audio_if_needed(talk: dict[str, Any], audio_path: Path) -> Path:
    start = talk.get("start_seconds")
    end = talk.get("end_seconds")
    if start is None and end is None:
        return audio_path

    clipped_path = AUDIO_DIR / f"{talk['talk_id']}_clip.wav"
    command = ["ffmpeg", "-y"]
    if start is not None:
        command += ["-ss", str(start)]
    command += ["-i", str(audio_path)]
    if end is not None and start is not None:
        command += ["-t", str(end - start)]
    elif end is not None:
        command += ["-to", str(end)]
    command += ["-ac", "1", "-ar", "16000", str(clipped_path)]
    run_command(command)
    return clipped_path


def transcribe_audio(model: Any, talk: dict[str, Any], audio_path: Path) -> dict[str, Any]:
    result = model.transcribe(str(audio_path), language=LANGUAGE, verbose=False)
    return {
        "talk_id": talk["talk_id"],
        "title": talk.get("title"),
        "url": talk["url"],
        "audio_path": str(audio_path),
        "language": result.get("language"),
        "text": result.get("text", "").strip(),
        "segments": [
            {
                "id": int(segment["id"]),
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": segment["text"].strip(),
            }
            for segment in result.get("segments", [])
        ],
    }


def main() -> None:
    if any("REPLACE_WITH_YOUTUBE_ID" in talk["url"] for talk in TALKS):
        raise ValueError("Replace the sample YouTube URL in TALKS before running.")

    model = whisper.load_model(WHISPER_MODEL)
    records = []
    for talk in TALKS:
        print(f"Downloading {talk['talk_id']}: {talk.get('title')}")
        raw_audio_path = download_audio(talk)
        audio_path = clip_audio_if_needed(talk, raw_audio_path)
        print(f"Transcribing {audio_path.name}")
        records.append(transcribe_audio(model, talk, audio_path))

    with TRANSCRIPTS_PATH.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} transcript(s) to {TRANSCRIPTS_PATH}")


if __name__ == "__main__":
    main()
