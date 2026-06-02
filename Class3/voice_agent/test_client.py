"""Quick test: send 5 turns of audio to the voice agent and play each response."""
import argparse
import subprocess
import sys
import os
import requests

BASE_URL = "http://127.0.0.1:8000"


def send_audio(audio_path: str, turn: int):
    print(f"\n--- Turn {turn} ---")
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/chat/",
            files={"file": (os.path.basename(audio_path), f, "audio/wav")},
        )

    if resp.status_code != 200:
        print(f"Error {resp.status_code}: {resp.text}")
        return

    user_text = resp.headers.get("X-User-Text", "(unknown)")
    bot_text = resp.headers.get("X-Bot-Text", "(unknown)")
    print(f"[User]      {user_text}")
    print(f"[Assistant] {bot_text}")

    out_path = f"response_turn{turn}.wav"
    with open(out_path, "wb") as f:
        f.write(resp.content)
    print(f"[Saved]     {out_path}")

    subprocess.run(["afplay", out_path], check=False)


def main():
    parser = argparse.ArgumentParser(description="Test voice agent with audio files")
    parser.add_argument("audio_files", nargs="+", help="Audio files to send (up to 5)")
    args = parser.parse_args()

    for i, path in enumerate(args.audio_files[:5], start=1):
        send_audio(path, i)

    print("\n--- History ---")
    resp = requests.get(f"{BASE_URL}/history/")
    data = resp.json()
    print(f"Total turns: {data['turns']}")


if __name__ == "__main__":
    main()
