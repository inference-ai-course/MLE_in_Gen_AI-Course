import subprocess
import tempfile
import os


def synthesize_speech(text: str) -> str:
    """Convert text to speech using macOS built-in `say` command.

    Returns path to a temporary WAV file (caller is responsible for cleanup).
    """
    tmp_dir = tempfile.mkdtemp()
    aiff_path = os.path.join(tmp_dir, "response.aiff")
    wav_path = os.path.join(tmp_dir, "response.wav")

    subprocess.run(
        ["say", "-o", aiff_path, "--", text],
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["ffmpeg", "-y", "-i", aiff_path, wav_path],
        check=True,
        capture_output=True,
    )

    os.remove(aiff_path)
    return wav_path
