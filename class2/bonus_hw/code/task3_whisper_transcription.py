"""
Whisper Transcription Bot for YouTube Videos
Downloads audio from YouTube and transcribes using OpenAI Whisper with timestamps.
"""

import os
import json
import subprocess
from typing import List, Dict, Optional
import whisper


class WhisperTranscriptionBot:
    """YouTube audio transcription using Whisper."""

    def __init__(self, audio_dir: str = "youtube_audio", model_size: str = "base"):
        """
        Initialize the transcription bot.

        Args:
            audio_dir: Directory to store downloaded audio files
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.audio_dir = audio_dir
        self.model_size = model_size
        os.makedirs(self.audio_dir, exist_ok=True)

        print(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        print("Model loaded successfully")

    def download_audio(self, youtube_url: str, output_filename: Optional[str] = None) -> Optional[str]:
        """
        Download audio from YouTube using yt-dlp.

        Args:
            youtube_url: YouTube video URL
            output_filename: Custom output filename (without extension)

        Returns:
            Path to downloaded audio file or None if failed
        """
        try:
            print(f"Downloading audio from: {youtube_url}")

            # Generate output filename if not provided
            if output_filename is None:
                # Use video ID as filename
                video_id = youtube_url.split('v=')[-1].split('&')[0]
                output_filename = video_id

            output_path = os.path.join(self.audio_dir, f"{output_filename}.%(ext)s")

            # yt-dlp command to download best audio
            cmd = [
                'yt-dlp',
                '-f', 'bestaudio/best',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '0',
                '-o', output_path,
                youtube_url
            ]

            # Run yt-dlp
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Error downloading: {result.stderr}")
                return None

            # Find the downloaded file
            audio_file = os.path.join(self.audio_dir, f"{output_filename}.mp3")

            if os.path.exists(audio_file):
                print(f"Downloaded: {audio_file}")
                return audio_file
            else:
                print(f"Audio file not found: {audio_file}")
                return None

        except Exception as e:
            print(f"Error downloading audio: {e}")
            return None

    def transcribe_audio(self, audio_path: str, language: str = "en") -> Optional[Dict]:
        """
        Transcribe audio using Whisper.

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en' for English)

        Returns:
            Transcription result with timestamps or None if failed
        """
        try:
            print(f"Transcribing: {os.path.basename(audio_path)}")

            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                language=language,
                task="transcribe",
                verbose=False,
                word_timestamps=True
            )

            print(f"Transcription completed")
            return result

        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return None

    def format_transcript(self, result: Dict, youtube_url: str, video_title: str = "") -> Dict:
        """
        Format transcription result for JSONL output.

        Args:
            result: Whisper transcription result
            youtube_url: Original YouTube URL
            video_title: Video title

        Returns:
            Formatted transcript dictionary
        """
        # Extract segments with timestamps
        segments = []
        for segment in result.get('segments', []):
            segments.append({
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })

        # Full transcript text
        full_text = result.get('text', '').strip()

        transcript = {
            'url': youtube_url,
            'title': video_title,
            'language': result.get('language', 'en'),
            'duration': segments[-1]['end'] if segments else 0,
            'full_text': full_text,
            'segments': segments
        }

        return transcript

    def process_video(self, youtube_url: str, video_title: str = "",
                     output_filename: Optional[str] = None) -> Optional[Dict]:
        """
        Download and transcribe a single YouTube video.

        Args:
            youtube_url: YouTube video URL
            video_title: Video title for metadata
            output_filename: Custom filename for audio

        Returns:
            Formatted transcript or None if failed
        """
        # Download audio
        audio_path = self.download_audio(youtube_url, output_filename)

        if not audio_path:
            return None

        # Transcribe
        result = self.transcribe_audio(audio_path)

        if not result:
            return None

        # Format transcript
        transcript = self.format_transcript(result, youtube_url, video_title)

        return transcript

    def process_videos(self, videos: List[Dict[str, str]]) -> List[Dict]:
        """
        Process multiple YouTube videos.

        Args:
            videos: List of dicts with 'url' and optionally 'title' keys

        Returns:
            List of formatted transcripts
        """
        transcripts = []
        total = len(videos)

        for idx, video in enumerate(videos, 1):
            print(f"\n{'='*80}")
            print(f"Processing video {idx}/{total}")
            print(f"{'='*80}")

            url = video['url']
            title = video.get('title', f"Video {idx}")

            transcript = self.process_video(url, title)

            if transcript:
                transcripts.append(transcript)
                print(f"✓ Successfully transcribed: {title}")
            else:
                print(f"✗ Failed to transcribe: {title}")

        return transcripts

    def save_to_jsonl(self, transcripts: List[Dict], output_file: str = "talks_transcripts.jsonl"):
        """
        Save transcripts to JSONL file.

        Args:
            transcripts: List of transcript dictionaries
            output_file: Output JSONL filename
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            for transcript in transcripts:
                f.write(json.dumps(transcript, ensure_ascii=False) + '\n')

        print(f"\n{'='*80}")
        print(f"Saved {len(transcripts)} transcripts to: {output_file}")
        print(f"{'='*80}")


def main():
    """Main execution function."""

    # Example YouTube URLs for NLP conference talks
    # Replace these with actual NLP conference talk URLs
    videos = [
        {
            'url': 'https://www.youtube.com/shorts/AnaBQacfH50',
            'title': '1. Agentic AI Explained | GTC 2025 Keynote with NVIDIA CEO Jensen Huang'
        },
        {
            'url': 'https://www.youtube.com/shorts/k-RPpX2NMJ0',
            'title': '2. Zuckerberg, Nadella on AI increasing role in code'
        },
        {
            'url': 'https://www.youtube.com/shorts/BzPKqymsR7A',
            'title': '3. Scale AI’s Alexandr Wang on Securing U.S. AI'
        },
        {
            'url': 'https://www.youtube.com/shorts/RllOxzu_4Qk',
            'title': '4: Tim Baldwin: Fairness in Natural Language Processing'
        },
        {
            'url': 'https://www.youtube.com/shorts/ucDGG2R-x1M',
            'title': '5. sing Generative Artificial Intelligence tools to write codes for computational linguistics'
        },
        {
            'url': 'https://www.youtube.com/shorts/Al3RH-TX-yA',
            'title': '6. Armando Gonzalez, CEO, RavenPack'
        },
        {
            'url': 'https://www.youtube.com/shorts/1xXy2RfHHCE',
            'title': '7. Gen AI Unprecedented Pace'
        },
        {
            'url': 'https://www.youtube.com/shorts/7MZLgME6778',
            'title': '8. AI Keynote Speaker Shane Gibson on Maximizing Content Creation with AI'
        },
        {
            'url': 'https://www.youtube.com/shorts/QSKjD84Z6Rc',
            'title': '9: The arrival of non-human intelligence is a very big deal, says the former Google CEO'
        },
        {
            'url': 'https://www.youtube.com/shorts/9lQpF3_WmkI',
            'title': '10. Jensen Huang Advice for Students: Master AI to Shape the Future'
        },
        # Add more videos here (up to 10)
    ]

    # Note: Replace the URLs above with real YouTube URLs of NLP conference talks

    # Initialize bot with 'base' model (faster, good accuracy)
    # Options: tiny, base, small, medium, large
    bot = WhisperTranscriptionBot(audio_dir="youtube_audio", model_size="base")

    # Process all videos
    print(f"Processing {len(videos)} videos")
    transcripts = bot.process_videos(videos)

    # Save to JSONL
    bot.save_to_jsonl(transcripts, output_file="talks_transcripts.jsonl")

    # Print summary
    print(f"\nSummary:")
    print(f"  Total videos: {len(videos)}")
    print(f"  Successfully transcribed: {len(transcripts)}")
    print(f"  Failed: {len(videos) - len(transcripts)}")


if __name__ == "__main__":
    main()
