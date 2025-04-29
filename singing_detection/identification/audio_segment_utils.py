import os
import subprocess
from typing import Optional


class AudioSegmentExtractor:
    """
    Utility class for extracting audio segments from a file using FFmpeg.
    """
    def __init__(self, audio_path: str, output_dir: Optional[str] = None):
        self.audio_path = audio_path
        self.output_dir = output_dir or os.path.dirname(audio_path)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

    def extract(self, start: float, end: float) -> Optional[str]:
        """
        Extracts an audio segment and saves it to a file.
        Args:
            start: Start time in seconds
            end: End time in seconds
        Returns:
            Path to the extracted audio segment, or None if extraction failed
        """
        try:
            segment_filename = f"segment_{start:.1f}_{end:.1f}.mp3"
            segment_path = os.path.join(self.output_dir, segment_filename)
            cmd = [
                "ffmpeg", "-y",
                "-i", self.audio_path,
                "-ss", str(start),
                "-to", str(end),
                "-c:a", "libmp3lame",
                "-q:a", "2",
                segment_path
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            if os.path.exists(segment_path):
                return segment_path
            return None
        except Exception as e:
            print(f"Error extracting segment: {e}")
            return None 