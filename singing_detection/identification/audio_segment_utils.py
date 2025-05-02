import os
import subprocess
import sys # Import sys
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
            # Determine the correct path for ffmpeg based on bundle type
            if hasattr(sys, '_MEIPASS'):
                # Running in a --onefile bundle
                # _MEIPASS is the temporary directory
                base_path = sys._MEIPASS
                ffmpeg_folder = '_internal' # Assuming you bundle it here
            elif getattr(sys, 'frozen', False):
                 # Running in a --onedir bundle (or frozen executable)
                 # sys.executable is the path to the executable
                 base_path = os.path.dirname(sys.executable)
                 ffmpeg_folder = '_internal' # Assuming you bundle it here
            else:
                # Not bundled - running as plain script
                # Assume ffmpeg is in PATH or handle as needed
                base_path = None
                ffmpeg_path = "ffmpeg" # Default to PATH

            # Construct path only if base_path is set (i.e., bundled)
            if base_path:
                 ffmpeg_path = os.path.join(base_path, ffmpeg_folder, 'ffmpeg.exe')
                 # Check if the bundled ffmpeg exists
                 if not os.path.exists(ffmpeg_path):
                     print(f"Error: Bundled ffmpeg not found at expected location: {ffmpeg_path}")
                     # Fallback or raise error? For now, try system PATH as last resort
                     ffmpeg_path = "ffmpeg"

            segment_filename = f"segment_{start:.1f}_{end:.1f}.mp3"
            segment_path = os.path.join(self.output_dir, segment_filename)

            # Use the determined ffmpeg_path in the command
            cmd = [
                ffmpeg_path, "-y",
                "-i", self.audio_path,
                "-ss", str(start),
                "-to", str(end),
                "-c:a", "libmp3lame", # Specify audio codec
                "-q:a", "2",          # Specify audio quality (VBR quality 2)
                segment_path
            ]
            # Run the command, hide console window on Windows
            startupinfo = None
            if os.name == 'nt': # Check if running on Windows
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, startupinfo=startupinfo)

            if os.path.exists(segment_path):
                return segment_path
            return None
        except FileNotFoundError: # Specifically catch if ffmpeg command itself fails
            print(f"Error: Failed to execute ffmpeg. Ensure ffmpeg is installed and in the system PATH, or bundled correctly.")
            print(f"  Attempted path: {ffmpeg_path}") # Log the path it tried
            return None
        except subprocess.CalledProcessError as e: # Catch errors during ffmpeg execution
            print(f"Error during ffmpeg execution: {e}")
            # Log stderr for more details
            print(f"FFmpeg stderr: {e.stderr.decode('utf-8', errors='ignore')}")
            return None
        except Exception as e:
            print(f"Error extracting segment: {e}")
            return None 