import os
import subprocess
import shutil
import tempfile
import sys
import atexit
from typing import Optional, Tuple, List


class AudioSegmentExtractor:
    """
    Utility class for extracting audio segments from a file using FFmpeg.
    """
    # Class-level list to track all temporary files created
    _temp_files: List[str] = []
    
    @classmethod
    def _register_cleanup(cls):
        """Register cleanup function to run at program exit"""
        atexit.register(cls._cleanup_temp_files)
    
    @classmethod
    def _cleanup_temp_files(cls):
        """Remove all temporary files created by all instances"""
        for file_path in cls._temp_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    print(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                print(f"Failed to clean up temporary file {file_path}: {e}")
        cls._temp_files.clear()
    
    def __init__(self, audio_path: str, output_dir: Optional[str] = None, ffmpeg_path: Optional[str] = None, use_temp_dir: bool = True):
        self.audio_path = audio_path
        
        # If no output_dir provided, or use_temp_dir is True, use system temp directory
        if output_dir is None or use_temp_dir:
            self.output_dir = tempfile.gettempdir()
            self.using_temp_dir = True
        else:
            self.output_dir = output_dir
            self.using_temp_dir = False
        
        # Register cleanup function if not already registered
        if not hasattr(AudioSegmentExtractor, "_cleanup_registered"):
            AudioSegmentExtractor._register_cleanup()
            AudioSegmentExtractor._cleanup_registered = True
        
        # Ensure output directory exists and is writable
        self._ensure_output_dir()
        
        # Set ffmpeg path (either custom or rely on system PATH)
        self.ffmpeg_path = ffmpeg_path if ffmpeg_path else self._find_ffmpeg()

    def _ensure_output_dir(self) -> None:
        """
        Ensures output directory exists and is writable.
        Falls back to temp directory if there are permission issues.
        """
        try:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir, exist_ok=True)
            
            # Test if directory is writable by creating a temp file
            test_file = os.path.join(self.output_dir, "test_write_permission.tmp")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except (PermissionError, OSError):
                print(f"Warning: Cannot write to {self.output_dir}, using system temp directory instead.")
                self.output_dir = tempfile.gettempdir()
                self.using_temp_dir = True
                
        except Exception as e:
            print(f"Warning: Error creating output directory {self.output_dir}: {e}. Using system temp directory.")
            self.output_dir = tempfile.gettempdir()
            self.using_temp_dir = True

    def _find_ffmpeg(self) -> str:
        """
        Find ffmpeg executable in system PATH or common locations.
        Returns the command name ('ffmpeg') or full path if found in common locations.
        """
        # First check if ffmpeg is in PATH
        ffmpeg_in_path = shutil.which("ffmpeg")
        if ffmpeg_in_path:
            return ffmpeg_in_path
        
        # Common locations to check on Windows
        common_locations = [
            # Executable files
            os.path.join(os.getcwd(), "ffmpeg.exe"),  # Current directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffmpeg.exe"),  # Project root
            "C:\\ProgramData\\chocolatey\\bin\\ffmpeg.exe",  # Chocolatey
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",  # Standard install
            "C:\\ffmpeg\\bin\\ffmpeg.exe",  # Manual install
            os.path.expanduser("~\\Documents\\ffmpeg\\bin\\ffmpeg.exe"),  # User Documents
            
            # Directories containing ffmpeg.exe
            os.path.join(os.getcwd(), "ffmpeg"),  # ffmpeg folder in current directory
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ffmpeg"),  # Project root/ffmpeg
            "C:\\Program Files\\ffmpeg\\bin",  # Standard install dir
            "C:\\ffmpeg\\bin",  # Manual install dir
            os.path.expanduser("~\\Documents\\ffmpeg\\bin"),  # User Documents
            # Common version-based paths
            os.path.expanduser("~\\Documents\\ffmpeg-7.1.1\\bin"),  # Version specific (7.1.1)
            os.path.expanduser("~\\Documents\\ffmpeg-6.0\\bin"),  # Version specific (6.0)
            os.path.expanduser("~\\Documents\\ffmpeg-5.1.2\\bin"),  # Version specific (5.1.2)
            os.path.expanduser("~\\Documents\\ffmpeg-4.4.1\\bin"),  # Version specific (4.4.1)
        ]
        
        # Check for executables first
        for location in common_locations:
            if os.path.exists(location):
                # If it's a directory, look for ffmpeg.exe inside
                if os.path.isdir(location):
                    exe_path = os.path.join(location, "ffmpeg.exe")
                    if os.path.exists(exe_path):
                        print(f"Found FFmpeg executable in directory: {exe_path}")
                        return exe_path
                else:
                    # It's a file path
                    print(f"Found FFmpeg executable at: {location}")
                    return location
        
        # If not found, scan user's Documents for any ffmpeg installation
        try:
            documents_path = os.path.expanduser("~\\Documents")
            if os.path.exists(documents_path):
                # Look for directories starting with 'ffmpeg'
                for item in os.listdir(documents_path):
                    if item.startswith("ffmpeg") and os.path.isdir(os.path.join(documents_path, item)):
                        bin_dir = os.path.join(documents_path, item, "bin")
                        exe_path = os.path.join(bin_dir, "ffmpeg.exe")
                        if os.path.exists(exe_path):
                            print(f"Found FFmpeg executable in Documents: {exe_path}")
                            return exe_path
        except Exception as e:
            print(f"Error scanning Documents directory: {e}")
        
        # If not found, return just the command name and let the system try to resolve it
        return "ffmpeg"

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
            # Create a unique filename to avoid collisions
            unique_suffix = os.urandom(4).hex()
            segment_filename = f"segment_{start:.1f}_{end:.1f}_{unique_suffix}.mp3"
            segment_path = os.path.join(self.output_dir, segment_filename)
            
            # Use ffmpeg_path instead of just "ffmpeg"
            ffmpeg_command = self.ffmpeg_path
            
            # Check if ffmpeg exists and is executable
            self._check_ffmpeg_executable(ffmpeg_command)
            
            # At this point, self.ffmpeg_path may have been updated by _check_ffmpeg_executable
            # if a directory was provided, so use the updated path
            ffmpeg_command = self.ffmpeg_path
            
            # Check if input file exists and is readable
            if not os.path.exists(self.audio_path):
                raise FileNotFoundError(f"Input audio file not found: {self.audio_path}")
            
            cmd = [
                ffmpeg_command, "-y",
                "-i", self.audio_path,
                "-ss", str(start),
                "-to", str(end),
                "-c:a", "libmp3lame",
                "-q:a", "2",
                segment_path
            ]
            
            # On Windows, add shell=True to help with permission issues
            is_windows = sys.platform.startswith('win')
            
            print(f"Running FFmpeg command: {' '.join(cmd)}")
            subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=True,
                shell=is_windows
            )
            
            if os.path.exists(segment_path):
                # If using temp directory, register file for cleanup
                if self.using_temp_dir:
                    AudioSegmentExtractor._temp_files.append(segment_path)
                return segment_path
            return None
        except PermissionError as e:
            error_msg = self._format_permission_error(e)
            print(f"Error extracting segment: {error_msg}")
            return None
        except subprocess.CalledProcessError as e:
            # Handle ffmpeg process errors (will contain ffmpeg's stderr)
            error_output = e.stderr.decode('utf-8', errors='replace') if e.stderr else str(e)
            print(f"FFmpeg error: {error_output}")
            return None
        except Exception as e:
            print(f"Error extracting segment: {e}")
            return None

    def _check_ffmpeg_executable(self, ffmpeg_path: str) -> None:
        """Check if ffmpeg is executable and handles potential permission issues."""
        # If the path is a directory, look for ffmpeg executable inside it
        if os.path.isdir(ffmpeg_path):
            possible_exes = ["ffmpeg.exe", "ffmpeg"]
            found = False
            for exe in possible_exes:
                full_path = os.path.join(ffmpeg_path, exe)
                if os.path.exists(full_path):
                    self.ffmpeg_path = full_path  # Update the path to use the actual executable
                    ffmpeg_path = full_path       # Use the full path for the rest of this method
                    found = True
                    print(f"Found FFmpeg executable at: {full_path}")
                    break
            
            if not found:
                raise FileNotFoundError(f"No FFmpeg executable found in directory: {ffmpeg_path}")
                
        # Standard check if it's not a directory
        if not os.path.exists(ffmpeg_path) and ffmpeg_path != "ffmpeg":
            raise FileNotFoundError(f"FFmpeg not found at: {ffmpeg_path}")
        
        # If it's a path (not just 'ffmpeg' command), check if executable
        if ffmpeg_path != "ffmpeg" and os.path.exists(ffmpeg_path):
            if not os.access(ffmpeg_path, os.X_OK):
                raise PermissionError(f"No permission to execute FFmpeg at: {ffmpeg_path}")

    def _format_permission_error(self, error: Exception) -> str:
        """Format a permission error with helpful suggestions."""
        error_str = str(error)
        suggestions = [
            "Make sure you have permission to access the input and output files.",
            "Try running the application as administrator.",
            "Check if the output directory is writable.",
            "If using a custom FFmpeg path, verify you have permission to execute it."
        ]
        return f"{error_str}\nSuggestions:\n- " + "\n- ".join(suggestions)

    @staticmethod
    def check_ffmpeg_availability(ffmpeg_path: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if ffmpeg is available at the specified path or in system PATH.
        Returns (success, message) tuple where:
        - success: True if ffmpeg is available, False otherwise
        - message: Version string on success, error message on failure
        """
        try:
            # If the path is a directory, look for ffmpeg executable inside it
            if ffmpeg_path and os.path.isdir(ffmpeg_path):
                possible_exes = ["ffmpeg.exe", "ffmpeg"]
                found = False
                for exe in possible_exes:
                    full_path = os.path.join(ffmpeg_path, exe)
                    if os.path.exists(full_path):
                        ffmpeg_path = full_path  # Use the full path for the check
                        found = True
                        break
                
                if not found:
                    return False, f"No FFmpeg executable found in directory: {ffmpeg_path}"
                    
            cmd = [ffmpeg_path or "ffmpeg", "-version"]
            # On Windows, add shell=True to help with permission issues
            is_windows = sys.platform.startswith('win')
            
            result = subprocess.run(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                check=False,
                shell=is_windows
            )
            
            if result.returncode == 0:
                # Extract version from output
                version_output = result.stdout.decode('utf-8', errors='replace')
                version_line = version_output.splitlines()[0] if version_output else "FFmpeg is available"
                return True, version_line
            else:
                error_output = result.stderr.decode('utf-8', errors='replace')
                return False, f"FFmpeg error: {error_output}"
        except PermissionError as e:
            return False, f"Permission error: {e}. Try running as administrator."
        except FileNotFoundError:
            return False, "FFmpeg not found in PATH or at specified location."
        except Exception as e:
            return False, f"Error checking FFmpeg: {e}" 