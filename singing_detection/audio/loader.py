import os
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Tuple

import librosa
import numpy as np


class AudioLoader(ABC):
    """Abstract base class for audio loaders."""
    
    @abstractmethod
    def load_audio(self) -> Tuple[np.ndarray, int]:
        """
        Load audio data.
        
        Returns:
            Tuple[np.ndarray, int]: Audio signal and sample rate
        """
    
    @property
    @abstractmethod
    def duration(self) -> float:
        """Get audio duration in seconds."""
    
    @property
    @abstractmethod
    def file_path(self) -> str:
        """Get file path of loaded audio."""


class LocalAudioLoader(AudioLoader):
    """Loader for local audio files."""
    
    def __init__(self, file_path: str, sr: Optional[int] = None):
        """
        Initialize local audio loader.
        
        Args:
            file_path: Path to audio file
            sr: Target sample rate (None for original)
        """
        self._file_path = file_path
        self.target_sr = sr
        self._y = None
        self._sr = None
        self._duration = None
    
    def load_audio(self) -> Tuple[np.ndarray, int]:
        """
        Load audio from file.
        
        Returns:
            Tuple[np.ndarray, int]: Audio signal and sample rate
        """
        if not os.path.exists(self._file_path):
            raise FileNotFoundError(f"Audio file not found: {self._file_path}")
        
        self._y, self._sr = librosa.load(self._file_path, sr=self.target_sr)
        self._duration = len(self._y) / self._sr
        
        return self._y, self._sr
    
    @property
    def duration(self) -> float:
        """Get audio duration in seconds."""
        if self._duration is None:
            # Load if not already loaded
            self.load_audio()
        return self._duration
    
    @property
    def file_path(self) -> str:
        """Get file path of loaded audio."""
        return self._file_path


class YouTubeAudioLoader(AudioLoader):
    """Loader for YouTube audio."""
    
    def __init__(self, youtube_url: str, output_dir: Optional[str] = None, sr: Optional[int] = None):
        """
        Initialize YouTube audio loader.
        
        Args:
            youtube_url: YouTube URL
            output_dir: Directory to save downloaded audio
            sr: Target sample rate (None for original)
        """
        self.youtube_url = youtube_url
        self.output_dir = output_dir or tempfile.gettempdir()
        self.target_sr = sr
        self._file_path = None
        self._y = None
        self._sr = None
        self._duration = None
        # Store video ID after download
        self._video_id = None 
    
    def load_audio(self) -> Tuple[np.ndarray, int]:
        """
        Download and load audio from YouTube using yt-dlp.
        
        Returns:
            Tuple[np.ndarray, int]: Audio signal and sample rate
        """
        try:
            import yt_dlp
        except ImportError:
            raise ImportError("yt-dlp is required for YouTube download. Install with: pip install yt-dlp")
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Downloading audio from YouTube: {self.youtube_url}")
        
        try:
            # Use video ID for the filename template
            output_filename = os.path.join(self.output_dir, '%(id)s.%(ext)s') 
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',  # Change to MP3 instead of WAV
                    'preferredquality': '192',
                }],
                'outtmpl': output_filename,
                'quiet': False,
                'no_warnings': False,
                'keepvideo': False # Don't keep the video file after extraction
            }
            
            # Download audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=True)
                
                # Get video ID and construct the expected path
                if 'entries' in info:  # Playlist
                    info = info['entries'][0]  # Get first video in playlist
                
                self._video_id = info['id']
                # Preferred extension is mp3 based on postprocessor
                expected_extension = 'mp3' 
                self._file_path = os.path.join(self.output_dir, f"{self._video_id}.{expected_extension}")
                
                # Check if file exists
                if not os.path.exists(self._file_path):
                    # Fallback check if extension somehow changed (less likely now)
                    possible_exts = ['m4a', 'webm', 'ogg'] # Common audio extensions yt-dlp might output
                    found = False
                    for ext in possible_exts:
                        potential_path = os.path.join(self.output_dir, f"{self._video_id}.{ext}")
                        if os.path.exists(potential_path):
                            self._file_path = potential_path
                            found = True
                            print(f"Warning: Expected .mp3, but found {ext}")
                            break
                    if not found:
                        raise FileNotFoundError(f"Downloaded audio file not found for video ID {self._video_id} at expected path: {self._file_path}")
            
            print(f"Downloaded to: {self._file_path}")
            
            # Load audio
            self._y, self._sr = librosa.load(self._file_path, sr=self.target_sr)
            self._duration = len(self._y) / self._sr
            
            return self._y, self._sr
            
        except Exception as e:
            print(f"Error downloading YouTube audio: {e}")
            # Clean up potentially partially downloaded file if path was set
            if self._file_path and os.path.exists(self._file_path):
                 try:
                     os.remove(self._file_path)
                     print(f"Cleaned up partially downloaded file: {self._file_path}")
                 except OSError as oe:
                     print(f"Error cleaning up file {self._file_path}: {oe}")
            raise
    
    @property
    def duration(self) -> float:
        """Get audio duration in seconds."""
        if self._duration is None:
            # Load if not already loaded
            self.load_audio()
        return self._duration
    
    @property
    def file_path(self) -> str:
        """Get file path of loaded audio."""
        if self._file_path is None:
            # Download if not already downloaded
            self.load_audio()
        return self._file_path


class AudioLoaderFactory:
    """Factory for creating audio loaders."""
    
    @staticmethod
    def create_loader(source: str, output_dir: Optional[str] = None, sr: Optional[int] = None) -> AudioLoader:
        """
        Create an appropriate audio loader based on the source.
        
        Args:
            source: Audio source (file path or YouTube URL)
            output_dir: Output directory for YouTube downloads
            sr: Target sample rate
            
        Returns:
            AudioLoader: An appropriate loader instance
        """
        if source.startswith(('http://', 'https://', 'www.')) or 'youtube' in source or 'youtu.be' in source:
            return YouTubeAudioLoader(source, output_dir, sr)
        else:
            return LocalAudioLoader(source, sr) 