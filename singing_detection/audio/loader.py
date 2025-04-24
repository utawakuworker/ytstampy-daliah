import os
import tempfile
import numpy as np
import librosa
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union

class AudioLoader(ABC):
    """Abstract base class for audio loaders."""
    
    @abstractmethod
    def load_audio(self) -> Tuple[np.ndarray, int]:
        """
        Load audio data.
        
        Returns:
            Tuple[np.ndarray, int]: Audio signal and sample rate
        """
        pass
    
    @property
    @abstractmethod
    def duration(self) -> float:
        """Get audio duration in seconds."""
        pass
    
    @property
    @abstractmethod
    def file_path(self) -> str:
        """Get file path of loaded audio."""
        pass


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
            # Generate output filename template
            output_filename = os.path.join(self.output_dir, '%(title)s.%(ext)s')
            
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
                'no_warnings': False
            }
            
            # Download audio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(self.youtube_url, download=True)
                # Get the actual output file path
                if 'entries' in info:  # Playlist
                    info = info['entries'][0]  # Get first video in playlist
                
                # Construct the output path from the title and extension
                title = info['title']
                self._file_path = os.path.join(self.output_dir, f"{title}.mp3")
                
                # Check if file exists (yt-dlp may have sanitized the filename)
                if not os.path.exists(self._file_path):
                    # Try to find the file with a similar name
                    for file in os.listdir(self.output_dir):
                        if file.endswith('.mp3') and file.startswith(title[:10]):
                            self._file_path = os.path.join(self.output_dir, file)
                            break
            
            print(f"Downloaded to: {self._file_path}")
            
            # Load audio
            self._y, self._sr = librosa.load(self._file_path, sr=self.target_sr)
            self._duration = len(self._y) / self._sr
            
            return self._y, self._sr
            
        except Exception as e:
            print(f"Error downloading YouTube audio: {e}")
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