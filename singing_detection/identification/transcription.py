# Standard library
import os
import sys
from typing import Optional

# Local/project
from .whisper_singleton import get_whisper_model


def _get_correct_path(relative_path):
    """ Get the absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)

class Transcriber:
    """
    Utility class for transcribing audio using Whisper.
    Handles model loading and transcription.
    """
    def __init__(self, model_name: str = "small"):
        self.model_name = model_name
        self.model = self._load_whisper_model()

    def _load_whisper_model(self):
        bundled_model_dir = None
        if getattr(sys, 'frozen', False):
            bundled_model_dir = _get_correct_path("whisper_models")
            expected_model_file = os.path.join(bundled_model_dir, f"{self.model_name}.pt")
            if not os.path.exists(expected_model_file):
                print(f"Warning: Bundled model file not found at {expected_model_file}")
                bundled_model_dir = None
        model_root = bundled_model_dir if bundled_model_dir else None
        return get_whisper_model(self.model_name, download_root=model_root)

    def transcribe(self, audio_path: str) -> Optional[str]:
        try:
            result = self.model.transcribe(audio_path)
            return result["text"].strip()
        except Exception as e:
            print(f"Error transcribing with Whisper: {e}")
            return None 