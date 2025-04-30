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
    Utility class for transcribing audio using Faster Whisper.
    Handles model loading and transcription.
    """
    def __init__(self, model_name: str = "small", device="cpu", compute_type="int8"):
        self.model_name = model_name
        # Store device and compute_type for model loading
        self.device = device
        self.compute_type = compute_type
        self.model = self._load_whisper_model()

    def _load_whisper_model(self):
        # Simplified loading - rely on the singleton's logic
        # Pass device and compute_type to the singleton getter
        try:
            # download_root logic is now handled within the singleton
            return get_whisper_model(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
        except Exception as e:
            print(f"Failed to load faster-whisper model {self.model_name} via singleton: {e}")
            # Depending on desired behavior, you might want to handle this more gracefully
            # For now, re-raising to make the failure obvious.
            raise

    def transcribe(self, audio_path: str) -> Optional[str]:
        """ Transcribes the audio file using the loaded faster-whisper model. """
        if not self.model:
            print("Error: Faster Whisper model not loaded.")
            return None
        try:
            # Faster-whisper returns an iterator of Segment objects and an info object
            segments, info = self.model.transcribe(audio_path, beam_size=5)
            
            # Optional: Log detected language
            print(f"Detected language '{info.language}' with probability {info.language_probability:.2f}")
            
            # Concatenate text from all segments
            full_text = " ".join(segment.text for segment in segments)
            
            return full_text.strip()
        except Exception as e:
            print(f"Error transcribing with Faster Whisper: {e}")
            return None

# Example Usage (Optional)
# if __name__ == '__main__':
#     # This requires a sample audio file named 'test_audio.mp3' in the same directory
#     if os.path.exists('test_audio.mp3'):
#         print("Initializing Transcriber (base model, cpu, int8)...")
#         transcriber = Transcriber(model_name='base', device='cpu', compute_type='int8')
#         print("Transcribing test_audio.mp3...")
#         transcript = transcriber.transcribe('test_audio.mp3')
#         if transcript:
#             print("\n--- Transcript ---")
#             print(transcript)
#             print("------------------\n")
#         else:
#             print("Transcription failed.")
#     else:
#         print("Skipping example usage: test_audio.mp3 not found.") 