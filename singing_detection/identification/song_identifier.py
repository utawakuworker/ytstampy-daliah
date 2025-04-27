import os
import tempfile
import numpy as np
import subprocess
import json
import requests
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import whisper
import sys
from singing_detection.identification.audio_segment_utils import AudioSegmentExtractor
from singing_detection.identification.transcription import Transcriber
from singing_detection.identification.gemini_client import GeminiClient
import re

# --- Import Data Models ---
# Assuming model is a sibling directory to singing_detection parent
try:
    # Adjust relative path based on your project structure
    # If song_identifier.py is in singing_detection/identification/
    # and data_models.py is in model/
    # We need to go up two levels then into model
    from ...model.data_models import Segment, SegmentIdentification, SongIdentificationResult
except ImportError:
    # Fallback if the relative import fails (e.g., running script directly)
    # This might require adding the project root to PYTHONPATH
    print("Warning: Relative import failed. Attempting direct import of data_models.")
    try:
        from model.data_models import Segment, SegmentIdentification, SongIdentificationResult
    except ImportError:
        print("Error: Could not import data models. Ensure 'model' is accessible.")
        sys.exit(1)

def _get_correct_path(relative_path):
    """ Get the absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # Not running in PyInstaller bundle, use script's directory
        base_path = os.path.dirname(os.path.abspath(__file__))
        # Adjust base_path if your bundled data is relative to the project root, not this file
        # For example, if this file is in singing_detection/identification and data is at root level:
        # base_path = os.path.dirname(os.path.dirname(os.path.dirname(base_path)))


    return os.path.join(base_path, relative_path)

class SongIdentifier:
    """
    Orchestrates the process of identifying songs from audio segments using modular components:
    - AudioSegmentExtractor for segment extraction
    - Transcriber for Whisper transcription
    - GeminiClient for lyrics correction and song identification
    """
    def __init__(self, 
                 audio_path: str,
                 output_dir: Optional[str] = None,
                 whisper_model: str = "small",
                 gemini_api_key: Optional[str] = None):
        """
        Initialize the song identifier.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory for temporary files and results
            whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
            gemini_api_key: Gemini API key
        """
        self.audio_path = audio_path
        self.output_dir = output_dir or tempfile.gettempdir()
        self.whisper_model_name = whisper_model
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.whisper_model = self._load_whisper_model()
    
    def identify_songs(self, 
                      segments: List[Segment], 
                      min_segment_duration: float = 30.0,
                      max_segment_duration: float = 30.0,
                      verbose: bool = True) -> List[SegmentIdentification]:
        """
        Identify songs for each segment using modular utilities.
        """
        def vprint(msg: str):
            if verbose:
                print(msg)

        results: List[SegmentIdentification] = []
        segment_extractor = AudioSegmentExtractor(self.audio_path, self.output_dir)
        transcriber = Transcriber(self.whisper_model_name)
        gemini = GeminiClient(self.gemini_api_key)
        for i, segment in enumerate(segments):
            try:
                if segment.duration < min_segment_duration:
                    vprint(f"Skipping segment {i+1} (too short: {segment.duration:.1f}s)")
                    continue
                vprint(f"Processing segment {i+1}: {segment.start:.1f}s - {segment.end:.1f}s (duration: {segment.duration:.1f}s)")
                extract_start, extract_end = self._get_extract_times(segment, max_segment_duration, vprint)
                segment_path = segment_extractor.extract(extract_start, extract_end)
                if not segment_path:
                    raise RuntimeError("Failed to extract audio segment")
                vprint(f"  Transcribing with Whisper (model: {self.whisper_model_name})...")
                transcript = transcriber.transcribe(segment_path)
                if not transcript:
                    raise RuntimeError("Transcription failed or yielded no text")
                vprint(f"  Transcript: {transcript[:100]}..." if len(transcript) > 100 else f"  Transcript: {transcript}")
                corrected_lyrics = gemini.correct_lyrics(transcript)
                vprint("  Corrected lyrics." if corrected_lyrics != transcript else "  No corrections needed")
                response_text = gemini.identify_song(corrected_lyrics, segment.start, segment.end)
                identification_result = self._parse_gemini_response(response_text)
                if identification_result and not identification_result.error:
                    identification_result.refined_lyrics_used = corrected_lyrics
                results.append(SegmentIdentification(segment=segment, transcript=transcript, identification=identification_result))
                vprint(self._format_identification_result(identification_result))
            except Exception as e:
                vprint(f"  Error: {e}")
                results.append(SegmentIdentification(segment=segment, transcript=None, identification=SongIdentificationResult(error=str(e))))
        return results
    
    def _get_extract_times(self, segment, max_duration, vprint):
        if segment.duration > max_duration:
            center = (segment.start + segment.end) / 2
            extract_start = center - (max_duration / 2)
            extract_end = center + (max_duration / 2)
            vprint(f"  Trimming segment for analysis to {extract_start:.1f}s - {extract_end:.1f}s ({max_duration:.1f}s)")
            return extract_start, extract_end
        return segment.start, segment.end

    def _format_identification_result(self, result):
        if result and result.title:
            return f"  Identified as: {result.title} by {result.artist} (Confidence: {result.confidence})"
        elif result and result.error:
            return f"  Identification error: {result.error}"
        else:
            return "  Could not identify song"

    def _parse_gemini_response(self, response_text: str) -> SongIdentificationResult:
        """
        Parse Gemini API response (JSON or unstructured) into a SongIdentificationResult.
        """
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end+1]
                try:
                    parsed_data = json.loads(json_str)
                    return SongIdentificationResult(
                        title=parsed_data.get("title"),
                        artist=parsed_data.get("artist"),
                        confidence=parsed_data.get("confidence", "low"),
                        explanation=parsed_data.get("explanation", response_text)
                    )
                except json.JSONDecodeError:
                    pass
            return self._parse_unstructured_response(response_text)
        except Exception as parse_err:
            print(f"Error parsing Gemini response or creating result object: {parse_err}")
            try:
                return self._parse_unstructured_response(response_text)
            except Exception as fallback_err:
                print(f"Error during fallback parsing: {fallback_err}")
                return SongIdentificationResult(error=f"Failed to parse response: {response_text}", explanation=response_text)
    
    def _parse_unstructured_response(self, response: str) -> SongIdentificationResult:
        """
        Fallback: Parse an unstructured Gemini response to extract song title, artist, and confidence.
        """
        def extract_value(indicators, lines):
            for indicator in indicators:
                pattern = re.compile(rf"{re.escape(indicator)}\s*(.*)", re.IGNORECASE)
                for line in lines:
                    match = pattern.match(line.strip())
                    if match:
                        value = match.group(1).strip().strip('"\'')
                        if value:
                            return value
            return None

        lines = response.splitlines()
        title = extract_value(["Title:", "Song title:", "Song:", "1."], lines)
        artist = extract_value(["Artist:", "Band:", "By:", "2."], lines)
        confidence_raw = extract_value(["Confidence:", "Confidence level:", "3."], lines)
        confidence = "low"
        if confidence_raw:
            if "high" in confidence_raw.lower():
                confidence = "high"
            elif "medium" in confidence_raw.lower():
                confidence = "medium"

        return SongIdentificationResult(
            title=title,
            artist=artist,
            confidence=confidence,
            explanation=response
        )
    
    def _load_whisper_model(self):
        """Loads the whisper model, checking for bundled version first."""
        model_name = self.whisper_model_name
        bundled_model_dir = None

        # Check if running as a PyInstaller bundle
        if getattr(sys, 'frozen', False):
             # Path relative to the executable where we told PyInstaller to put models
             # This matches the destination in --add-data "...;whisper_models"
             bundled_model_dir = _get_correct_path("whisper_models")
             print(f"Running in bundle, checking for models in: {bundled_model_dir}")
             # Check if the specific model file exists
             expected_model_file = os.path.join(bundled_model_dir, f"{model_name}.pt")
             if not os.path.exists(expected_model_file):
                  print(f"Warning: Bundled model file not found at {expected_model_file}")
                  bundled_model_dir = None # Fallback to default download/cache
        else:
             print("Not running in bundle, using default Whisper model loading.")


        try:
            # If bundled path exists, tell whisper to use it, otherwise use default cache
            model_root = bundled_model_dir if bundled_model_dir else None # whisper uses default if None
            print(f"Loading Whisper model '{model_name}' with download_root='{model_root}'")
            # Note: Whisper's load_model might not directly accept a full file path,
            # it usually expects a directory where it can find model_name.pt.
            # Setting download_root tells it *where* to look or download *to*.
            return whisper.load_model(model_name, download_root=model_root)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            # Handle error appropriately, maybe raise it or return None
            raise 