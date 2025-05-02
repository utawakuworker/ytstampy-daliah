# Standard library
import json
import os
import re
import sys
import tempfile
from typing import List, Optional, Any

# Third-party

from model.data_models import (Segment, SegmentIdentification,
                               SongIdentificationResult)
# Local/project
from singing_detection.identification.audio_segment_utils import \
    AudioSegmentExtractor
from singing_detection.identification.gemini_client import GeminiClient
from singing_detection.identification.transcription import Transcriber
from singing_detection.identification.whisper_singleton import get_whisper_model


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
                 gemini_api_key: Optional[str] = None,
                 ffmpeg_path: Optional[str] = None):
        """
        Initialize the song identifier.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory for temporary files and results
            whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
            gemini_api_key: Gemini API key
            ffmpeg_path: Path to ffmpeg executable (optional)
        """
        self.audio_path = audio_path
        self.output_dir = output_dir or tempfile.gettempdir()
        self.whisper_model_name = whisper_model
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.ffmpeg_path = ffmpeg_path
        
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
        segment_extractor = AudioSegmentExtractor(self.audio_path, self.output_dir, ffmpeg_path=self.ffmpeg_path)
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
        """ Loads the faster-whisper model using the singleton. """
        try:
            # Parameters like device and compute_type can be added here if needed
            # or read from configuration.
            # Using defaults from the singleton for now.
            return get_whisper_model(
                self.whisper_model_name
                # device="cpu", # Optional: specify device
                # compute_type="int8" # Optional: specify compute type
            )
        except Exception as e:
            print(f"Failed to load faster-whisper model {self.whisper_model_name} in SongIdentifier: {e}")
            # Handle error appropriately - perhaps raise it to stop initialization
            raise 