import os
import tempfile
import numpy as np
import subprocess
import json
import librosa
import requests
from typing import List, Tuple, Dict, Optional, Any
from pathlib import Path
import whisper
import sys

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
    """Class for identifying songs from audio segments using Whisper and Gemini."""
    
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
                      segments: List[Tuple[float, float]], 
                      min_segment_duration: float = 30.0,
                      max_segment_duration: float = 30.0,
                      verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Identify songs for each segment.
        
        Args:
            segments: List of (start, end) timestamp tuples
            min_segment_duration: Minimum duration for analysis
            max_segment_duration: Maximum duration for analysis
            verbose: Whether to print status
            
        Returns:
            List of dictionaries with identification results
        """
        results = []
        
        for i, (start, end) in enumerate(segments):
            segment_duration = end - start
            
            # Skip segments that are too short
            if segment_duration < min_segment_duration:
                if verbose:
                    print(f"Skipping segment {i+1} (too short: {segment_duration:.1f}s)")
                continue
            
            if verbose:
                print(f"Processing segment {i+1}: {start:.1f}s - {end:.1f}s (duration: {segment_duration:.1f}s)")
            
            # Extract audio segment
            if segment_duration > max_segment_duration:
                # If segment is too long, use the middle portion
                center = (start + end) / 2
                start = center - (max_segment_duration / 2)
                end = center + (max_segment_duration / 2)
                if verbose:
                    print(f"  Trimming segment to {start:.1f}s - {end:.1f}s ({max_segment_duration:.1f}s)")
            
            # Extract the segment
            segment_path = self._extract_segment(start, end)
            if not segment_path:
                if verbose:
                    print(f"  Failed to extract segment")
                continue
            
            # Transcribe with Whisper
            if verbose:
                print(f"  Transcribing with Whisper (model: {self.whisper_model_name})...")
            
            transcript = self._transcribe_with_whisper(segment_path)
            if not transcript:
                if verbose:
                    print(f"  No transcription available")
                continue
            
            if verbose:
                print(f"  Transcript: {transcript[:100]}..." if len(transcript) > 100 else f"  Transcript: {transcript}")
            
            # Identify song with Gemini
            if verbose:
                print(f"  Identifying song with Gemini...")
                
            identification_result = self._identify_with_gemini(transcript, start, end)
            
            # Store result
            result = {
                "segment": (start, end),
                "duration": end - start,
                "transcript": transcript,
                "identification": identification_result
            }
            results.append(result)
            
            if verbose:
                if identification_result.get("title"):
                    print(f"  Identified as: {identification_result.get('title')} by {identification_result.get('artist')}")
                else:
                    print(f"  Could not identify song")
            
        return results
    
    def _extract_segment(self, start: float, end: float) -> Optional[str]:
        """
        Extract audio segment and save to file.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Path to extracted audio segment, or None if extraction failed
        """
        try:
            # Generate output path
            segment_filename = f"segment_{start:.1f}_{end:.1f}.mp3"
            segment_path = os.path.join(self.output_dir, segment_filename)
            
            # Use FFmpeg to extract segment
            cmd = [
                "ffmpeg", "-y",  # Overwrite output file if it exists
                "-i", self.audio_path,
                "-ss", str(start),
                "-to", str(end),
                "-c:a", "libmp3lame",  # Use MP3 codec
                "-q:a", "2",  # Quality (2 = high quality)
                segment_path
            ]
            
            # Run FFmpeg
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            
            # Return path to segment
            if os.path.exists(segment_path):
                return segment_path
            return None
            
        except Exception as e:
            print(f"Error extracting segment: {e}")
            return None
    
    def _transcribe_with_whisper(self, audio_path: str) -> Optional[str]:
        """
        Transcribe audio using Whisper.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcription text, or None if transcription failed
        """
        try:
            # Load audio
            audio = self.whisper_model.load_audio(audio_path)
            
            # Transcribe
            result = self.whisper_model.transcribe(audio)
            
            # Return transcription
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error transcribing with Whisper: {e}")
            return None
    
    def _identify_with_gemini(self, 
                             transcript: str, 
                             start: float, 
                             end: float) -> Dict[str, Any]:
        """
        Identify song using Gemini API, focusing primarily on title and artist.
        
        Args:
            transcript: Transcribed lyrics
            start: Start time in seconds
            end: End time in seconds
            
        Returns:
            Dictionary with identification results
        """
        if not self.gemini_api_key:
            return {"error": "No Gemini API key provided"}
        
        try:
            # Format time for better context
            formatted_start = self._format_time(start)
            formatted_end = self._format_time(end)
            
            # Create simplified prompt for Gemini that focuses on title and artist
            # Added a note that the lyrics are transcribed and may contain errors.
            prompt = f"""
            I have a song segment from {formatted_start} to {formatted_end}. Below is an *automated transcription* of the lyrics from this segment, which may contain errors:

            {transcript}

            Please identify this song based on the transcribed lyrics, providing the following information:
            1. Song title
            2. Artist/band
            3. Confidence level (high, medium, low)

            Respond in JSON format with fields: title, artist, confidence, and explanation.
            If you cannot identify the song with confidence, provide your best guess and mark the confidence as "low".
            """
            
            # Call Gemini API
            response = self._call_gemini_api(prompt)
            
            # Parse JSON from response
            try:
                # Try to find JSON in the response (looking for curly braces)
                json_start = response.find('{')
                json_end = response.rfind('}')
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end+1]
                    result = json.loads(json_str)
                else:
                    # Fall back to simple parsing if JSON is not well-formed
                    result = self._parse_unstructured_response(response)
                
                return result
                
            except json.JSONDecodeError:
                # Fall back to simple parsing
                return self._parse_unstructured_response(response)
                
        except Exception as e:
            print(f"Error identifying with Gemini: {e}")
            return {"error": str(e)}
    
    def _call_gemini_api(self, prompt: str) -> str:
        """
        Call the Gemini API with the given prompt, enabling web search.

        Args:
            prompt: Prompt text

        Returns:
            API response text
        """
        # Note: Check availability and pricing for different models.
        api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            # --- Add Tool Configuration for Web Search ---
            "tools": [
                {
                    "google_search": {} # Enable Google Search tool
                }
            ],
            # --- Optional: Control how the tool is used ---
            # "tool_config": {
            #     "function_calling_config": {
            #         "mode": "AUTO" # Let the model decide when to search
            #     }
            # },
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "topK": 40
            }
        }

        # Call API with API key as parameter
        response = requests.post(
            f"{api_url}?key={self.gemini_api_key}",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            response_json = response.json()

            # Extract text from response - might be structured differently with tool use
            try:
                # Check if the response contains grounded content or just text
                candidate = response_json.get("candidates", [{}])[0]
                content = candidate.get("content", {})
                parts = content.get("parts", [{}])

                # Combine text parts, potentially including grounded results
                full_response_text = ""
                for part in parts:
                    if "text" in part:
                        full_response_text += part["text"] + "\n"
                    # You could potentially extract grounding metadata here if needed
                    # elif "grounding_metadata" in part:
                    #     pass

                return full_response_text.strip()

            except (KeyError, IndexError, TypeError) as e:
                print(f"Error parsing Gemini response: {e}")
                # Fallback to raw response text if parsing fails
                return response.text
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            # Return error message or empty string
            return f"API Error: {response.status_code}"
    
    def _parse_unstructured_response(self, response: str) -> Dict[str, Any]:
        """
        Parse an unstructured response to extract song title and artist.
        
        Args:
            response: Text response from Gemini
            
        Returns:
            Dictionary with parsed information
        """
        # Initialize result with just title and artist
        result = {
            "title": None,
            "artist": None,
            "confidence": "low",
            "explanation": ""
        }
        
        # Look for song title
        title_indicators = ["Title:", "Song title:", "Song:", "1."]
        for indicator in title_indicators:
            if indicator in response:
                # Extract line containing the indicator
                line = next((l for l in response.split('\n') if indicator in l), "")
                if line:
                    # Extract text after the indicator
                    title = line.split(indicator)[1].strip()
                    # Clean up any quotes
                    title = title.strip('"\'')
                    if title:
                        result["title"] = title
                        break
        
        # Look for artist
        artist_indicators = ["Artist:", "Band:", "By:", "2."]
        for indicator in artist_indicators:
            if indicator in response:
                # Extract line containing the indicator
                line = next((l for l in response.split('\n') if indicator in l), "")
                if line:
                    # Extract text after the indicator
                    artist = line.split(indicator)[1].strip()
                    # Clean up any quotes
                    artist = artist.strip('"\'')
                    if artist:
                        result["artist"] = artist
                        break
        
        # Look for confidence
        confidence_indicators = ["Confidence:", "Confidence level:", "3."]
        for indicator in confidence_indicators:
            if indicator in response:
                # Extract line containing the indicator
                line = next((l for l in response.split('\n') if indicator in l), "")
                if line:
                    # Extract text after the indicator
                    confidence = line.split(indicator)[1].strip().lower()
                    # Map to standard values
                    if "high" in confidence:
                        result["confidence"] = "high"
                    elif "medium" in confidence:
                        result["confidence"] = "medium"
                    else:
                        result["confidence"] = "low"
                    break
        
        # Set explanation from full response
        result["explanation"] = response
        
        return result
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds as HH:MM:SS.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes}:{secs:02d}"
    
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