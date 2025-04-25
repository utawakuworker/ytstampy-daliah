import os
import tempfile
import subprocess
import json
import requests # Added
from typing import List, Tuple, Dict, Optional, Any
import whisper
import sys
import time
# --- Custom Exceptions Removed --- #

# Import the new dataclasses
from model.data_models import Segment, SongIdentificationResult, SegmentIdentification


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
                 gemini_api_key: Optional[str] = None,
                 gemini_model_name: str = "gemini-2.0-flash",
                 whisper_fp16: bool = False,
                 generation_config_override: Optional[Dict[str, Any]] = None):
        """
        Initialize the song identifier.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory for temporary files and results
            whisper_model: Whisper model size ("tiny", "base", "small", "medium", "large")
            gemini_api_key: Gemini API key
            gemini_model_name: Specific Gemini model to use for identification
            whisper_fp16: Whether to use fp16 precision for Whisper (requires compatible GPU)
            generation_config_override: Optional dictionary to override Gemini generation settings
        """
        self.audio_path = audio_path
        self.output_dir = output_dir or tempfile.gettempdir()
        self.whisper_model_name = whisper_model
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.gemini_model_name = gemini_model_name
        self.whisper_fp16 = whisper_fp16
        self.generation_config_override = generation_config_override
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        
        self.whisper_model = self._load_whisper_model()
        
        # No SDK configuration needed here anymore
        if not self.gemini_api_key:
            print("Warning: Gemini API key not provided. Identification will fail.")
    
    def identify_songs(self, 
                      segments: List[Tuple[float, float]], 
                      min_segment_duration: float = 30.0,
                      max_segment_duration: float = 60.0,
                      verbose: bool = True) -> List[SegmentIdentification]:
        """
        Identify songs for each segment.
        
        Args:
            segments: List of (start, end) timestamp tuples
            min_segment_duration: Minimum duration for analysis
            max_segment_duration: Maximum duration for analysis
            verbose: Whether to print status
            
        Returns:
            List of SegmentIdentification dataclass instances
        """
        results: List[SegmentIdentification] = []
        
        # Use a single temporary directory for all segments in this run
        with tempfile.TemporaryDirectory() as temp_dir:
            if verbose:
                 print(f"Using temporary directory for segments: {temp_dir}")
                 
            for i, (start, end) in enumerate(segments):
                segment_duration = end - start
                
                # Skip segments that are too short
                if segment_duration < min_segment_duration:
                    if verbose:
                        print(f"Skipping segment {i+1} (too short: {segment_duration:.1f}s)")
                    continue
                
                if verbose:
                    print(f"Processing segment {i+1}: {start:.1f}s - {end:.1f}s (duration: {segment_duration:.1f}s)")
                
                # Define the current segment using the dataclass
                current_segment_obj = Segment(start=start, end=end)

                # Extract audio segment
                effective_start, effective_end = start, end
                if segment_duration > max_segment_duration:
                    # If segment is too long, use the middle portion
                    center = (start + end) / 2
                    effective_start = center - (max_segment_duration / 2)
                    effective_end = center + (max_segment_duration / 2)
                    if verbose:
                        print(f"  Trimming segment extraction to {effective_start:.1f}s - {effective_end:.1f}s ({max_segment_duration:.1f}s)")
                
                # Extract the potentially trimmed segment into the temporary directory
                segment_path = self._extract_segment(effective_start, effective_end, temp_dir)
                if not segment_path:
                    if verbose:
                        print("  Failed to extract segment")
                    # Consider if an error result should be added here
                    continue
                
                # Transcribe with Whisper
                if verbose:
                    print(f"  Transcribing with Whisper (model: {self.whisper_model_name})...")
                
                transcript = self._transcribe_with_whisper(segment_path)
                # Decision: Handle None transcript in the identification step
                
                if verbose:
                    print(f"  Transcript: {transcript[:100]}..." if transcript and len(transcript) > 100 else f"  Transcript: {transcript}")
                
                # Identify song with Gemini
                identification_result_obj: SongIdentificationResult
                if transcript: # Only attempt identification if we have a transcript
                    if verbose:
                         print(f"  Identifying song with Gemini...")
                    identification_result_obj = self._identify_with_gemini(transcript, start, end) # Pass original start/end for context
                else:
                    if verbose:
                        print("  Skipping Gemini identification due to missing transcript.")
                    # Create an error result if no transcript is available
                    identification_result_obj = SongIdentificationResult(error="Skipped identification: No transcript available")

                # Store result using dataclasses
                segment_id_result = SegmentIdentification(
                    segment=current_segment_obj,
                    transcript=transcript,
                    identification=identification_result_obj
                )
                results.append(segment_id_result)
                
                if verbose:
                    # Access attributes of the dataclass
                    if identification_result_obj and not identification_result_obj.error and identification_result_obj.title:
                        print(f"  Identified as: {identification_result_obj.title} by {identification_result_obj.artist}")
                    elif identification_result_obj and identification_result_obj.error:
                        print(f"  Identification error: {identification_result_obj.error}")
                    else:
                        print(f"  Could not identify song")
                
                # Check if the error was due to rate limiting and stop if so
                if identification_result_obj and identification_result_obj.error and "rate limit" in identification_result_obj.error.lower():
                    print("\n*** API Rate Limit Hit. Stopping further identification. ***\n")
                    break # Exit the loop over segments

                # Pause to respect API rate limits (e.g., 60 QPM)
                if verbose:
                    print("  Pausing for 1 second to respect API rate limits...")
                time.sleep(1)
            
        return results
    
    def _extract_segment(self, start: float, end: float, temp_dir: str) -> Optional[str]:
        """
        Extract audio segment and save to a temporary file.
        
        Args:
            start: Start time in seconds
            end: End time in seconds
            temp_dir: The temporary directory to save the segment in.
            
        Returns:
            Path to extracted audio segment, or None if extraction failed
        """
        try:
            # Generate output path within the temporary directory
            segment_filename = f"segment_{start:.1f}_{end:.1f}.mp3"
            segment_path = os.path.join(temp_dir, segment_filename)
            
            # Use FFmpeg to extract segment
            cmd = ["ffmpeg", "-y", "-i", self.audio_path, "-ss", str(start), "-to", str(end), "-c:a", "libmp3lame", "-q:a", "2", segment_path]
            
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
            # Transcribe directly from the file path
            result = self.whisper_model.transcribe(
                audio_path,
                fp16=self.whisper_fp16
            )
            
            # Return transcription
            return result["text"].strip()
            
        except Exception as e:
            print(f"Error transcribing with Whisper: {e}")
            return None
    
    def _identify_with_gemini(self, 
                             transcript: str, 
                             start: float, 
                             end: float) -> SongIdentificationResult:
        """
        Identify song using Gemini API based on a potentially noisy transcript,
        focusing primarily on title and artist, and requesting JSON output.
        Handles API errors (including rate limits) by returning an error result.
        Args:
            transcript: Raw transcribed lyrics from Whisper
            start: Start time in seconds
            end: End time in seconds

        Returns:
            SongIdentificationResult dataclass instance
        """
        if not self.gemini_api_key:
            print("Warning: Gemini API key not configured. Cannot identify.")
            return SongIdentificationResult(error="No Gemini API key provided")

        # --- Prepare Prompt --- #
        formatted_start = self._format_time(start)
        formatted_end = self._format_time(end)

        # Create identification prompt using the raw transcript
        identification_prompt = f"""
Below are the *automated transcriptions* of lyrics from a song segment. **These lyrics likely contain errors, noise, or mishearings.**
 
             Transcript:
{transcript}

Please identify the song based on the **entire transcript provided above**. **Use web search to verify against known song lyrics databases.** A confident match requires significant alignment between the transcript and known lyrics, not just isolated common phrases or words.

Provide the following information:
1. Song title (String)
2. Artist/band (String)
3. Confidence level (String: "high", "medium", or "low") - base this on how well the transcript matches known lyrics after potential verification.

- "high": A substantial portion of the transcript strongly matches verified lyrics.
- "medium": Some distinctive phrases match, but parts of the transcript are unclear or don't fit.
- "low": Only short/common phrases match, or the overall transcript coherence is poor compared to known lyrics.

Respond ONLY in valid JSON format with fields: "title", "artist", "confidence", and "explanation".
The "explanation" field should briefly mention the verification source (e.g., specific lyrics website) if web search was successful. Explain the confidence level, especially if low (e.g., "Matched phrase X, but the rest of the transcript is too noisy/doesn't match known lyrics for song Y", "Only common phrase matched", "Ambiguous lyrics").
If you cannot identify the song with at least medium confidence after searching, provide your best guess for title and artist, but MUST mark the confidence as "low".
**Do not assign high confidence based on very short or generic phrases, even if verified, unless the context strongly supports it.**
"""

        identification_response_str = ""
        try:
            # --- Call API --- #
            print("  Calling Gemini for song identification (using raw transcript)...")
            identification_response_str = self._call_gemini_api(
                identification_prompt,
                is_json_response_expected=True
            )

            # --- Parse Valid Response --- #
            print(f"  Raw identification response string: {identification_response_str}")

            # Parse the JSON response directly
            parsed_data = json.loads(identification_response_str)

            result_dict: Optional[Dict[str, Any]] = None
            if isinstance(parsed_data, list):
                if parsed_data and isinstance(parsed_data[0], dict):
                    result_dict = parsed_data[0]
                else:
                    raise ValueError("Received JSON list, but it was empty or contained non-dictionary items.")
            elif isinstance(parsed_data, dict):
                 result_dict = parsed_data
            else:
                raise ValueError(f"Received unexpected JSON type: {type(parsed_data)}")
            
            # Basic validation of expected keys
            if not all(k in result_dict for k in ("title", "artist", "confidence", "explanation")):
                  print("  Warning: Missing required fields in JSON response. Attempting unstructured parse.")
                  parsed_fallback = self._parse_unstructured_response(identification_response_str) # Fallback
                  if parsed_fallback.get("title") or parsed_fallback.get("artist"):
                     result_dict = parsed_fallback
                     result_dict.setdefault("confidence", "low")
                     result_dict.setdefault("explanation", "Parsed from unstructured response.")
                  else:
                     raise ValueError("JSON missing required fields & unstructured fallback failed.")
            
            # Create dataclass instance from validated dictionary
            return SongIdentificationResult(
                title=result_dict.get("title"),
                artist=result_dict.get("artist"),
                confidence=result_dict.get("confidence", "low"),
                explanation=result_dict.get("explanation"),
                refined_lyrics_used=None # This field seems unused now, maybe remove later?
            )

        except requests.exceptions.HTTPError as e:
            # Check specifically for Rate Limit error
            if e.response is not None and e.response.status_code == 429:
                print(f"  Identification skipped due to API rate limit (429).")
                return SongIdentificationResult(error="API rate limit exceeded.", refined_lyrics_used=None)
            else:
                # Handle other HTTP errors
                print(f"  Gemini API HTTP Error during identification: {e}")
                error_details = e.response.text if e.response is not None else "No response body"
                return SongIdentificationResult(error=f"Gemini API HTTP Error: {e.response.status_code if e.response is not None else 'N/A'} - {error_details[:100]}...", refined_lyrics_used=None)

        except requests.exceptions.RequestException as e:
            # Handle network/connection errors
            print(f"  Gemini API Request Error during identification: {e}")
            return SongIdentificationResult(error=f"Network/Request Error: {str(e)}", refined_lyrics_used=None)

        except json.JSONDecodeError as e:
            # Handle errors parsing the JSON response from the API
            print(f"  Warning: Could not parse Gemini identification JSON response: {e}. Response text: {identification_response_str}")
            parsed_fallback = self._parse_unstructured_response(identification_response_str)
            if parsed_fallback.get("title") or parsed_fallback.get("artist"):
                 print("    --> Fallback parse successful after JSONDecodeError.")
                 return SongIdentificationResult(
                    title=parsed_fallback.get("title"),
                    artist=parsed_fallback.get("artist"),
                    confidence="low",
                    explanation=f"AI response was not valid JSON, but info extracted: {parsed_fallback.get('explanation', '')[:100]}...",
                    error="Partially parsed non-JSON identification response",
                    refined_lyrics_used=None
                 )
            else:
                print("    --> Fallback parse failed after JSONDecodeError.")
                return SongIdentificationResult(
                    error=f"Failed to parse identification JSON response: {e}",
                    refined_lyrics_used=None
                )

        except (ValueError, KeyError, IndexError, TypeError) as e:
            # Handle errors related to configuration, blocked content, or unexpected response structure
            print(f"  Error processing Gemini response or configuration: {e}")
            return SongIdentificationResult(error=f"Processing Error: {str(e)}", refined_lyrics_used=None)

        except Exception as e: # Catch any other unexpected errors
            print(f"Unexpected error during identification: {e}")
            import traceback
            print(traceback.format_exc())
            return SongIdentificationResult(
                error=f"Unexpected identification error: {str(e)}",
                refined_lyrics_used=None
            )
    
    def _call_gemini_api(self, prompt: str, is_json_response_expected: bool = False) -> str:
        """
        Call the Gemini REST API using the requests library, enabling web search 
        and optionally enforcing JSON output.
        Raises standard exceptions (requests.exceptions, ValueError, etc.) on failure.

        Args:
            prompt: Prompt text
            is_json_response_expected: If True, configure the API to return only JSON.

        Returns:
            API response text (the content part).
        
        Raises:
            requests.exceptions.HTTPError: For 4xx/5xx status codes (including 429).
            requests.exceptions.RequestException: For network or request errors.
            json.JSONDecodeError: If the response is not valid JSON.
            ValueError/KeyError/IndexError/TypeError: If response JSON structure is unexpected.
        """
        if not self.gemini_api_key:
             # Using ValueError for configuration issues might be suitable
             raise ValueError("Gemini API Key not configured.")

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model_name}:generateContent?key={self.gemini_api_key}"
        headers = {
            "Content-Type": "application/json",
        }

        config_params = {
            "temperature": 0, # Low temperature for deterministic identification
            "topP": 0.95,
            "topK": 10,
        }
        if self.generation_config_override:
             config_params.update(self.generation_config_override)
        
        if is_json_response_expected:
             config_params['responseMimeType'] = "application/json"

        # --- Tool Config (for Web Search) --- #
        # Note: REST API uses tool_config for search, not the 'tools' field
        # Trying "google_search" as an alternative key suggested by user
        tool_config = {
            # "google_search_retrieval": {} # Use snake_case as often seen in REST examples
            "google_search": {} # Alternative key based on user feedback
        }

        # --- Request Payload --- #
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": config_params,
            # Remove the incorrect 'tools' field for search
            "tool_config": tool_config # Add the correct tool_config field
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload)

            # Check for HTTP errors (4xx/5xx) - includes 429 Rate Limit
            response.raise_for_status()

            # Parse Successful Response (can raise json.JSONDecodeError)
            response_data = response.json()
            
            # Extract Text Content (can raise KeyError, IndexError, TypeError)
            try:
                candidate = response_data['candidates'][0] # Let potential KeyError/IndexError propagate
                if candidate.get('finishReason') in ('SAFETY', 'RECITATION', 'OTHER'):
                     safety_ratings = candidate.get('safetyRatings', 'No safety ratings provided')
                     # Raise a ValueError for blocked content
                     raise ValueError(f"Gemini response blocked or flagged. Finish Reason: {candidate.get('finishReason')}, Safety: {safety_ratings}")

                # Navigate to the text part
                text_content = candidate['content']['parts'][0]['text'] # Let potential Key/Index/TypeError propagate
                return text_content
            except (KeyError, IndexError, TypeError) as e:
                # Re-raise parsing errors as ValueError for simpler catching upstream? Or let them propagate?
                # Let's let them propagate for now, more specific.
                print(f"Error parsing Gemini response structure: {e}. Response JSON: {response_data}")
                raise # Re-raise the original parsing error
        
        except requests.exceptions.HTTPError as e:
             # Handle HTTP errors (raised by raise_for_status())
             print(f"HTTP Error calling Gemini API: {e}")
             # Add specific check for rate limit in the calling function
             raise # Re-raise the HTTPError
             
        except requests.exceptions.RequestException as e:
            # Handle other request errors (connection, timeout, etc.)
            print(f"Error calling Gemini API (RequestException): {e}")
            raise # Re-raise the RequestException
        
        # No need for the broad Exception catch here anymore if we let others propagate
        # If needed, uncomment and potentially wrap in RuntimeError
        # except Exception as e:
        #      print(f"Unexpected error in _call_gemini_api: {e}")
        #      import traceback
        #      print(traceback.format_exc())
        #      raise RuntimeError(f"Unexpected error during API call: {e}") from e
    
    def _parse_unstructured_response(self, response: str) -> Dict[str, Any]:
        """
        Parse an unstructured response to extract song title and artist.
        
        Args:
            response: Text response from Gemini
            
        Returns:
            Dictionary with parsed information
        """
        # Initialize result with default values
        result: Dict[str, Any] = {
            "title": None,
            "artist": None,
            "confidence": "low",
            "explanation": ""
        }
        
        # Look for song title (including potential JSON key)
        title_indicators = ["Title:", "Song title:", "Song:", "1.", '\\"title\\":']
        for indicator in title_indicators:
            if indicator in response:
                line = next((l for l in response.split('\\n') if indicator in l), "")
                if line:
                    title = line.split(indicator)[-1].strip() # Take text after last occurrence
                    title = title.strip('"\\\', ') # Clean quotes, commas, spaces
                    if title:
                        result["title"] = title
                        break
        
        # Look for artist (including potential JSON key)
        artist_indicators = ["Artist:", "Band:", "By:", "2.", '\\"artist\\":']
        for indicator in artist_indicators:
            if indicator in response:
                line = next((l for l in response.split('\\n') if indicator in l), "")
                if line:
                    artist = line.split(indicator)[-1].strip() # Take text after last occurrence
                    artist = artist.strip('"\\\', ') # Clean quotes, commas, spaces
                    if artist:
                        result["artist"] = artist
                        break
        
        # Look for confidence (including potential JSON key)
        confidence_indicators = ["Confidence:", "Confidence level:", "3.", '\\"confidence\\":']
        for indicator in confidence_indicators:
            if indicator in response:
                line = next((l for l in response.split('\\n') if indicator in l), "")
                if line:
                    confidence = line.split(indicator)[-1].strip().lower()
                    confidence = confidence.strip('"\\\', ') # Clean quotes, commas, spaces
                    if "high" in confidence:
                        result["confidence"] = "high"
                    elif "medium" in confidence:
                        result["confidence"] = "medium"
                    else: # Default to low if present but not high/medium
                        result["confidence"] = "low"
                    break
        
        # Use original response as explanation if not otherwise set and nothing found
        if not result.get("explanation") and not result.get("title") and not result.get("artist"):
             result["explanation"] = f"Could not parse structured info. Raw response: {response[:200]}..."
        elif not result.get("explanation"):
             result["explanation"] = f"Parsed from potentially unstructured response."

        return result
    
    def _format_time(self, seconds: float) -> str:
        """
        Format time in seconds as HH:MM:SS or MM:SS.
        
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
        bundled_model_dir: Optional[str] = None

        # Check if running as a PyInstaller bundle
        if getattr(sys, 'frozen', False):
             # Path relative to the executable where we told PyInstaller to put models
             bundled_model_dir = _get_correct_path("whisper_models")
             print(f"Running in bundle, checking for models in: {bundled_model_dir}")
             # Check if the specific model file exists
             expected_model_file = os.path.join(bundled_model_dir, f"{model_name}.pt")
             if not os.path.exists(expected_model_file):
                  print(f"Warning: Bundled model file not found at {expected_model_file}")
                  bundled_model_dir = None # Fallback to default download/cache location
        else:
             print("Not running in bundle, using default Whisper model loading.")


        try:
            # If bundled path exists, tell whisper to use it, otherwise use default cache
            model_root = bundled_model_dir if bundled_model_dir else None
            print(f"Loading Whisper model '{model_name}' (root: '{model_root}')")
            return whisper.load_model(model_name, download_root=model_root)
        except Exception as e:
            print(f"Error loading Whisper model: {e}")
            raise # Re-raise the exception after printing 