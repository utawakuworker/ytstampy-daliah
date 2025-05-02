import json
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

# Import the pipeline and config from the model package
try:
    from model.config import DEFAULT_CONFIG
except ImportError:
    import sys
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from model.config import DEFAULT_CONFIG

# Define path for user config file (in the same directory as main.py usually)
CONFIG_FILE_PATH = "user_config.json"

# --- Import Dataclasses ---
try:
    from model.data_models import (AnalysisResults, Segment,
                                   SegmentIdentification, SongIdentificationResult)
except ImportError:
    import os
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)
    from model.data_models import AnalysisResults, Segment, SegmentIdentification, SongIdentificationResult

from model.analysis_pipeline import (SingingAnalysisOrchestrator,
                                     StatusReporter)
from model.pipeline_steps import (AudioLoaderStep, FeatureExtractionStep,
                                  SaveResultsStep, SegmentDetectionStep,
                                  SegmentGroupingStep,
                                  SegmentProcessingStep,
                                  SongIdentificationStep, VisualizationStep)


class AnalysisViewModel:
    """
    Manages the state and logic for the Singing Analysis GUI.
    Acts as the intermediary between the View (GUI) and the Model (Pipeline).
    """
    def __init__(self, view_update_callback: Callable):
        """
        Args:
            view_update_callback: A function the ViewModel calls to signal
                                  the View that its state has changed and
                                  the UI needs refreshing.
        """
        self.view_update_callback = view_update_callback

        # --- UI State Properties ---
        # Configuration state (Initialize with defaults, then try loading saved)
        self.config: Dict[str, Any] = DEFAULT_CONFIG.copy()
        self._load_config_from_file() # Load saved config over defaults

        # Runtime state
        self.is_running: bool = False
        self.status_message: str = "Idle"
        self.progress_value: Optional[float] = 0.0
        self.log_messages: List[str] = ["[INFO] Application started."]
        if os.path.exists(CONFIG_FILE_PATH): # Add log message if config was loaded
             self.log_messages.append(f"[INFO] Loaded configuration from {CONFIG_FILE_PATH}")
        else:
             self.log_messages.append("[INFO] No saved configuration found, using defaults.")

        self._start_time: Optional[datetime] = None
        self._analysis_just_completed: bool = False

        # Results state
        self.analysis_results: Optional[AnalysisResults] = None

        # Internal state
        self.pipeline_thread: Optional[threading.Thread] = None
        self.result_context = None

        # Set up reporter with ViewModel-bound callbacks
        self.reporter = StatusReporter(
            status_callback=self._pipeline_status_update,
            progress_callback=self._pipeline_progress_update
        )

        # Compose the pipeline steps
        self.steps = [
            AudioLoaderStep(self.config),
            FeatureExtractionStep(self.config),
            SegmentDetectionStep(self.config),
            SegmentGroupingStep(self.config),
            SegmentProcessingStep(self.config),
            SongIdentificationStep(self.config),
            # VisualizationStep(self.config),      # Optional - Disabled
            SaveResultsStep(self.config),        # Optional
        ]
        self.pipeline = SingingAnalysisOrchestrator(self.steps, self.reporter)

    # --- Config Persistence ---
    def _load_config_from_file(self):
        """Loads configuration from the JSON file, merging with defaults."""
        try:
            if os.path.exists(CONFIG_FILE_PATH):
                with open(CONFIG_FILE_PATH, 'r') as f:
                    saved_config = json.load(f)
                    # Merge saved config onto defaults (saved values override defaults)
                    self.config.update(saved_config)
                    print(f"ViewModel: Loaded config from {CONFIG_FILE_PATH}") # Debug log
            else:
                 print(f"ViewModel: No config file found at {CONFIG_FILE_PATH}. Using defaults.")
        except (json.JSONDecodeError, IOError, Exception) as e:
            # Log error but continue with defaults
            print(f"Warning: Failed to load config from {CONFIG_FILE_PATH}: {e}")
            # Keep self.config as it was (initialized with DEFAULT_CONFIG)

    def _save_config_to_file(self):
        """Saves the current configuration to the JSON file."""
        try:
            with open(CONFIG_FILE_PATH, 'w') as f:
                # Save only the current self.config (which includes user input)
                json.dump(self.config, f, indent=4)
            print(f"ViewModel: Saved current config to {CONFIG_FILE_PATH}") # Debug log
        except (IOError, Exception) as e:
            print(f"Warning: Failed to save config to {CONFIG_FILE_PATH}: {e}")

    # --- Existing Methods ---
    def _notify_view(self):
        """Signals the view to update itself based on current ViewModel state."""
        self.view_update_callback()

    def _pipeline_status_update(self, message: str):
        """Receives status updates from the pipeline."""
        self.status_message = message
        self.log_messages.append(f"[STATUS] {message}")
        if len(self.log_messages) > 200:
            self.log_messages = self.log_messages[-200:]
        self._notify_view()

    def _pipeline_progress_update(self, value: Optional[float]):
        """Receives progress updates from the pipeline."""
        self.progress_value = value
        self._notify_view()

    def update_config_value(self, key: str, value: Any):
        """Updates a single configuration value."""
        print(f"ViewModel: Updating config '{key}' to '{value}'")
        self.config[key] = value
        # Consider if saving should happen here too, or only on start_analysis

    def update_full_config(self, new_config_values: Dict[str, Any]):
        """Updates the entire configuration dictionary (e.g., from GUI fields)."""
        self.config.update(new_config_values)
        print(f"ViewModel: Full config updated by View.")
        # Saving will happen in start_analysis

    def start_analysis(self):
        """Starts the analysis process in a background thread."""
        if self.is_running:
            self._pipeline_status_update("Analysis is already running.")
            return

        # --- Save the current config before starting ---
        self._save_config_to_file() # Save the config gathered from UI + existing

        # --- Proceed with starting analysis ---
        self.is_running = True
        self._start_time = datetime.now()
        self._analysis_just_completed = False
        self.analysis_results = None
        self.log_messages.append("[INFO] Starting analysis...") # Overwrite previous log? Or append? Append is safer.
        self.status_message = "Initializing..."
        self.progress_value = 0.0
        self._notify_view()

        current_config = self.config.copy() # Use the potentially updated and saved config
        # Make sure API key is handled (check saved config first, then env)
        if not current_config.get('gemini_api_key'):
             current_config['gemini_api_key'] = os.environ.get('GEMINI_API_KEY', '')
             if not current_config['gemini_api_key']:
                  self._pipeline_status_update("Warning: Gemini API Key not set in config or environment.")

        # Rebuild steps and pipeline with the latest config
        self.steps = [
            AudioLoaderStep(current_config),
            FeatureExtractionStep(current_config),
            SegmentDetectionStep(current_config),
            SegmentGroupingStep(current_config),
            SegmentProcessingStep(current_config),
            SongIdentificationStep(current_config),
            # VisualizationStep(current_config), # Disabled
            SaveResultsStep(current_config),
        ]
        self.pipeline = SingingAnalysisOrchestrator(self.steps, self.reporter)

        def run_in_thread():
            success = False
            try:
                self.result_context = self.pipeline.run()
                if self.result_context is not None:
                    # Optionally, convert context to AnalysisResults for compatibility
                    self.analysis_results = AnalysisResults(
                        total_duration=self.result_context.get('total_duration'),
                        final_segments=self.result_context.get('final_segments', []),
                        identification_results=self.result_context.get('identification_results', [])
                    )
                    success = True
            except Exception as e:
                self._pipeline_status_update(f"Pipeline Thread Error: {e}")
                import traceback
                self.log_messages.append(f"[ERROR] {traceback.format_exc()}")
            finally:
                self.is_running = False
                self._start_time = None
                if success:
                    self._analysis_just_completed = True
                    self._pipeline_status_update("Analysis completed successfully.")
                self._notify_view()

        self.pipeline_thread = threading.Thread(target=run_in_thread, daemon=True)
        self.pipeline_thread.start()

    def request_stop(self):
        """Requests the analysis pipeline to stop."""
        if not self.is_running:
             self._pipeline_status_update("Cannot stop: Analysis not running.")
             return

        self._pipeline_status_update("Requesting stop...")
        self._notify_view() # Show status update immediately
        if hasattr(self.pipeline, 'request_stop'):
             self.pipeline.request_stop()
        else:
             self.log_messages.append("[WARNING] Pipeline does not have a 'request_stop' method. Stopping may be delayed.")

    # --- Getter Methods for UI State ---
    def get_status_message(self) -> str:
        if self.is_running and self._start_time:
            elapsed = datetime.now() - self._start_time
            total_seconds = int(elapsed.total_seconds())
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            elapsed_str = f"{minutes:02}:{seconds:02}"
            return f"{self.status_message} (Elapsed: {elapsed_str})"
        return self.status_message

    def get_progress_value(self) -> Optional[float]:
        return self.progress_value

    def get_log_messages(self) -> list[str]:
        # Return a copy to prevent accidental modification by the view
        return list(self.log_messages)

    def check_and_reset_completion_flag(self) -> bool:
        """Checks if analysis just completed and resets the flag."""
        completed = self._analysis_just_completed
        if completed:
            self._analysis_just_completed = False
        return completed

    def get_detected_segments(self) -> List[Segment]:
        """Returns the final detected segments from the analysis results."""
        # Access attribute directly from the dataclass
        return self.analysis_results.final_segments if self.analysis_results else []

    def get_identification_results(self) -> List[SegmentIdentification]:
        """Returns the list of SegmentIdentification results from the analysis."""
        # Access the attribute of the AnalysisResults dataclass
        return self.analysis_results.identification_results if self.analysis_results else []

    def get_identified_songs(self) -> list:
        """
        Extracts song identification info from analysis results, including failed identifications.
        Returns a list of dictionaries with song info and error messages if applicable.
        """
        # Use the get_identification_results method which returns the correct list
        identification_results = self.get_identification_results()
        if not identification_results:
             return []

        identified_songs = []

        for result in identification_results:
            # result is a SegmentIdentification object
            song_info = result.identification  # This is a SongIdentificationResult object or None
            
            # Include all identification attempts, even failed ones
            if song_info:
                # Create a dictionary representation for output
                song_data = {
                    "title": song_info.title,
                    "artist": song_info.artist,
                    "confidence": song_info.confidence,
                    "explanation": song_info.explanation,
                    "segment_start": result.segment.start,  # Get start time from Segment object
                    "refined_lyrics": song_info.refined_lyrics_used,  # Pass refined lyrics too
                    "error": song_info.error  # Include any error message
                }
                
                # If there's an error but no title/artist, provide placeholder values
                if song_info.error and not (song_info.title or song_info.artist):
                    song_data["title"] = "Identification Failed"
                    song_data["artist"] = f"Error: {song_info.error[:50]}..."  # Truncate long error messages
                    song_data["confidence"] = "none"
                
                identified_songs.append(song_data)
            else:
                # Handle case where identification is None
                identified_songs.append({
                    "title": "Identification Failed",
                    "artist": "Unknown error occurred",
                    "confidence": "none",
                    "segment_start": result.segment.start,
                    "error": "No identification result available"
                })

        return identified_songs

    def get_youtube_comment_string(self) -> str:
        """Formats identified songs into a string suitable for YouTube comments."""
        identified_songs = self.get_identified_songs()  # This now returns list of dicts with needed info

        if not identified_songs:
            return "No songs identified (segments might have been too short or identification failed)."

        comment_lines = ["[Detected Songs - Timestamps]"]
        success_count = 0
        failed_count = 0

        for song in identified_songs:
            start_seconds = song.get('segment_start')
            if start_seconds is not None:
                timestamp = str(timedelta(seconds=int(start_seconds))).split('.')[0]  # Format as H:MM:SS
            else:
                timestamp = "? Unknown Time ?"

            # Check if this is a failed identification
            if song.get('error'):
                title = song.get('title', 'Identification Failed')
                artist = song.get('artist', 'Error')
                comment_lines.append(f"{timestamp} - ❌ {title}")
                failed_count += 1
            else:
                title = song.get('title', 'Unknown Title')
                artist = song.get('artist', 'Unknown Artist')
                comment_lines.append(f"{timestamp} - {artist} - {title}")
                success_count += 1

        # Add a summary at the end
        total = success_count + failed_count
        if failed_count > 0:
            comment_lines.append(f"\nIdentified {success_count} of {total} segments successfully. {failed_count} segments failed identification.")
        
        if success_count == 0 and failed_count > 0:
            comment_lines.append("No songs were successfully identified.")

        return "\n".join(comment_lines)

    def get_summary_info(self) -> Dict[str, Any]:
         # Check if results exist and contain final_segments
         if not self.analysis_results:
             return {"key": "summary_no_analysis", "kwargs": {}}

         # Access attributes from the dataclass
         segments: List[Segment] = self.analysis_results.final_segments

         # Check if segments list is empty
         if not segments:
              return {"key": "summary_no_segments", "kwargs": {}}

         # Check if total_duration exists in results
         total_duration = self.analysis_results.total_duration or 0 # Handle None

         # Calculate total singing time using tuple indexing
         total_singing_time = sum(seg.duration for seg in segments)

         percentage = (total_singing_time / total_duration * 100) if total_duration else 0
         return {
             "key": "summary_with_segments",
             "kwargs": {
                 "count": len(segments),
                 "total_time": f"{total_singing_time:.1f}", # Format here for simplicity
                 "percentage": f"{percentage:.1f}"        # Format here for simplicity
             }
         }

    def is_analysis_running(self) -> bool:
        return self.is_running

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Safely get a config value for the View to display."""
        # Now reads from self.config, which includes loaded values
        return self.config.get(key, default)

    def get_full_config(self) -> Dict[str, Any]:
        """Returns a copy of the current configuration."""
        # Now returns config potentially loaded from file
        return self.config.copy()

    def run_analysis(self):
        self.result_context = self.pipeline.run()
        # ... handle result_context as needed ... 

    def _show_save_feedback(self, saved_files, success=True):
        """
        Creates a feedback message for the UI after saving results.
        
        Args:
            saved_files: List of file paths that were saved
            success: Boolean indicating if the save was successful
        """
        if not saved_files or not success:
            return False
            
        # Format the paths for better display
        for i, file_path in enumerate(saved_files):
            try:
                # Show only filename to make message cleaner
                saved_files[i] = os.path.basename(file_path)
            except:
                pass
                
        files_str = ", ".join(saved_files)
        self.log_messages.append(f"[INFO] Files saved: {files_str}")
        return True
        
    def save_results(self):
        """
        Save analysis results manually when triggered from the UI.
        Saves segments and identification results to separate CSV files with UTF-8-SIG encoding.
        """
        if not self.analysis_results:
            self._pipeline_status_update("No results to save.")
            return False
        
        try:
            # Use existing config for output directory
            output_dir = self.config.get('output_dir', './output_analysis_gui')
            
            # Ensure output directory exists
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            # Generate a base filename using the current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save segments to CSV
            saved_files = []
            if self.analysis_results.final_segments:
                import pandas as pd
                segments_df = pd.DataFrame([
                    {'start': seg.start, 'end': seg.end, 'duration': seg.duration}
                    for seg in self.analysis_results.final_segments
                ])
                segments_csv_path = os.path.join(output_dir, f"segments_{timestamp}.csv")
                segments_df.to_csv(segments_csv_path, index=False, encoding='utf-8-sig', float_format='%.3f')
                saved_files.append(segments_csv_path)
            
            # Save song identification results to CSV if available
            identified_songs = self.get_identified_songs()
            if identified_songs:
                import pandas as pd
                songs_df = pd.DataFrame([
                    {
                        'segment_start': song.get('segment_start', 0),
                        'title': song.get('title', ''),
                        'artist': song.get('artist', ''),
                        'confidence': song.get('confidence', ''),
                        'error': song.get('error', '')
                    }
                    for song in identified_songs
                ])
                songs_csv_path = os.path.join(output_dir, f"identified_songs_{timestamp}.csv")
                songs_df.to_csv(songs_csv_path, index=False, encoding='utf-8-sig')
                saved_files.append(songs_csv_path)
            
            if saved_files:
                self._pipeline_status_update(f"Results saved to {', '.join(saved_files)}")
                success = self._show_save_feedback(saved_files)
            else:
                self._pipeline_status_update("No results to save.")
                self.log_messages.append("[INFO] No results to save.")
                success = False
            
            self._notify_view()
            return success
        except Exception as e:
            import traceback
            self._pipeline_status_update(f"Error saving results: {e}")
            self.log_messages.append(f"[ERROR] Error saving results: {e}")
            self.log_messages.append(f"[ERROR] {traceback.format_exc()}")
            self._notify_view()
            return False 