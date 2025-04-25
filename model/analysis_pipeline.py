import os
import sys
import json
import warnings
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple, Callable
import re # Import regex module for sanitization
from urllib.parse import urlparse, parse_qs # Import URL parsing tools
import numpy as np # Ensure numpy is imported

# --- Import Dataclasses ---
from .data_models import AnalysisResults, Segment, SegmentIdentification, SongIdentificationResult

# --- Setup Environment ---
warnings.filterwarnings('ignore')

# --- Import Project Modules ---
# Assuming singing_detection is now a sibling directory or in PYTHONPATH
try:
    from singing_detection.audio.loader import AudioLoaderFactory
    from singing_detection.audio.feature_extraction import FeatureExtractorFacade
    from singing_detection.detection.detection_engine import ClusterEnhancedHMMDetectionEngine
    from singing_detection.segments.segment_processor import SegmentFilter, SegmentProcessingPipeline, SegmentRefiner
    from singing_detection.visualization.plots import plot_waveform_with_segments, plot_feature_comparison
    from singing_detection.identification.song_identifier import SongIdentifier
except ImportError as e:
    print(f"Error importing project modules in analysis_pipeline.py: {e}")
    # Depending on how you run, you might need path adjustments
    # Example: Add parent directory if running script directly from model/
    # parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # if parent_dir not in sys.path:
    #     sys.path.append(parent_dir)
    #     from singing_detection.audio.loader import AudioLoaderFactory # etc.
    raise # Re-raise the error after attempting path fix or logging

# --- Helper Function ---
def sanitize_filename(name: str, max_length: int = 100) -> str:
    """Removes or replaces invalid filename characters and limits length."""
    if not name:
        return "unknown_source"
    # Remove invalid characters (using regex for efficiency)
    # Allow letters, numbers, underscore, hyphen, period
    sanitized = re.sub(r'[^a-zA-Z0-9_.-]+', '_', name)
    # Replace multiple underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    # Remove leading/trailing underscores/periods
    sanitized = sanitized.strip('_.')
    # Limit length
    return sanitized[:max_length] if len(sanitized) > max_length else sanitized

# --- Core Pipeline Class ---

class SingingAnalysisPipeline:
    """
    Encapsulates the singing detection and identification process.
    Designed to be used by a ViewModel, providing status and progress updates via callbacks.
    """

    def __init__(self,
                 config: Dict[str, Any],
                 status_callback: Optional[Callable[[str], None]] = None,
                 progress_callback: Optional[Callable[[Optional[float]], None]] = None):
        """
        Initializes the pipeline.

        Args:
            config: Dictionary containing configuration parameters.
            status_callback: Function to call with status update messages (str).
            progress_callback: Function to call with progress updates (float 0.0-1.0 or None for indeterminate).
        """
        self.config = config
        # Use provided callbacks or default to no-op lambdas
        self.status_callback = status_callback or (lambda msg: None)
        self.progress_callback = progress_callback or (lambda val: None)

        self.loader = None
        self.extractor = None
        self.detector = None
        self.segment_processor = None
        self.identifier = None
        self.results: Dict[str, Any] = {} # Stores intermediate data
        self._stop_requested: bool = False # Flag for stopping

        # Main output directory from config
        self.main_output_dir = self.config.get('output_dir', './output_analysis_model')
        os.makedirs(self.main_output_dir, exist_ok=True)
        self.run_output_dir = self.main_output_dir # Placeholder, will be updated after filename generation
        # self._update_status(f"Main output directory set to: {self.main_output_dir}") # Initial status update

    # --- Callback Helpers ---

    def _update_status(self, message: str):
        """Uses the callback to report status."""
        # print(f"Pipeline Status: {message}") # Optional: Keep console log for debugging
        self.status_callback(message)

    def _update_progress(self, value: Optional[float], message: Optional[str] = None):
        """Uses the callback to report progress and optionally status."""
        # Example: value=None for indeterminate, 0.0-1.0 for determinate
        if message:
            self._update_status(message)
        self.progress_callback(value)

    # --- Pipeline Steps (Adapted for Callbacks) ---

    def _generate_base_filename(self) -> bool:
        """Generates a safe base filename from the input source."""
        source_file = self.config.get('file')
        source_url = self.config.get('url')
        base_name = "analysis" # Default fallback

        if source_file:
            try:
                base_name = os.path.splitext(os.path.basename(source_file))[0]
            except Exception:
                pass # Keep default if path parsing fails
        elif source_url:
            try:
                parsed_url = urlparse(source_url)
                if 'youtube.com' in parsed_url.netloc and 'v' in parse_qs(parsed_url.query):
                    base_name = parse_qs(parsed_url.query)['v'][0]
                elif 'youtu.be' in parsed_url.netloc:
                    base_name = parsed_url.path.lstrip('/')
                else:
                    # Fallback for other URLs: use netloc + path
                     base_name = f"{parsed_url.netloc}{parsed_url.path.replace('/', '_')}"

            except Exception:
                 pass # Keep default if URL parsing fails

        sanitized_base = sanitize_filename(base_name)
        self.results['base_filename'] = sanitized_base

        # Create and set the specific output directory for this run
        self.run_output_dir = os.path.join(self.main_output_dir, sanitized_base)
        try:
            os.makedirs(self.run_output_dir, exist_ok=True)
            self._update_status(f"Output for this run will be in: {self.run_output_dir}")
            return True
        except OSError as e:
             self._update_status(f"Error creating output subdirectory '{self.run_output_dir}': {e}. Using main directory.")
             self.run_output_dir = self.main_output_dir # Fallback
             return False

    def _load_audio(self) -> bool:
        """Loads audio using the factory."""
        source = self.config.get('url') or self.config.get('file')
        if not source:
            self._update_status("Error: No audio source (URL or file) specified.")
            return False

        self._update_progress(None, f"Loading audio source: {os.path.basename(str(source))}")
        try:
            # Pass the *main* output_dir, loader might need it for initial download/conversion
            self.loader = AudioLoaderFactory.create_loader(source, output_dir=self.main_output_dir)
            y, sr = self.loader.load_audio()
            self.results['y'] = y
            self.results['sr'] = sr
            self.results['audio_path'] = self.loader.file_path # Keep the path to the raw/converted audio
            self.results['total_duration'] = self.loader.duration
            self._update_status(f"Audio loaded: {self.results['total_duration']:.1f}s @ {sr}Hz")
            return True # Now generate filename *after* successful load
        except FileNotFoundError:
            self._update_status(f"Error: Audio file not found at {source}")
            return False
        except ImportError as e:
             self._update_status(f"Error: Missing dependency for loading. {e}")
             return False
        except Exception as e:
            self._update_status(f"Error loading audio: {e}")
            import traceback
            print(traceback.format_exc()) # Log full error for debugging
            return False

    def _extract_features(self) -> bool:
        """Extracts audio features."""
        self._update_progress(0.1, "Extracting features...") # Example progress point
        enable_hpss_flag = self.config.get('enable_hpss', True) # Read flag from config
        try:
            # Using 2-second windows as in the original script example
            self.extractor = FeatureExtractorFacade(
                include_pitch=False, # As per original script example
                enable_hpss=enable_hpss_flag, # Pass the flag
                window_size_seconds=2.0
            )
            feature_df = self.extractor.extract_all_features(
                self.results['y'], self.results['sr']
            )
            self.results['feature_df'] = feature_df
            if feature_df.empty:
                self._update_status("Warning: No features extracted.")
                return False # Treat as failure if no features
            self._update_status(f"Extracted {len(feature_df)} feature frames.")
            return True
        except Exception as e:
            self._update_status(f"Error extracting features: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def _detect_segments(self) -> bool:
        """Detects singing segments using the HMM engine."""
        if 'feature_df' not in self.results or self.results['feature_df'].empty:
            self._update_status("Skipping detection: No features available.")
            # Decide if this is an error or just means no segments
            self.results['initial_segments'] = []
            self.results['frame_df_with_states'] = pd.DataFrame() # Empty dataframe
            return True # Allow pipeline to continue, just with no segments

        self._update_progress(0.3, "Detecting singing segments (HMM)...")
        try:
            self.detector = ClusterEnhancedHMMDetectionEngine()

            # Define reference segments based on config time points
            ref_dur = self.config.get('ref_duration', 2.0)
            sing_start = self.config.get('singing_ref_time', 0) # Default to 0 if missing
            nsing_start = self.config.get('non_singing_ref_time', 0)
            total_dur = self.results.get('total_duration', float('inf'))

            singing_ref = (max(0, sing_start), min(total_dur, sing_start + ref_dur))
            non_singing_ref = (max(0, nsing_start), min(total_dur, nsing_start + ref_dur))
            self._update_status(f"Using Singing Ref: {singing_ref[0]:.1f}-{singing_ref[1]:.1f}s")
            self._update_status(f"Using Non-Singing Ref: {non_singing_ref[0]:.1f}-{non_singing_ref[1]:.1f}s")

            segments, frame_df_with_states = self.detector.detect_singing(
                self.results['y'], self.results['sr'], self.results['feature_df'],
                singing_ref, non_singing_ref,
                threshold=self.config.get('hmm_threshold', 0.55),
                min_duration=self.config.get('min_segment_duration', 10.0),
                min_gap=self.config.get('min_segment_gap', 1.5),
                visualize=False, # Visualization handled separately by caller
                dim_reduction=self.config.get('dim_reduction', 'pca'),
                n_components=self.config.get('n_components', 4),
                verbose=False # Reduce console noise from HMM
            )
            self.results['initial_segments'] = segments
            self.results['frame_df_with_states'] = frame_df_with_states
            self._update_status(f"Initial detection found {len(segments)} segments.")
            return True
        except Exception as e:
            self._update_status(f"Error during segment detection: {e}")
            import traceback
            print(traceback.format_exc())
            return False

    def _process_segments(self) -> bool:
        """Refines and filters the detected segments."""
        initial_segments = self.results.get('initial_segments')
        if initial_segments is None: # Check if detection step failed or was skipped
             self._update_status("Skipping segment processing: No initial segments.")
             self.results['final_segments'] = [] # Store empty list of Segments
             return True # Allow pipeline to continue

        if not initial_segments: # Empty list means detection ran but found nothing
             self._update_status("No initial segments to process.")
             self.results['final_segments'] = [] # Store empty list of Segments
             return True

        self._update_progress(0.6, "Processing detected segments...")
        try:
            # Initialize processors without config args in __init__
            segment_refiner = SegmentRefiner()
            segment_filter = SegmentFilter()

            # Pass parameters when calling process_segments
            # Note: SegmentRefiner might need different params (e.g., window_ms, threshold)
            # Let's assume it uses defaults for now, or adjust if needed later.
            # Refiner is often less crucial than filter/merger.
            refined_segments = segment_refiner.process_segments(
                segments=initial_segments,
                y=self.results['y'],
                sr=self.results['sr']
                # Pass specific params for refiner if needed, e.g.:
                # window_ms=200, threshold=0.25
            )

            # Pass min_duration to the filter's process_segments
            filtered_segments = segment_filter.process_segments(
                segments=refined_segments,
                y=self.results['y'],
                sr=self.results['sr'],
                # Pass params expected by SegmentFilter.process_segments
                min_duration=self.config.get('min_segment_duration', 5.0),
                merge_threshold=self.config.get('merge_threshold', 0.5), # Example param, adjust if needed
                verbose=False
            )

            # Convert final list of tuples to list of Segment dataclasses
            final_segment_objects = [Segment(start=s[0], end=s[1]) for s in filtered_segments]
            self.results['final_segments'] = final_segment_objects

            self._update_status(f"Processed segments: {len(filtered_segments)} final segments remaining.")
            return True
        except Exception as e:
            self._update_status(f"Error processing segments: {e}")
            import traceback
            print(traceback.format_exc())
            # Fallback: use initial segments if processing fails? Or fail pipeline?
            # self.results['final_segments'] = initial_segments # Option: Fallback
            return False # Option: Fail

    def _identify_songs(self) -> bool:
        """Identifies songs in the final segments."""
        if self._stop_requested: return False # Check stop flag

        final_segments = self.results.get('final_segments')
        audio_path = self.results.get('audio_path')

        if not final_segments:
            self._update_status("Skipping song identification: No final segments to analyze.")
            self.results['identification_results'] = [] # Ensure key exists, empty list
            return True # Not an error, just nothing to do

        if not audio_path:
            self._update_status("Error: Cannot identify songs, audio path is missing.")
            self.results['identification_results'] = []
            return False # This is an error

        # Check for API key
        api_key = self.config.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self._update_status("Warning: Gemini API key not found. Skipping song identification.")
            self.results['identification_results'] = [] # Ensure key exists, empty list
            return True # Allow pipeline to finish, but without identification

        # Create identifier instance once
        self.identifier = SongIdentifier(
            audio_path=audio_path,
            output_dir=self.run_output_dir, # Pass the subdirectory path here
            whisper_model=self.config.get('whisper_model', 'base'),
            gemini_api_key=api_key
        )

        try:
            # Convert Segment objects back to tuples if identify_songs expects that
            # *** Correction: identify_songs was updated to expect tuples ***
            # Keep using the Segment objects directly if identify_songs is updated
            # segment_tuples = [(seg.start, seg.end) for seg in final_segments]
            # *** UPDATE: identify_songs *still* expects tuples based on its signature ***
            # Let's refactor identify_songs to accept List[Segment] later if needed.
            # For now, convert back to tuples.
            segment_tuples_for_id = [(seg.start, seg.end) for seg in final_segments]

            self._update_progress(0.8, f"Identifying songs in {len(final_segments)} segments...")
            # The identify_songs method now returns List[SegmentIdentification]
            identification_results: List[SegmentIdentification] = self.identifier.identify_songs(
                segments=segment_tuples_for_id, # Pass the tuples
                min_segment_duration=self.config.get('min_duration_for_id', 30.0),
                max_segment_duration=self.config.get('max_duration_for_id', 60.0),
                verbose=False # Let pipeline handle status updates
            )

            # Store the list of SegmentIdentification dataclasses directly
            self.results['identification_results'] = identification_results

            num_identified = sum(1 for res in identification_results if res.identification and not res.identification.error and res.identification.title)
            self._update_status(f"Song identification attempted. Found potential titles for {num_identified} segments.")
            return True
        except Exception as e:
            self._update_status(f"Error during song identification: {e}")
            import traceback
            print(traceback.format_exc())
            self.results['identification_results'] = [] # Ensure key exists, empty list on error
            return False

    # --- Control Methods ---

    def run(self) -> Optional[AnalysisResults]:
        """
        Executes the full analysis pipeline.

        Returns:
            An AnalysisResults dataclass instance, or None if a critical step failed.
        """
        self._update_progress(0.0, "Starting analysis pipeline...")
        self._stop_requested = False # Reset stop flag

        if not self._generate_base_filename(): return None
        if self._stop_requested or not self._load_audio(): return None
        if self._stop_requested or not self._extract_features(): return None
        if self._stop_requested or not self._detect_segments(): return None
        if self._stop_requested or not self._process_segments(): return None
        if self._stop_requested or not self._identify_songs():
            # If song ID fails, maybe still return detection results?
            self._update_status("Warning: Song identification failed, but returning detection results.")
            # Continue to create AnalysisResults object with available data
            pass # Continue to create AnalysisResults object

        # --- Final Result Packaging --- #
        final_results = AnalysisResults(
            total_duration=self.results.get('total_duration'),
            final_segments=self.results.get('final_segments', []),
            identification_results=self.results.get('identification_results', [])
            # Add other fields from self.results if needed in the dataclass
        )

        # Store the final results object internally in case save_results needs it
        self.results['final_analysis_results'] = final_results

        self._update_progress(1.0, "Analysis pipeline completed.")
        return final_results # Return the dataclass instance

    def visualize(self):
        """Generates and saves visualizations based on results."""
        # Import matplotlib only when needed and handle if it's missing
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            self._update_status("Visualization skipped: matplotlib library not found or excluded.")
            # Don't consider this a failure of the pipeline, just skip visualization
            return

        # Check if visualization is enabled in config
        if not self.config.get('visualize', False):
            self._update_status("Visualization disabled in config.")
            return

        # Check if necessary data exists
        if 'y' not in self.results or 'final_segments' not in self.results:
             self._update_status("Cannot visualize: Missing audio data or segments.")
             return

        base_filename = self.results.get('base_filename', 'analysis_plot')
        self._update_status(f"Generating visualization files for {base_filename}...")
        try:
            fig1_path = os.path.join(self.run_output_dir, f"{base_filename}_waveform_segments.png")
            # Get the List[Segment] and convert back to tuples for the plotting function
            final_segment_objects: List[Segment] = self.results.get('final_segments', [])
            segment_tuples_for_plot = [(seg.start, seg.end) for seg in final_segment_objects]

            # Create and save the first plot (waveform)
            plt.figure(figsize=(15, 5))
            plot_waveform_with_segments(
                y=self.results.get('y'),
                sr=self.results.get('sr'),
                segments=segment_tuples_for_plot, # Pass tuples
                title=f"Detected Singing Segments - {base_filename}",
                # Pass output_path directly to the function if it supports it
                # otherwise, save manually after plotting
            )
            plt.savefig(fig1_path)
            plt.close()
            self._update_status(f"Saved waveform plot: {fig1_path}")

            # Create and save the second plot (features) if available
            if 'frame_df_with_states' in self.results and not self.results['frame_df_with_states'].empty:
                fig2_path = os.path.join(self.run_output_dir, f"{base_filename}_feature_comparison.png")
                plt.figure(figsize=(15, 7))
                plot_feature_comparison(self.results['frame_df_with_states'], title=f"Feature Comparison ({base_filename})")
                plt.savefig(fig2_path)
                plt.close()
                self._update_status(f"Saved feature plot: {fig2_path}")
        except Exception as e: # Catch potential errors during plotting/saving
            self._update_status(f"Error generating visualization files: {e}")
            import traceback
            print(traceback.format_exc())

    def save_results(self):
        """Saves the analysis results to files."""
        # Attempt to get the final results object stored by the run() method
        final_analysis_results: Optional[AnalysisResults] = self.results.get('final_analysis_results')

        # Determine base filename from intermediate results (should always exist if run started)
        base_filename = self.results.get('base_filename')
        if not base_filename:
             # This case indicates run() didn't even get to filename generation
             if not self.results: # Check if intermediate results even exist
                 self._update_status("Cannot save results: No analysis data available.")
                 return
             else:
                 self._update_status("Cannot save results: Base filename not generated.")
                 return

        self._update_status(f"Saving results for {base_filename}...")

        # --- Save Segments CSV --- #
        if self.config.get('save_results_dataframe', False):
            try:
                final_segments_data: List[Segment] = []
                if final_analysis_results:
                    final_segments_data = final_analysis_results.final_segments
                elif 'final_segments' in self.results and isinstance(self.results['final_segments'], list):
                    # Fallback to intermediate results (should be List[Segment] now)
                    final_segments_data = self.results['final_segments']
                    if final_segments_data and not isinstance(final_segments_data[0], Segment):
                         # If intermediate is somehow still tuples, handle it (though shouldn't happen)
                         self._update_status("Warning: Intermediate segments were not Segment objects, converting for CSV.")
                         final_segments_data = [Segment(start=s[0], end=s[1]) for s in final_segments_data if isinstance(s, tuple) and len(s) >= 2]

                if final_segments_data:
                    # Create DataFrame from list of Segment objects
                    segments_df = pd.DataFrame([
                        {'start': seg.start, 'end': seg.end, 'duration': seg.duration}
                        for seg in final_segments_data
                    ])
                    csv_path = os.path.join(self.run_output_dir, f"{base_filename}_segments.csv")
                    segments_df.to_csv(csv_path, index_label='segment_index', float_format='%.3f')
                    self._update_status(f"Saved segments CSV: {csv_path}")
                else:
                     self._update_status("No final segments found to save in CSV.")

            except Exception as e:
                self._update_status(f"Error saving segments CSV: {e}")
                import traceback; print(traceback.format_exc())

        # --- Save Full Results JSON --- #
        if self.config.get('save_results_json', False):
            try:
                serializable_data = None
                if final_analysis_results:
                    import dataclasses # Ensure import
                    serializable_data = dataclasses.asdict(final_analysis_results)
                else:
                    # If no final object, don't save JSON or save minimal info?
                    self._update_status("Skipping JSON save: Final results object not available.")
                    # Or potentially build a minimal dict from intermediates?
                    # serializable_data = { ... construct manually ... }

                if serializable_data:
                    json_path = os.path.join(self.run_output_dir, f"{base_filename}_results.json")
                    with open(json_path, 'w', encoding='utf-8') as f: # Add encoding
                        json.dump(serializable_data, f, indent=4, default=str) # Use default=str for any tricky types
                    self._update_status(f"Saved results JSON: {json_path}")
            except Exception as e:
                self._update_status(f"Error saving results JSON: {e}")
                import traceback; print(traceback.format_exc())

    def request_stop(self):
        """Sets a flag to indicate that the pipeline should stop gracefully."""
        self._update_status("Stop requested. Pipeline will halt after the current step.")
        self._stop_requested = True
        # You might need more sophisticated logic if steps involve external processes
        # or long-running loops that need explicit interruption. 