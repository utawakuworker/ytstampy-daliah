import os
import sys
import json
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, List, Tuple, Callable

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
        self.results = {} # Stores intermediate and final results

        # Ensure output directory exists (use config value)
        self.output_dir = self.config.get('output_dir', './output_analysis_model') # Default if not in config
        os.makedirs(self.output_dir, exist_ok=True)
        # self._update_status(f"Output directory set to: {self.output_dir}") # Initial status update

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

    def _load_audio(self) -> bool:
        """Loads audio using the factory."""
        source = self.config.get('url') or self.config.get('file')
        if not source:
            self._update_status("Error: No audio source (URL or file) specified.")
            return False

        self._update_progress(None, f"Loading audio source: {os.path.basename(str(source))}")
        try:
            # Pass the configured output_dir to the loader
            self.loader = AudioLoaderFactory.create_loader(source, output_dir=self.output_dir)
            y, sr = self.loader.load_audio()
            self.results['y'] = y
            self.results['sr'] = sr
            self.results['audio_path'] = self.loader.file_path
            self.results['total_duration'] = self.loader.duration
            self._update_status(f"Audio loaded: {self.results['total_duration']:.1f}s @ {sr}Hz")
            return True
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
        try:
            # Using 2-second windows as in the original script example
            self.extractor = FeatureExtractorFacade(
                include_pitch=False, # As per original script example
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
             self.results['final_segments'] = []
             return True # Allow pipeline to continue

        if not initial_segments: # Empty list means detection ran but found nothing
             self._update_status("No initial segments to process.")
             self.results['final_segments'] = []
             return True

        self._update_progress(0.6, "Processing and refining segments...")
        try:
            # Create segment processing pipeline (could be configured further)
            self.segment_processor = SegmentProcessingPipeline([
                SegmentRefiner(),
                # SegmentMerger(), # Optional: Add merger if desired
                SegmentFilter()
            ])

            processed_segments = self.segment_processor.process(
                initial_segments,
                self.results['y'], self.results['sr'],
                min_duration=self.config.get('min_segment_duration', 10.0),
                min_gap_for_merge=self.config.get('min_segment_gap', 1.5), # If merger is used
                verbose=False, # Reduce console noise
                frame_df=self.results.get('frame_df_with_states') # Pass for validation if available
            )
            self.results['final_segments'] = processed_segments
            self._update_status(f"Processed segments: {len(processed_segments)} final segments.")
            return True
        except Exception as e:
            self._update_status(f"Error processing segments: {e}")
            import traceback
            print(traceback.format_exc())
            # Fallback: use initial segments if processing fails? Or fail pipeline?
            # self.results['final_segments'] = initial_segments # Option: Fallback
            return False # Option: Fail

    def _identify_songs(self) -> bool:
        """Identifies songs in the final segments using Whisper and Gemini."""
        final_segments = self.results.get('final_segments')
        if not final_segments:
            self._update_status("Skipping song identification: No final segments.")
            self.results['identification_results'] = []
            return True # Not an error, just nothing to identify

        # Check for API key early
        api_key = self.config.get('gemini_api_key') or os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self._update_status("Warning: Gemini API key not found. Skipping song identification.")
            self.results['identification_results'] = []
            return True # Allow pipeline to finish, but without identification

        self._update_progress(0.8, "Identifying songs in segments...")
        try:
            self.identifier = SongIdentifier(
                audio_path=self.results['audio_path'],
                output_dir=self.output_dir, # Use the same output dir
                whisper_model=self.config.get('whisper_model', 'base'),
                gemini_api_key=api_key
            )

            # Filter segments based on min duration for ID
            segments_to_id = [
                seg for seg in final_segments
                if (seg[1] - seg[0]) >= self.config.get('min_duration_for_id', 30.0)
            ]

            if not segments_to_id:
                 self._update_status("No segments long enough for identification.")
                 self.results['identification_results'] = []
                 return True

            self._update_status(f"Attempting identification on {len(segments_to_id)} segments...")

            # Note: identify_songs might take a while. Consider finer-grained progress?
            # The SongIdentifier itself would need progress callbacks for that.
            identification_results = self.identifier.identify_songs(
                segments_to_id,
                min_segment_duration=self.config.get('min_duration_for_id', 30.0),
                verbose=True # Keep identifier verbose for now
            )
            self.results['identification_results'] = identification_results
            identified_count = sum(1 for r in identification_results if r.get('identification', {}).get('title'))
            self._update_status(f"Song identification complete. Identified {identified_count} songs.")
            return True
        except ImportError:
             self._update_status("Error: Whisper/Gemini dependency missing. Cannot identify songs.")
             # Log specific error if possible
             self.results['identification_results'] = []
             return True # Allow pipeline to finish, but without identification
        except Exception as e:
            self._update_status(f"Error during song identification: {e}")
            import traceback
            print(traceback.format_exc())
            self.results['identification_results'] = []
            return False # Treat identification error as pipeline failure? Or just warning?

    # --- Main Execution Method ---

    def run(self) -> Optional[Dict[str, Any]]:
        """Executes the full analysis pipeline step-by-step."""
        self._update_progress(0.0, "Starting Analysis Pipeline...")

        if not self._load_audio():
             self._update_progress(0.0, "Pipeline failed: Could not load audio.")
             return None
        # Progress: ~10%

        if not self._extract_features():
             self._update_progress(0.0, "Pipeline failed: Could not extract features.")
             return None
        # Progress: ~30%

        if not self._detect_segments():
             self._update_progress(0.0, "Pipeline failed: Could not detect segments.")
             return None
        # Progress: ~60%

        if not self._process_segments():
             self._update_progress(0.0, "Pipeline failed: Could not process segments.")
             return None
        # Progress: ~80%

        if not self._identify_songs():
             # Decide if identification failure stops the whole pipeline
             self._update_status("Warning: Song identification failed or was skipped.")
             # Continue anyway to provide detection results
             # return None # Uncomment this line to make ID failure fatal

        # Progress: ~100%
        self._update_progress(1.0, "Pipeline finished.")
        # self._print_summary() # Summary printing should be handled by caller (ViewModel/UI)
        return self.results

    # --- Utility/Result Methods (Called by ViewModel/Caller) ---

    def visualize(self):
        """Generates and shows visualizations using Matplotlib."""
        if not self.config.get('visualize'):
            self._update_status("Visualization disabled in config.")
            return

        self._update_status("Generating visualizations...")
        y = self.results.get('y')
        sr = self.results.get('sr')
        segments = self.results.get('final_segments', [])
        frame_df = self.results.get('frame_df_with_states')

        if y is None or sr is None:
            self._update_status("Cannot visualize: Audio data not available.")
            return

        if not segments:
            self._update_status("No segments to visualize.")
            # Optionally plot just the waveform if desired
            return

        # Define reference segments for plotting
        ref_dur = self.config.get('ref_duration', 2.0)
        sing_start = self.config.get('singing_ref_time', 0)
        nsing_start = self.config.get('non_singing_ref_time', 0)
        total_dur = self.results.get('total_duration', float('inf'))
        reference_segments_vis = {
            'singing': (max(0, sing_start), min(total_dur, sing_start + ref_dur)),
            'non-singing': (max(0, nsing_start), min(total_dur, nsing_start + ref_dur))
        }

        try:
            plot_waveform_with_segments(
                y, sr, segments,
                title="Final Detected Singing Segments",
                reference_segments=reference_segments_vis
            )

            if frame_df is not None and not frame_df.empty:
                 plot_feature_comparison(frame_df, segments)
            else:
                 self._update_status("Skipping feature comparison plot: Feature data not available.")

            plt.show() # Display all plots - This blocks! May need adjustment for GUI.
            self._update_status("Visualization complete.")

        except Exception as e:
            self._update_status(f"Error during visualization: {e}")
            import traceback
            print(traceback.format_exc())


    def save_results(self):
        """Saves the identification results to JSON and/or CSV."""
        output_dir = self.output_dir # Use the initialized output dir
        identification_results = self.results.get('identification_results')

        # Save JSON results
        if self.config.get('save_results_json'):
            if identification_results is None:
                 self._update_status("Skipping JSON save: No identification results available.")
            else:
                json_path = os.path.join(output_dir, self.config.get('results_json_file', 'results.json'))
                self._update_status(f"Saving results to JSON: {json_path}")
                try:
                    output_data = {
                        "source": self.config.get('url') or self.config.get('file'),
                        "total_duration": self.results.get('total_duration'),
                        "analysis_config": {k: v for k, v in self.config.items() if k != 'gemini_api_key'},
                        "detected_segments_count": len(self.results.get('final_segments', [])),
                        "final_segments": self.results.get('final_segments', []), # Include segment times
                        "identified_songs": identification_results
                    }
                    with open(json_path, 'w') as f:
                        json.dump(output_data, f, indent=4)
                    self._update_status(f"Results saved to: {json_path}")
                except Exception as e:
                    self._update_status(f"Error saving JSON results: {e}")

        # Save DataFrame results
        if self.config.get('save_results_dataframe'):
             if identification_results is None:
                  self._update_status("Skipping CSV save: No identification results available.")
             elif not identification_results:
                  self._update_status("No identification results to save to DataFrame.")
             else:
                 df_path = os.path.join(output_dir, self.config.get('results_dataframe_file', 'results.csv'))
                 self._update_status(f"Saving results to DataFrame (CSV): {df_path}")
                 try:
                     df_data = []
                     # Match identified results back to final_segments if needed, or assume order matches
                     # This assumes identify_songs returns results in the same order as input segments_to_id
                     # A more robust approach might involve matching timestamps.
                     for item in identification_results: # This list only contains results for segments >= min_id_duration
                         row = {
                             'segment_start': item['segment'][0],
                             'segment_end': item['segment'][1],
                             'segment_duration': item['duration'],
                             'transcript': item.get('transcript', ''),
                             'identified_title': item.get('identification', {}).get('title'),
                             'identified_artist': item.get('identification', {}).get('artist'),
                             'identification_confidence': item.get('identification', {}).get('confidence'),
                             'identification_explanation': item.get('identification', {}).get('explanation'),
                             'identification_error': item.get('identification', {}).get('error')
                         }
                         df_data.append(row)

                     df = pd.DataFrame(df_data)
                     df.to_csv(df_path, index=False)
                     self._update_status(f"DataFrame saved to: {df_path}")
                 except Exception as e:
                     self._update_status(f"Error saving DataFrame results: {e}") 