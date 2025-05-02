import json
import os
from typing import Any, Dict

import pandas as pd
import numpy as np
import dataclasses

from model.data_models import AnalysisResults, Segment
from singing_detection.audio.feature_extraction import FeatureExtractorFacade
from singing_detection.audio.loader import AudioLoaderFactory
from singing_detection.detection.detection_pipeline import \
    SingingDetectionPipeline
from singing_detection.identification.song_identifier import SongIdentifier
from singing_detection.segments.segment_processor import (SegmentFilter,
                                                          SegmentRefiner)

# Helper function to format seconds into hh:mm:ss.ms
# Copied from app_view.py for use in SaveResultsStep
def format_seconds_to_hms(total_seconds: float) -> str:
    if total_seconds is None or total_seconds < 0:
        return "0:00.000"
    try:
        td = pd.Timedelta(seconds=total_seconds)
        total_secs_int = int(td.total_seconds())
        milliseconds = int((td.total_seconds() - total_secs_int) * 1000)
        hours, remainder = divmod(total_secs_int, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}:{minutes:02}:{seconds:02}.{milliseconds:03}"
        else:
            return f"{minutes:02}:{seconds:02}.{milliseconds:03}"
    except Exception:
        return "0:00.000"

class PipelineStep:
    def run(self, context: Dict[str, Any]) -> bool:
        raise NotImplementedError

class AudioLoaderStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        source = self.config.get('url') or self.config.get('file')
        if not source:
            context['error'] = "No audio source (URL or file) specified."
            return False
        loader = AudioLoaderFactory.create_loader(source, output_dir=self.config.get('output_dir', './output_analysis_model'))
        y, sr = loader.load_audio()
        context['y'] = y
        context['sr'] = sr
        context['audio_path'] = loader.file_path
        context['total_duration'] = loader.duration
        return True

class FeatureExtractionStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        enable_hpss_flag = self.config.get('enable_hpss', True)
        analysis_window = self.config.get('analysis_window_seconds', 2.0)
        print(f"[Feature Extraction] Using analysis window size: {analysis_window}s")
        extractor = FeatureExtractorFacade(
            include_pitch=False,
            enable_hpss=enable_hpss_flag,
            window_size_seconds=analysis_window
        )
        feature_df = extractor.extract_all_features(
            context['y'], context['sr']
        )
        context['feature_df'] = feature_df
        if feature_df.empty:
            context['error'] = "No features extracted."
            return False
        return True

class SegmentDetectionStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        if 'feature_df' not in context or context['feature_df'].empty:
            context['initial_segments'] = []
            context['frame_df_with_states'] = None
            return True
        feature_df = context['feature_df']
        ref_dur = self.config.get('analysis_window_seconds', 2.0)
        print(f"[Segment Detection] Using reference duration: {ref_dur}s")
        sing_start = self.config.get('singing_ref_time', 0)
        nsing_start = self.config.get('non_singing_ref_time', 0)
        total_dur = context.get('total_duration', float('inf'))
        singing_ref = (max(0, sing_start), min(total_dur, sing_start + ref_dur))
        non_singing_ref = (max(0, nsing_start), min(total_dur, nsing_start + ref_dur))

        # --- Reference-based energy normalization ---
        singing_mask = (feature_df['time'] >= singing_ref[0]) & (feature_df['time'] <= singing_ref[1])
        non_singing_mask = (feature_df['time'] >= non_singing_ref[0]) & (feature_df['time'] <= non_singing_ref[1])
        singing_energy = feature_df.loc[singing_mask, 'rms_mean'].mean()
        non_singing_energy = feature_df.loc[non_singing_mask, 'rms_mean'].mean()
        normalized_energy = (feature_df['rms_mean'] - non_singing_energy) / (singing_energy - non_singing_energy + 1e-8)
        normalized_energy = normalized_energy.clip(0, 1)

        params = {
            'threshold': self.config.get('hmm_threshold', 0.55),
            'min_duration': self.config.get('min_segment_duration', 10.0),
            'min_gap': self.config.get('min_segment_gap', 1.5),
            'dim_reduction': self.config.get('dim_reduction', 'pca'),
            'n_components': self.config.get('n_components', 4),
            'verbose': False
        }
        segments, results = SingingDetectionPipeline.run(
            feature_df,
            singing_ref, non_singing_ref, params
        )

        # --- HMM-based Viterbi segmentation ---
        posteriors = results.get('posteriors')
        states = results.get('states')
        # Get the cluster identified by the pipeline (might be incorrect)
        reported_singing_cluster = results.get('singing_cluster', 1) 
        
        if states is not None:
            # --- Verification: Determine actual singing cluster using references ---
            times = feature_df['time'].values
            states_arr = np.array(states)
            
            singing_ref_mask = (times >= singing_ref[0]) & (times <= singing_ref[1])
            # non_singing_ref_mask = (times >= non_singing_ref[0]) & (times <= non_singing_ref[1])

            if np.any(singing_ref_mask):
                # Find the most frequent state during the singing reference period
                states_in_singing_ref = states_arr[singing_ref_mask]
                unique_states, counts = np.unique(states_in_singing_ref, return_counts=True)
                actual_singing_cluster = unique_states[np.argmax(counts)]
                print(f"[Verification] Pipeline reported singing cluster: {reported_singing_cluster}")
                print(f"[Verification] Actual singing cluster based on reference period: {actual_singing_cluster}")
                if actual_singing_cluster != reported_singing_cluster:
                    print(f"[Verification] Overriding pipeline cluster with reference-based cluster.")
            else:
                print("[Verification] Warning: Singing reference period has no corresponding frames. Using pipeline default.")
                actual_singing_cluster = reported_singing_cluster
            # --- End Verification ---
            
            singing_segments_viterbi = []
            non_singing_segments_viterbi = [] # Also track non-singing for potential analysis
            in_segment = False
            current_segment_state = -1 # Initialize with an invalid state

            # --- Use actual_singing_cluster for segment extraction ---
            for i, state in enumerate(states):
                # Use the verified singing cluster index here
                is_singing_state = (state == actual_singing_cluster) 

                if not in_segment and is_singing_state:
                    # Start of a singing segment
                    start_idx = i
                    in_segment = True
                    # Use the verified singing cluster index here
                    current_segment_state = actual_singing_cluster 
                elif not in_segment and not is_singing_state:
                    # Start of a non-singing segment
                    start_idx = i
                    in_segment = True
                    current_segment_state = state # Store the actual non-singing state
                elif in_segment and is_singing_state and current_segment_state != actual_singing_cluster:
                    # Transition from non-singing to singing
                    end_idx = i - 1
                    # Check if indices are valid before accessing time
                    if start_idx >= 0 and end_idx >= start_idx:
                        start_time = feature_df['time'].iloc[start_idx]
                        end_time = feature_df['time'].iloc[end_idx]
                        if end_time - start_time >= params['min_duration']:
                            non_singing_segments_viterbi.append(Segment(start=start_time, end=end_time))
                    # Start the new singing segment
                    start_idx = i
                     # Use the verified singing cluster index here
                    current_segment_state = actual_singing_cluster
                elif in_segment and not is_singing_state and current_segment_state == actual_singing_cluster:
                     # Transition from singing to non-singing
                    end_idx = i - 1
                     # Check if indices are valid before accessing time
                    if start_idx >= 0 and end_idx >= start_idx:
                        start_time = feature_df['time'].iloc[start_idx]
                        end_time = feature_df['time'].iloc[end_idx]
                        if end_time - start_time >= params['min_duration']:
                            singing_segments_viterbi.append(Segment(start=start_time, end=end_time))
                     # Start the new non-singing segment
                    start_idx = i
                    current_segment_state = state

            # Handle the segment at the end of the audio
            if in_segment and start_idx >= 0:
                end_idx = len(states) - 1
                start_time = feature_df['time'].iloc[start_idx]
                end_time = feature_df['time'].iloc[end_idx]
                segment_duration = end_time - start_time
                if segment_duration >= params['min_duration']:
                    segment = Segment(start=start_time, end=end_time)
                     # Use the verified singing cluster index here
                    if current_segment_state == actual_singing_cluster:
                        singing_segments_viterbi.append(segment)
                    else:
                        non_singing_segments_viterbi.append(segment)

            context['initial_segments'] = singing_segments_viterbi # Store Segment objects directly
            context['non_singing_segments'] = non_singing_segments_viterbi  # For analysis if needed
            
            # Store the actual singing cluster used
            results['actual_singing_cluster'] = actual_singing_cluster 
            context['frame_df_with_states'] = results # Keep the raw results
            
            # Extract posteriors for the actual singing cluster
            if posteriors is not None and posteriors.shape[1] > actual_singing_cluster: 
                P_hmm_singing = posteriors[:, actual_singing_cluster]
                results['hmm_singing_prob'] = P_hmm_singing.tolist()
            # Remove fallback? Or keep it? If verification fails, actual_singing_cluster might be default 1
            # elif posteriors is not None and posteriors.shape[1] == 2: 
            #      P_hmm_singing = posteriors[:, 1]
            #      results['hmm_singing_prob'] = P_hmm_singing.tolist()

        else:
            # If no states (e.g., HMM failed), use the initial segments if any
            initial_segments_tuples = segments # Assuming 'segments' are tuples (start, end)
            context['initial_segments'] = [Segment(start=s[0], end=s[1]) for s in initial_segments_tuples]
            context['non_singing_segments'] = []
            context['frame_df_with_states'] = results
        return True

class SegmentGroupingStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        # Retrieve and normalize initial segments to Segment dataclass
        raw_segments = context.get('initial_segments', [])
        segments: list[Segment] = []
        
        # Convert all segment formats to Segment class
        for item in raw_segments:
            if isinstance(item, Segment):
                segments.append(item)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                segments.append(Segment(start=item[0], end=item[1]))
        
        # If no valid segments, exit early
        feature_df = context.get('feature_df')
        if not segments or feature_df is None:
            context['final_segments'] = []
            return True

        print(f"[SegmentGrouping] Processing {len(segments)} initial segments")
        
        # Sort segments by start time for consistent processing
        segments.sort(key=lambda s: s.start)
        
        # Get configuration for temporal merging
        max_merge_gap = self.config.get('max_merge_gap', 45.0)  # Use a larger default gap
        min_duration = self.config.get('min_segment_duration', 5.0)
        
        # Log segment details
        print(f"[DIAGNOSTIC] Segment details before merging:")
        for i, segment in enumerate(segments):
            print(f"[DIAGNOSTIC]   {i}: {segment.start:.2f}-{segment.end:.2f} (duration: {segment.duration:.2f}s)")
        
        # Directly merge segments based on temporal proximity
        final_segments = self._merge_segments_by_time(segments, min_duration, max_merge_gap)
        
        print(f"[SegmentGrouping] Grouped {len(segments)} initial segments into {len(final_segments)} final segments")
        
        # Store the results in the context
        context['final_segments'] = final_segments
        return True
    
    def _merge_segments_by_time(self, segments: list[Segment], min_duration: float, max_merge_gap: float) -> list[Segment]:
        """
        Merge segments based on temporal proximity.
        
        Args:
            segments: List of segments sorted by start time
            min_duration: Minimum duration for a merged segment
            max_merge_gap: Maximum gap between segments to merge
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        print(f"[DIAGNOSTIC] Merging {len(segments)} segments with max_merge_gap={max_merge_gap}s")
        
        # Begin with the first segment
        merged_segments = []
        current_cluster = [segments[0]]
        
        # Process remaining segments
        for i in range(1, len(segments)):
            current_segment = segments[i]
            prev_segment = current_cluster[-1]
            
            # Calculate the gap between segments
            gap = current_segment.start - prev_segment.end
            
            # If current segment is close enough to the previous one, add to current cluster
            if gap <= max_merge_gap:
                print(f"[DIAGNOSTIC] Merging: Segment gap is {gap:.2f}s ≤ {max_merge_gap}s, adding to cluster")
                current_cluster.append(current_segment)
            else:
                # Create a merged segment from the current cluster
                print(f"[DIAGNOSTIC] Not merging: Segment gap is {gap:.2f}s > {max_merge_gap}s, starting new cluster")
                # Merge all segments in the current cluster
                start = current_cluster[0].start
                end = max(seg.end for seg in current_cluster)
                
                # Log the segments in this cluster
                print(f"[DIAGNOSTIC] Temporal cluster contains {len(current_cluster)} segments from {start:.2f}s to {end:.2f}s")
                if len(current_cluster) > 1:
                    print(f"[DIAGNOSTIC]   Merging segments with start times: {[f'{s.start:.2f}' for s in current_cluster]}")
                
                if end - start >= min_duration:
                    merged_segments.append(Segment(start=start, end=end))
                    print(f"[SegmentGrouping] Created merged segment {start:.2f}-{end:.2f} from {len(current_cluster)} segments")
                else:
                    print(f"[SegmentGrouping] Skipped merged segment {start:.2f}-{end:.2f} (too short: {end-start:.2f}s)")
                
                # Start a new cluster with the current segment
                current_cluster = [current_segment]
        
        # Don't forget the last cluster
        if current_cluster:
            start = current_cluster[0].start
            end = max(seg.end for seg in current_cluster)
            
            print(f"[DIAGNOSTIC] Final temporal cluster contains {len(current_cluster)} segments from {start:.2f}s to {end:.2f}s")
            if len(current_cluster) > 1:
                print(f"[DIAGNOSTIC]   Merging segments with start times: {[f'{s.start:.2f}' for s in current_cluster]}")
            
            if end - start >= min_duration:
                merged_segments.append(Segment(start=start, end=end))
                print(f"[SegmentGrouping] Created merged segment {start:.2f}-{end:.2f} from {len(current_cluster)} segments")
            else:
                print(f"[SegmentGrouping] Skipped merged segment {start:.2f}-{end:.2f} (too short: {end-start:.2f}s)")
        
        return merged_segments

class SegmentProcessingStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        # Re-enabled: This step refines/filters segments after grouping
        # Takes grouped_segments as input
        grouped_segments = context.get('final_segments')
        if not grouped_segments:
            context['final_segments'] = []
            print("[SegmentProcessing] No segments found after grouping step. Skipping processing.")
            return True
        print(f"[SegmentProcessing] Starting processing on {len(grouped_segments)} segments from grouping step.")

        # Convert Segment objects to list of (start, end) tuples for processors
        segment_tuples = [(seg.start, seg.end) for seg in grouped_segments]
        # Initialize refiner and filter
        segment_refiner = SegmentRefiner()
        segment_filter = SegmentFilter()

        try:
            refined_tuples = segment_refiner.process_segments(
                segments=segment_tuples,
                y=context['y'],
                sr=context['sr']
            )
            print(f"[SegmentProcessing] Refined {len(segment_tuples)} segments into {len(refined_tuples)} segments.")
        except Exception as e:
            print(f"[SegmentProcessing] Error during segment refinement: {e}. Using segments before refinement.")
            refined_tuples = segment_tuples

        # Convert back to Segment objects
        refined_segments = [Segment(start=s, end=e) for (s, e) in refined_tuples]

        # Filter segments (e.g., remove short ones, merge close ones based on min_gap)
        try:
            filtered_tuples = segment_filter.process_segments(
                segments=[(seg.start, seg.end) for seg in refined_segments],
                y=context['y'],
                sr=context['sr'],
                min_duration=self.config.get('min_segment_duration', 5.0),
                min_gap_for_merge=self.config.get('min_segment_gap', 1.5),
                verbose=False
            )
            print(f"[SegmentProcessing] Filtered {len(refined_segments)} segments into {len(filtered_tuples)} segments.")
        except Exception as e:
            print(f"[SegmentProcessing] Error during segment filtering/merging: {e}. Using segments before filtering.")
            filtered_tuples = [(seg.start, seg.end) for seg in refined_segments]

        # Convert back to Segment objects
        filtered_segments = [Segment(start=s, end=e) for (s, e) in filtered_tuples]

        context['final_segments'] = filtered_segments
        return True

class SongIdentificationStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        final_segments = context.get('final_segments')
        audio_path = context.get('audio_path')
        if not final_segments:
            context['identification_results'] = []
            return True
        if not audio_path:
            context['error'] = "Cannot identify songs, audio path is missing."
            context['identification_results'] = []
            return False
        api_key = self.config.get('gemini_api_key')
        if not api_key:
            context['identification_results'] = []
            return True
        identifier = SongIdentifier(
            audio_path=audio_path,
            output_dir=self.config.get('output_dir', './output_analysis_model'),
            whisper_model=self.config.get('whisper_model', 'base'),
            gemini_api_key=api_key,
            ffmpeg_path=self.config.get('ffmpeg_path', '')
        )
        identification_results = identifier.identify_songs(
            segments=final_segments,
            min_segment_duration=self.config.get('min_duration_for_id', 30.0),
            max_segment_duration=self.config.get('max_duration_for_id', 60.0),
            verbose=False
        )
        context['identification_results'] = identification_results
        return True

# --- Visualization utility for Viterbi path vs. audio ---
def plot_viterbi_vs_audio(y, sr, times, states, singing_cluster, 
                          singing_viterbi_segments=None, 
                          non_singing_viterbi_segments=None, 
                          posteriors=None, save_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    plt.figure(figsize=(15, 6))
    
    # Plot waveform
    t_audio = np.arange(len(y)) / sr
    plt.plot(t_audio, y / np.max(np.abs(y)), color='gray', alpha=0.5, label='Waveform')
    
    # Plot Viterbi path (binary: singing vs non-singing)
    is_singing_state = (np.array(states) == singing_cluster).astype(int)
    plt.step(times, is_singing_state, where='post', color='blue', alpha=0.6, label=f'Viterbi State (=={singing_cluster}?)')
    
    # Optional: plot HMM singing posterior
    if posteriors is not None:
        plt.plot(times, posteriors, color='orange', alpha=0.7, label='HMM Singing Posterior')
        
    # Optional: plot Viterbi-derived singing segments as shaded regions
    if singing_viterbi_segments is not None:
        for i, seg in enumerate(singing_viterbi_segments):
            plt.axvspan(seg.start, seg.end, color='green', alpha=0.2, label='Viterbi Singing Seg' if i == 0 else "")

    # Optional: plot Viterbi-derived non-singing segments as shaded regions
    if non_singing_viterbi_segments is not None:
        for i, seg in enumerate(non_singing_viterbi_segments):
            plt.axvspan(seg.start, seg.end, color='red', alpha=0.15, label='Viterbi Non-Singing Seg' if i == 0 else "")

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude / State / Probability')
    plt.title('Viterbi Path & Segments vs. Audio')
    plt.legend()
    plt.ylim(-1.05, 1.05) # Ensure waveform and binary state/probability fit well
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"[Viterbi Visualization] Attempting to save to {save_path}")
        try:
            plt.savefig(save_path)
            print(f"[Viterbi Visualization] Saved to {save_path}")
        except Exception as e:
            print(f"[Viterbi Visualization] Error saving plot: {e}")
        finally:
             plt.close() # Close the figure regardless of saving success
    else:
        plt.show()
        plt.close() # Close the figure after showing

class VisualizationStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        y = context.get('y')
        sr = context.get('sr')
        final_segments = context.get('final_segments') # Segments after grouping/processing
        initial_segments = context.get('initial_segments') # Segments directly from Viterbi (singing)
        non_singing_segments = context.get('non_singing_segments') # Segments directly from Viterbi (non-singing)
        feature_df = context.get('feature_df')
        base_filename = context.get('base_filename', 'analysis')
        output_dir = self.config.get('output_dir', './output_analysis_model')
        
        if y is None or sr is None or feature_df is None:
            print("[Visualization] Missing required base data (y, sr, feature_df) for visualization.")
            return True # Not an error, just nothing to visualize

        # --- Standard Plots (using final segments) ---
        try:
            from singing_detection.visualization.plots import (
                plot_feature_comparison, plot_waveform_with_segments)

            # Waveform plot with FINAL segments
            if final_segments is not None:
                 plot_waveform_with_segments(y, sr, final_segments, title=f"Detected Final Segments - {base_filename}")
            else:
                 print("[Visualization] No final segments to plot on waveform.")


            # Feature comparison plot (can use final_segments or initial_segments depending on desired analysis)
            # Using final_segments for now as it represents the 'result'
            if final_segments is not None:
                plot_feature_comparison(feature_df, final_segments, features=['rms_mean', 'harmonic_ratio_mean', 'singing_probability', 'hmm_state'])
            else:
                 print("[Visualization] No final segments for feature comparison plot.")

        except ImportError as e:
            print(f"[Visualization] ImportError for standard plots: {e}. Skipping standard plots.")
        except Exception as e:
            print(f"[Visualization] Exception during standard plots: {e}")
            # Don't raise, try Viterbi plot next

        # --- Viterbi path vs. audio visualization (using initial Viterbi segments) ---
        try:
            results = context.get('frame_df_with_states')
            print(f"[Viterbi Visualization] Checking data: results: {results is not None}, 'states' in results: {'states' in results if results else False}, 'singing_cluster' in results: {'singing_cluster' in results if results else False}")
            
            if results is not None and 'states' in results and 'singing_cluster' in results:
                times = feature_df['time'].values
                states = results['states']
                singing_cluster = results['singing_cluster']
                posteriors = results.get('hmm_singing_prob') # Already extracted in detection step
                
                save_path = os.path.join(output_dir, f'{base_filename}_viterbi_vs_audio.png')
                
                print(f"[Viterbi Visualization] Plotting with {len(initial_segments or [])} initial singing and {len(non_singing_segments or [])} initial non-singing segments.")
                
                plot_viterbi_vs_audio(
                    y, sr, times, states, singing_cluster, 
                    singing_viterbi_segments=initial_segments, # Pass initial singing segments
                    non_singing_viterbi_segments=non_singing_segments, # Pass initial non-singing segments
                    posteriors=posteriors, 
                    save_path=save_path
                )
            else:
                 print("[Viterbi Visualization] Missing HMM state results ('states', 'singing_cluster') in context. Skipping Viterbi plot.")

        except ImportError as e:
            # This specific import is now inside plot_viterbi_vs_audio
            print(f"[Viterbi Visualization] ImportError: {e}. Skipping Viterbi plot.") 
            pass 
        except Exception as e:
            print(f"[Viterbi Visualization] Exception during Viterbi plot: {e}")
            # Decide if this should be fatal; for now, just report it.
            # raise # Uncomment to make plotting errors fatal

        return True # Return True even if plotting fails, as core analysis might be okay

class SaveResultsStep(PipelineStep):
    def __init__(self, config):
        self.config = config

    def run(self, context: Dict[str, Any]) -> bool:
        output_dir = self.config.get('output_dir', '.')
        os.makedirs(output_dir, exist_ok=True)

        # Get base filename from audio path or default
        audio_path = context.get('audio_path', 'analysis_results')
        base_filename = os.path.splitext(os.path.basename(audio_path))[0]

        saved_files = []
        final_segments = context.get('final_segments', [])
        identification_results = context.get('identification_results', [])

        # --- Save DataFrame CSV --- 
        if self.config.get('save_results_dataframe', False) and final_segments:
            df_data = []
            for i, seg in enumerate(final_segments):
                 df_data.append({
                     'segment_num': i + 1,
                     'start_seconds': seg.start,
                     'end_seconds': seg.end,
                     'duration_seconds': seg.duration
                 })
            segments_df = pd.DataFrame(df_data)
            csv_path = os.path.join(output_dir, f"{base_filename}_segments.csv")
            segments_df.to_csv(csv_path, index=False)
            saved_files.append(csv_path)

        # --- Save JSON --- 
        if self.config.get('save_results_json', False) and (final_segments or identification_results):
             results_dict = dataclasses.asdict(AnalysisResults(
                 total_duration=context.get('total_duration'),
                 final_segments=final_segments,
                 identification_results=identification_results
             ))
             json_path = os.path.join(output_dir, f"{base_filename}_results.json")
             with open(json_path, 'w', encoding='utf-8') as f:
                 json.dump(results_dict, f, indent=4, ensure_ascii=False)
             saved_files.append(json_path)
             
        # --- Save YouTube Comment TXT --- 
        if self.config.get('save_youtube_comment', False) and identification_results:
            comment_lines = []
            for idx, result in enumerate(identification_results):
                seg = result.segment
                timestamp = format_seconds_to_hms(seg.start)
                # Check for error within the nested identification object
                if result.identification and result.identification.error:
                    line = f"{timestamp} Error: {result.identification.error}" # Include 'Error:' prefix for clarity
                # Check for identified song within the nested identification object
                elif result.identification and (result.identification.title or result.identification.artist):
                    # Access title and artist from the nested identification object
                    title = result.identification.title or "Unknown Title"
                    artist = result.identification.artist or "Unknown Artist"
                    line = f"{timestamp} {title} - {artist}"
                else:
                    # Handle case where there's no error but no song either
                    line = f"{timestamp} Unknown Song (No identification attempt or result)" # More specific message
                comment_lines.append(line)
            
            if comment_lines:
                comment_text = "\n".join(comment_lines)
                txt_path = os.path.join(output_dir, f"{base_filename}_comment.txt")
                try:
                    with open(txt_path, 'w', encoding='utf-8') as f:
                        f.write(comment_text)
                    saved_files.append(txt_path)
                except IOError as e:
                     print(f"Error saving comment file: {e}")
                     context['error'] = f"Error saving comment file: {e}"
                     # Optionally decide if this error should stop the pipeline (return False)
                     # For now, just log it and continue.

        context['saved_files'] = saved_files
        return True 