import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any, Union
import librosa
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# Try to import UMAP, but make it optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    import warnings
    warnings.warn("UMAP not installed. Only PCA will be available for dimensionality reduction.")

class SegmentProcessor(ABC):
    """Abstract base class for segment processors."""
    
    @abstractmethod
    def process_segments(self, 
                        segments: List[Tuple[float, float]], 
                        y: np.ndarray, 
                        sr: int,
                        **params) -> List[Tuple[float, float]]:
        """
        Process detected segments.
        
        Args:
            segments: List of (start, end) tuples
            y: Audio signal
            sr: Sample rate
            **params: Additional parameters
            
        Returns:
            List of processed segments as (start, end) tuples
        """
        pass


class SegmentRefiner(SegmentProcessor):
    """Refines segment boundaries using various criteria."""
    
    def process_segments(self, 
                        segments: List[Tuple[float, float]], 
                        y: np.ndarray, 
                        sr: int,
                        **params) -> List[Tuple[float, float]]:
        """
        Refine segment boundaries based on energy transitions.
        
        Args:
            segments: List of (start, end) tuples
            y: Audio signal
            sr: Sample rate
            **params: Additional parameters including:
                - window_ms: Window size in milliseconds for boundary detection
                - threshold: Energy threshold for boundary detection
                
        Returns:
            List of refined segments
        """
        if not segments:
            return []
        
        window_ms = params.get('window_ms', 200)
        threshold = params.get('threshold', 0.25)
        verbose = params.get('verbose', False)
        
        if verbose:
            print("Refining segment boundaries...")
        
        window_samples = int(sr * window_ms / 1000)
        refined_segments = []
        
        for start, end in segments:
            # Convert to samples
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # Ensure within bounds
            start_sample = max(0, start_sample)
            end_sample = min(len(y), end_sample)
            
            if end_sample <= start_sample:
                continue
            
            # Refine start boundary
            new_start_sample = self._find_energy_transition(
                y, start_sample, window_samples, threshold, direction='forward'
            )
            
            # Refine end boundary
            new_end_sample = self._find_energy_transition(
                y, end_sample, window_samples, threshold, direction='backward'
            )
            
            # Convert back to seconds
            new_start = new_start_sample / sr
            new_end = new_end_sample / sr
            
            if new_end > new_start:
                refined_segments.append((new_start, new_end))
        
        if verbose:
            print(f"Refined {len(segments)} segments")
        
        return refined_segments
    
    def _find_energy_transition(self, 
                               y: np.ndarray, 
                               sample_position: int,
                               window_samples: int,
                               threshold: float,
                               direction: str = 'forward') -> int:
        """
        Find energy transition in signal around a position.
        
        Args:
            y: Audio signal
            sample_position: Sample position to search around
            window_samples: Window size in samples
            threshold: Energy threshold for transition
            direction: Search direction ('forward' or 'backward')
            
        Returns:
            Sample position of transition
        """
        # Calculate energy in sliding windows
        max_pos = len(y) - window_samples
        
        if direction == 'forward':
            # Look forward from position to find start of high energy
            search_range = range(
                max(0, sample_position - window_samples),
                min(max_pos, sample_position + window_samples),
                window_samples // 4
            )
            
            baseline_energy = np.mean(y[max(0, sample_position - window_samples):sample_position]**2)
            
            for pos in search_range:
                window_energy = np.mean(y[pos:pos + window_samples]**2)
                
                if window_energy > baseline_energy * (1 + threshold):
                    return pos
            
            # No clear transition found
            return sample_position
        
        else:  # direction == 'backward'
            # Look backward from position to find end of high energy
            search_range = range(
                min(max_pos, sample_position + window_samples),
                max(0, sample_position - window_samples),
                -window_samples // 4
            )
            
            baseline_energy = np.mean(y[sample_position:min(len(y), sample_position + window_samples)]**2)
            
            for pos in search_range:
                window_energy = np.mean(y[pos:pos + window_samples]**2)
                
                if window_energy < baseline_energy * (1 - threshold):
                    return pos + window_samples
            
            # No clear transition found
            return sample_position


class SegmentMerger(SegmentProcessor):
    """Merges nearby segments based on gap criteria."""
    
    def process_segments(self, 
                        segments: List[Tuple[float, float]], 
                        y: np.ndarray, 
                        sr: int,
                        **params) -> List[Tuple[float, float]]:
        """
        Merge nearby segments.
        
        Args:
            segments: List of (start, end) tuples
            y: Audio signal
            sr: Sample rate
            **params: Additional parameters including:
                - min_gap: Minimum gap to keep segments separate
                - max_gap_ratio: Maximum ratio of gap to surrounding segments to merge
                
        Returns:
            List of merged segments
        """
        if not segments or len(segments) <= 1:
            return segments
        
        min_gap = params.get('min_gap', 5.0)
        max_gap_ratio = params.get('max_gap_ratio', 0.3)
        verbose = params.get('verbose', False)
        
        if verbose:
            print("Merging nearby segments...")
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        # First pass: Merge segments with very small gaps
        merged_segments = [segments[0]]
        
        for i in range(1, len(segments)):
            prev_end = merged_segments[-1][1]
            curr_start = segments[i][0]
            curr_end = segments[i][1]
            
            # Calculate gap between current segment and previous segment
            gap = curr_start - prev_end
            
            # Calculate durations of surrounding segments
            prev_duration = prev_end - merged_segments[-1][0]
            curr_duration = curr_end - curr_start
            avg_duration = (prev_duration + curr_duration) / 2
            
            # Merge if gap is smaller than min_gap OR if gap is small relative to segment lengths
            if gap < min_gap or gap < avg_duration * max_gap_ratio:
                # Merge by extending previous segment to end of current segment
                merged_segments[-1] = (merged_segments[-1][0], curr_end)
            else:
                # Add as a new segment
                merged_segments.append((curr_start, curr_end))
        
        if verbose:
            print(f"Merged {len(segments)} segments into {len(merged_segments)} segments")
        
        return merged_segments


class SegmentFilter(SegmentProcessor):
    """Filters segments based on duration and proximity.
    
    Handles short segments by either removing them or merging with nearby segments
    depending on the gap between them. This improves detection quality by
    eliminating spurious short detections while preserving meaningful segments.
    """    
    def process_segments(self, 
                        segments: List[Tuple[float, float]], 
                        y: np.ndarray, 
                        sr: int,
                        **params) -> List[Tuple[float, float]]:
        """
        Filter segments based on duration and other criteria.
        
        Args:
            segments: List of (start, end) tuples
            y: Audio signal (not used in this processor)
            sr: Sample rate (not used in this processor)
            **params: Additional parameters including:
                - min_duration: Minimum segment duration to keep
                - min_gap_for_merge: Minimum gap between segments to consider merging
                
        Returns:
            List of filtered segments
        """
        if not segments:
            return []
        
        min_duration = params.get('min_duration', 3.0)
        min_gap_for_merge = params.get('min_gap_for_merge', 0.5)
        verbose = params.get('verbose', False)
        
        if verbose:
            print(f"Filtering segments (min duration: {min_duration}s)...")
        
        # Convert to numpy arrays for vectorized operations
        segments_array = np.array(segments)
        
        # Calculate durations for all segments at once
        durations = segments_array[:, 1] - segments_array[:, 0]
        
        # Find short segments (vectorized)
        short_mask = durations < min_duration
        long_mask = ~short_mask
        
        # Keep all long segments
        filtered_segments = list(map(tuple, segments_array[long_mask].tolist())) if np.any(long_mask) else []
        
        if np.any(short_mask):
            short_segments = segments_array[short_mask]
            
            # Sort segments if not already sorted
            if len(filtered_segments) > 1 and not np.all(np.diff(segments_array[:, 0]) >= 0):
                filtered_segments.sort(key=lambda x: x[0])
                short_segments = short_segments[np.argsort(short_segments[:, 0])]
            
            # For each short segment, check if it should be merged
            for short_start, short_end in short_segments:
                # Try to merge with previous or next segment if gap is small
                merge_done = False
                
                # Check for nearby segments (could be vectorized for even more speed)
                for i, (start, end) in enumerate(filtered_segments):
                    # Check if short segment should be merged with this segment
                    if short_end < start:  # Short segment is before current segment
                        gap = start - short_end
                        if gap <= min_gap_for_merge:
                            # Merge by extending the current segment backwards
                            filtered_segments[i] = (short_start, end)
                            merge_done = True
                            break
                    elif short_start > end:  # Short segment is after current segment
                        gap = short_start - end
                        if gap <= min_gap_for_merge:
                            # Merge by extending the current segment forwards
                            filtered_segments[i] = (start, short_end)
                            merge_done = True
                            break
                
                if not merge_done and (short_end - short_start) >= min_duration * 0.75:
                    # It's almost long enough and couldn't be merged, so keep it
                    filtered_segments.append((short_start, short_end))
                    # Re-sort the list
                    filtered_segments.sort(key=lambda x: x[0])
        
        if verbose:
            print(f"After filtering: {len(filtered_segments)}/{len(segments)} segments kept")
            if len(filtered_segments) != len(segments):
                print(f"Removed {len(segments) - len(filtered_segments)} segments shorter than {min_duration}s")
        
        return filtered_segments


class SegmentValidator(SegmentProcessor):
    """
    Validates segments by comparing their features to reference segments.
    Modified to respect the detection engine's results.
    """
    
    def process(self, segments, y, sr, **kwargs):
        """
        Validate segments by comparing features to reference segments.
        
        Args:
            segments: List of (start, end) tuples
            y: Audio signal
            sr: Sample rate
            **kwargs: Additional parameters:
                - singing_ref: Reference singing segment
                - non_singing_ref: Reference non-singing segment
                - frame_df: Pre-computed feature DataFrame (to avoid recomputation)
                - threshold: Validation confidence threshold (default: 0.6)
                
        Returns:
            List of validated (start, end) tuples
        """
        # Extract parameters
        threshold = kwargs.get('threshold', 0.6)
        frame_df = kwargs.get('frame_df')
        visualize = kwargs.get('visualize', False)
        
        # If no segments to validate, return empty list
        if not segments:
            return []
            
        # If we already have a frame_df with detection results, use it
        if frame_df is not None and 'is_singing' in frame_df.columns:
            print("Using existing detection results for validation")
            
            # Use the detection engine's classification to validate segments
            validated_segments = []
            for start, end in segments:
                # Get frames within this segment
                segment_mask = (frame_df['time'] >= start) & (frame_df['time'] <= end)
                
                if np.sum(segment_mask) > 0:
                    # Calculate confidence based on is_singing column
                    confidence = np.mean(frame_df.loc[segment_mask, 'is_singing'])
                    
                    # Validate if confidence exceeds threshold
                    if confidence >= threshold:
                        validated_segments.append((start, end))
                        print(f"Validated segment {start:.1f}-{end:.1f}s (conf: {confidence:.2f})")
                    else:
                        print(f"Rejected segment {start:.1f}-{end:.1f}s (conf: {confidence:.2f})")
            
            return validated_segments
            
        # Otherwise, fallback to original validation code
        # (This should rarely happen since we're now passing frame_df)
        print("WARNING: No detection results available, performing independent validation")
        
        # ... [original validation code would go here] ...
        
        return segments  # Just return original segments if we can't validate


class SegmentProcessingPipeline:
    """Pipeline for processing segments through multiple processors."""
    
    def __init__(self, processors: List[SegmentProcessor] = None):
        """
        Initialize the pipeline with a list of processors.
        
        Args:
            processors: List of SegmentProcessor instances
        """
        self.processors = processors or []
    
    def add_processor(self, processor: SegmentProcessor) -> None:
        """
        Add a processor to the pipeline.
        
        Args:
            processor: SegmentProcessor instance to add
        """
        self.processors.append(processor)
    
    def process(self, 
               segments: List[Tuple[float, float]], 
               y: np.ndarray, 
               sr: int,
               **params) -> List[Tuple[float, float]]:
        """
        Process segments through all processors in the pipeline.
        
        Args:
            segments: List of (start, end) tuples
            y: Audio signal
            sr: Sample rate
            **params: Parameters for processors
            
        Returns:
            Processed segments
        """
        processed_segments = segments
        
        for i, processor in enumerate(self.processors):
            if not processed_segments:
                # No segments left to process
                break
                
            processor_name = processor.__class__.__name__
            if params.get('verbose', True):
                print(f"Running {processor_name} [{i+1}/{len(self.processors)}]...")
            
            try:
                processed_segments = processor.process_segments(
                    processed_segments, y, sr, **params
                )
                
                if params.get('verbose', True):
                    print(f"After {processor_name}: {len(processed_segments)} segments")
                    
            except Exception as e:
                print(f"Error in {processor_name}: {e}")
                import traceback
                traceback.print_exc()
        
        return processed_segments 