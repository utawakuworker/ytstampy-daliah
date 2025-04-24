import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any, Union

class DetectionEngine(ABC):
    """Abstract base class for singing detection engines."""
    
    @abstractmethod
    def detect_singing(self, 
                      y: np.ndarray, 
                      sr: int, 
                      features_df: pd.DataFrame, 
                      singing_ref: Tuple[float, float], 
                      non_singing_ref: Tuple[float, float],
                      **params) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
        """
        Detect singing segments in audio.
        
        Args:
            y: Audio signal
            sr: Sample rate
            features_df: DataFrame with extracted features
            singing_ref: (start, end) of singing reference
            non_singing_ref: (start, end) of non-singing reference
            **params: Additional parameters for detection
            
        Returns:
            Tuple containing:
            - List of detected segments as (start, end) tuples
            - Updated DataFrame with detection results
        """
        pass


class HMMDetectionEngine(DetectionEngine):
    """
    Hidden Markov Model based singing detection.
    """
    
    def __init__(self):
        """Initialize the HMM detection engine."""
        self.model = None
        self.feature_cols = None
        self.singing_state = None
    
    def detect_singing(self, 
                      y: np.ndarray, 
                      sr: int, 
                      features_df: pd.DataFrame, 
                      singing_ref: Tuple[float, float], 
                      non_singing_ref: Tuple[float, float],
                      **params) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
        """
        Detect singing using HMM.
        
        Args:
            y: Audio signal
            sr: Sample rate
            features_df: DataFrame with extracted features
            singing_ref: (start, end) of singing reference
            non_singing_ref: (start, end) of non-singing reference
            **params: Additional parameters including:
                - threshold: Detection threshold
                - min_duration: Minimum segment duration
                
        Returns:
            Tuple containing:
            - List of detected segments as (start, end) tuples
            - Updated DataFrame with detection results
        """
        # Default parameters
        threshold = params.get('threshold', 0.6)
        min_duration = params.get('min_duration', 2.0)
        min_gap = params.get('min_gap', 1.0)
        verbose = params.get('verbose', True)
        
        if verbose:
            print("Performing HMM-based singing detection...")
        
        # Check if features_df is valid
        if features_df is None or len(features_df) == 0:
            raise ValueError("No features available for HMM detection")
        
        # Select feature columns - check if each exists first
        available_features = features_df.columns
        self.feature_cols = []
        
        for feature in ['harmonic_ratio_mean', 'spectral_contrast_mean', 'spectral_flatness_mean', 
                       'mfcc_std', 'zcr_mean', 'rms_mean']:
            if feature in available_features:
                self.feature_cols.append(feature)
        
        # Add pitch features if available
        if 'pitch_stability_mean' in available_features:
            self.feature_cols.append('pitch_stability_mean')
        
        if len(self.feature_cols) == 0:
            raise ValueError("No suitable features found for HMM detection")
            
        if verbose:
            print(f"Using features: {', '.join(self.feature_cols)}")
        
        # Get reference segment indices
        times = features_df['time'].values
        sing_start, sing_end = singing_ref
        non_sing_start, non_sing_end = non_singing_ref
        
        sing_indices = np.where((times >= sing_start) & (times <= sing_end))[0]
        non_sing_indices = np.where((times >= non_sing_start) & (times <= non_sing_end))[0]
        
        if len(sing_indices) == 0 or len(non_sing_indices) == 0:
            raise ValueError("Reference segments don't align with feature times")
            
        # Prepare features
        X = features_df[self.feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train HMM model
        if verbose:
            print("Training HMM model...")
            
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="diag", n_iter=100, 
                                   verbose=False, init_params="mc")
        self.model.fit(X_scaled)
        
        # Predict states
        states = self.model.predict(X_scaled)
        
        # Determine which state corresponds to singing
        sing_state_votes = states[sing_indices]
        non_sing_state_votes = states[non_sing_indices]
        
        sing_votes = np.bincount(sing_state_votes)
        non_sing_votes = np.bincount(non_sing_state_votes)
        
        # Normalize votes by segment length
        sing_votes = sing_votes / len(sing_indices)
        non_sing_votes = non_sing_votes / len(non_sing_indices)
        
        # The state that appears more in the singing reference is the singing state
        if sing_votes[0] / (non_sing_votes[0] + 0.001) > sing_votes[1] / (non_sing_votes[1] + 0.001):
            self.singing_state = 0
        else:
            self.singing_state = 1
            
        if verbose:
            print(f"Identified singing state as: {self.singing_state}")
        
        # Add state predictions to DataFrame
        features_df = features_df.copy()
        features_df['hmm_state'] = states
        features_df['is_singing'] = (states == self.singing_state).astype(int)
        
        # Find continuous segments
        singing_mask = (states == self.singing_state)
        segments = self._find_continuous_segments(singing_mask, times, min_duration)
        
        if verbose:
            print(f"Found {len(segments)} initial segments")
        
        # Merge very close segments
        if min_gap > 0 and len(segments) > 1:
            segments = self._merge_close_segments(segments, min_gap)
            if verbose:
                print(f"After merging close segments: {len(segments)} segments")
        
        return segments, features_df
    
    def _find_continuous_segments(self, 
                                 mask: np.ndarray, 
                                 times: np.ndarray, 
                                 min_duration: float) -> List[Tuple[float, float]]:
        """
        Find continuous segments in a boolean mask.
        
        Args:
            mask: Boolean mask indicating singing frames
            times: Time points corresponding to frames
            min_duration: Minimum segment duration
            
        Returns:
            List of (start, end) tuples
        """
        segments = []
        
        # Find transitions
        transitions = np.diff(mask.astype(int))
        segment_starts = np.where(transitions == 1)[0]
        segment_ends = np.where(transitions == -1)[0]
        
        # Handle edge cases
        if mask[0]:
            segment_starts = np.insert(segment_starts, 0, 0)
        if mask[-1]:
            segment_ends = np.append(segment_ends, len(mask) - 1)
        
        # Create segments
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            start_time = times[start_idx]
            end_time = times[end_idx]
            duration = end_time - start_time
            
            if duration >= min_duration:
                segments.append((start_time, end_time))
        
        return segments
    
    def _merge_close_segments(self, 
                             segments: List[Tuple[float, float]], 
                             min_gap: float) -> List[Tuple[float, float]]:
        """
        Merge segments that are very close to each other.
        
        Args:
            segments: List of (start, end) tuples
            min_gap: Minimum gap between segments
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x[0])
        
        merged = [segments[0]]
        
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            
            # If this segment starts soon after the previous one ends
            if start - prev_end < min_gap:
                # Merge by extending the previous segment
                merged[-1] = (prev_start, end)
            else:
                # Add as a new segment
                merged.append((start, end))
        
        return merged


class ClusterEnhancedHMMDetectionEngine(HMMDetectionEngine):
    """
    Detection engine that combines clustering with HMM for improved accuracy.
    """
    
    def detect_singing(self, 
                      y: np.ndarray, 
                      sr: int, 
                      features_df: pd.DataFrame, 
                      singing_ref: Tuple[float, float], 
                      non_singing_ref: Tuple[float, float],
                      **params) -> Tuple[List[Tuple[float, float]], pd.DataFrame]:
        """
        Detect singing segments using cluster-enhanced HMM with dimensionality reduction.
        Evaluates segments from both states against reference data instead of 
        choosing a single "singing state".
        """
        # Get parameters
        threshold = params.get('threshold', 0.6)
        min_duration = params.get('min_duration', 2.0)
        min_gap = params.get('min_gap', 1.5)
        visualize = params.get('visualize', False)
        dim_reduction = params.get('dim_reduction', 'pca')  # 'pca', 'umap', or None
        n_components = params.get('n_components', 10)  # Number of dimensions to reduce to
        verbose = params.get('verbose', True)
        focus_on_interludes = params.get('focus_on_interludes', True)  # Focus on interludes rather than preludes
        interlude_threshold = params.get('interlude_threshold', 0.3)  # Threshold for interlude detection
        max_interlude_duration = params.get('max_interlude_duration', 20.0)  # Maximum interlude duration in seconds
        
        if verbose:
            print("Using cluster-enhanced HMM with segment-level evaluation...")
        
        # Extract feature columns
        feature_cols = [col for col in features_df.columns 
                       if col != 'time' and not col.endswith('_cluster')]
        
        # Prepare features
        features = features_df[feature_cols].values
        times = features_df['time'].values
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply dimensionality reduction if requested
        if dim_reduction == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=min(n_components, features_scaled.shape[1]))
            features_reduced = reducer.fit_transform(features_scaled)
            if verbose:
                print(f"Reduced features from {features_scaled.shape[1]} to {features_reduced.shape[1]} dimensions using PCA")
        elif dim_reduction == 'umap' and UMAP_AVAILABLE:
            import umap
            reducer = umap.UMAP(n_components=min(n_components, features_scaled.shape[1]))
            features_reduced = reducer.fit_transform(features_scaled)
            if verbose:
                print(f"Reduced features from {features_scaled.shape[1]} to {features_reduced.shape[1]} dimensions using UMAP")
        else:
            # No reduction or UMAP not available
            features_reduced = features_scaled
            if verbose:
                print(f"Using {features_scaled.shape[1]} original features without reduction")
        
        # Always use 2 clusters for singing/non-singing
        n_clusters = 2
        
        # Get reference indices
        sing_start, sing_end = singing_ref
        non_sing_start, non_sing_end = non_singing_ref
        
        # Find reference frames
        singing_ref_mask = (times >= sing_start) & (times <= sing_end)
        non_singing_ref_mask = (times >= non_sing_start) & (times <= non_sing_end)
        
        singing_ref_indices = np.where(singing_ref_mask)[0]
        non_singing_ref_indices = np.where(non_singing_ref_mask)[0]
        
        # Get average feature vectors for singing and non-singing references
        singing_centroid = np.mean(features_reduced[singing_ref_indices], axis=0).reshape(1, -1)
        non_singing_centroid = np.mean(features_reduced[non_singing_ref_indices], axis=0).reshape(1, -1)
        
        # Stack the centroids for initialization
        initial_centroids = np.vstack([singing_centroid, non_singing_centroid])
        
        # Use KMeans with the predetermined centroids
        kmeans = KMeans(n_clusters=n_clusters, init=initial_centroids, n_init=1)
        cluster_labels = kmeans.fit_predict(features_reduced)
        
        # Add cluster labels to DataFrame
        features_df = features_df.copy()
        features_df['cluster'] = cluster_labels
        
        if verbose:
            print(f"Applied K-means clustering with {n_clusters} clusters")
            
        # Determine which cluster corresponds to singing based on reference overlap
        singing_cluster_votes = np.bincount(cluster_labels[singing_ref_mask], minlength=n_clusters)
        non_singing_cluster_votes = np.bincount(cluster_labels[non_singing_ref_mask], minlength=n_clusters)
        
        # Calculate overlap ratios
        singing_ratio = singing_cluster_votes / np.sum(singing_cluster_votes)
        non_singing_ratio = non_singing_cluster_votes / np.sum(non_singing_cluster_votes)
        
        # The cluster with higher singing ratio is the singing cluster
        singing_scores = singing_ratio / (non_singing_ratio + 0.001)
        singing_cluster = np.argmax(singing_scores)
        
        if verbose:
            print(f"Identified cluster {singing_cluster} as likely singing cluster")
            
        # Use the reduced features for HMM, along with cluster label
        # Add the cluster as a feature to help HMM
        features_for_hmm = np.column_stack([
            features_reduced,
            np.array([1.0 if c == singing_cluster else 0.0 for c in cluster_labels]).reshape(-1, 1)
        ])
        
        # Train HMM with the enhanced features
        self.model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        self.model.fit(features_for_hmm)
        
        # Get state assignments
        states = self.model.predict(features_for_hmm)
        
        # Get posterior probabilities for each state
        posteriors = self.model.predict_proba(features_for_hmm)
        
        # Add state information to DataFrame
        features_df['hmm_state'] = states
        features_df['state0_posterior'] = posteriors[:, 0]
        features_df['state1_posterior'] = posteriors[:, 1]
        
        # Find continuous segments from both states
        segments_state0 = self._find_segments_from_state(states, times, 0, min_duration)
        segments_state1 = self._find_segments_from_state(states, times, 1, min_duration)
        
        # Combine segments from both states
        all_segments = segments_state0 + segments_state1
        
        # Sort segments by start time
        all_segments.sort(key=lambda x: x[0])
        
        if verbose:
            print(f"Found {len(all_segments)} initial segments from both states")
        
        # Calculate similarity to reference segments for each detected segment
        # Using Mahalanobis distance instead of cosine similarity
        evaluated_segments = []
        
        # Calculate covariance matrices for distance calculations
        singing_cov = np.cov(features_reduced[singing_ref_indices].T)
        non_singing_cov = np.cov(features_reduced[non_singing_ref_indices].T)
        
        # Add a small regularization to ensure invertibility
        singing_cov += np.eye(singing_cov.shape[0]) * 1e-6
        non_singing_cov += np.eye(non_singing_cov.shape[0]) * 1e-6
        
        # Inverse covariance matrices
        try:
            singing_cov_inv = np.linalg.inv(singing_cov)
            non_singing_cov_inv = np.linalg.inv(non_singing_cov)
        except np.linalg.LinAlgError:
            # If inversion fails, use pseudoinverse
            singing_cov_inv = np.linalg.pinv(singing_cov)
            non_singing_cov_inv = np.linalg.pinv(non_singing_cov)
        
        for start, end in all_segments:
            # Get indices for this segment
            segment_mask = (times >= start) & (times <= end)
            segment_indices = np.where(segment_mask)[0]
            
            if len(segment_indices) > 0:
                # Get segment features
                segment_features = features_reduced[segment_indices]
                
                # Calculate segment centroid
                segment_centroid = np.mean(segment_features, axis=0)
                
                # Calculate Mahalanobis distance to reference centroids
                try:
                    dist_to_singing = self._mahalanobis_distance(
                        segment_centroid, singing_centroid.flatten(), singing_cov_inv)
                    dist_to_non_singing = self._mahalanobis_distance(
                        segment_centroid, non_singing_centroid.flatten(), non_singing_cov_inv)
                except:
                    # Fall back to Euclidean distance if Mahalanobis fails
                    dist_to_singing = np.linalg.norm(segment_centroid - singing_centroid.flatten())
                    dist_to_non_singing = np.linalg.norm(segment_centroid - non_singing_centroid.flatten())
                
                # Convert distances to probability-like scores (closer = higher probability)
                total_dist = dist_to_singing + dist_to_non_singing
                if total_dist > 0:
                    similarity_score = dist_to_non_singing / total_dist  # Higher when closer to singing
                else:
                    similarity_score = 0.5  # Ambiguous case
                
                # Also check cluster composition in this segment
                cluster_counts = np.bincount(cluster_labels[segment_indices], minlength=n_clusters)
                cluster_probs = cluster_counts / np.sum(cluster_counts)
                cluster_score = cluster_probs[singing_cluster]
                
                # Weighted combination of distance-based similarity and cluster evidence
                singing_probability = 0.6 * similarity_score + 0.4 * cluster_score
                
                # Store segment with its probability
                evaluated_segments.append((start, end, float(singing_probability)))
        
        # Filter segments based on singing probability
        singing_segments = [(start, end) for start, end, prob in evaluated_segments if prob >= threshold]
        
        # Merge consecutive segments with the same label
        if len(singing_segments) > 1:
            merged_segments = []
            current_segment = singing_segments[0]
            
            for next_segment in singing_segments[1:]:
                # If the gap is small enough, merge
                if next_segment[0] - current_segment[1] < min_gap:
                    current_segment = (current_segment[0], next_segment[1])
                else:
                    merged_segments.append(current_segment)
                    current_segment = next_segment
            
            # Add the last segment
            merged_segments.append(current_segment)
            singing_segments = merged_segments
            
            if verbose:
                print(f"After merging consecutive segments: {len(singing_segments)} segments")
        
        # Identify interludes (musical sections between singing segments)
        if focus_on_interludes and len(singing_segments) > 1:
            final_segments = self._identify_song_structure_and_interludes(
                singing_segments,
                times,
                features_reduced,
                states,
                posteriors,
                cluster_labels,
                singing_cluster,
                interlude_threshold,
                max_interlude_duration,
                verbose
            )
        else:
            final_segments = singing_segments
        
        # Update DataFrame with segment information
        features_df['is_singing'] = 0
        for start, end in final_segments:
            segment_mask = (times >= start) & (times <= end)
            features_df.loc[segment_mask, 'is_singing'] = 1
            
        # Store segment probabilities for visualization
        segment_probs = np.zeros_like(times)
        for start, end, prob in evaluated_segments:
            segment_mask = (times >= start) & (times <= end)
            segment_probs[segment_mask] = prob
            
        features_df['singing_probability'] = segment_probs
        
        # Visualize if requested
        if visualize:
            self._visualize_cluster_enhanced_hmm(
                times, cluster_labels, states, None,
                final_segments, features_df,
                singing_ref=singing_ref,
                non_singing_ref=non_singing_ref,
                evaluated_segments=evaluated_segments
            )
        
        if verbose:
            print(f"Detected {len(final_segments)} singing segments after evaluation")
            
        return final_segments, features_df
        
    def _mahalanobis_distance(self, x, mean, cov_inv):
        """
        Calculate Mahalanobis distance between a point and a distribution.
        
        Args:
            x: Point vector
            mean: Distribution mean vector
            cov_inv: Inverse of the covariance matrix
            
        Returns:
            Mahalanobis distance (float)
        """
        diff = x - mean
        return np.sqrt(diff.dot(cov_inv).dot(diff.T))
        
    def _identify_song_structure_and_interludes(self,
                                              singing_segments,
                                              times,
                                              features_reduced,
                                              states,
                                              posteriors,
                                              cluster_labels,
                                              singing_cluster,
                                              interlude_threshold=0.3,
                                              max_interlude_duration=20.0,
                                              verbose=True):
        """
        Identify song structure and interludes between singing segments.
        Focuses on finding patterns in the song structure like verse-chorus patterns.
        
        Args:
            singing_segments: List of identified singing segments
            times: Time array
            features_reduced: Reduced feature matrix
            states: HMM state assignments
            posteriors: HMM posterior probabilities
            cluster_labels: Cluster assignments
            singing_cluster: Which cluster corresponds to singing
            interlude_threshold: Threshold for interlude detection
            max_interlude_duration: Maximum interlude duration
            verbose: Whether to print status
            
        Returns:
            List of segments with interludes incorporated
        """
        if verbose:
            print("Analyzing song structure to identify interludes...")
        
        # If we have fewer than 2 segments, no interludes to find
        if len(singing_segments) < 2:
            return singing_segments
        
        # Get the transition matrix from the trained HMM for analysis
        transition_matrix = self.model.transmat_
        
        # Determine which state is more likely during singing
        state0_in_singing = np.mean([1 if s == 0 else 0 for s in states[cluster_labels == singing_cluster]])
        state1_in_singing = np.mean([1 if s == 1 else 0 for s in states[cluster_labels == singing_cluster]])
        
        singing_state = 0 if state0_in_singing > state1_in_singing else 1
        non_singing_state = 1 - singing_state
        
        # Extract segment features for similarity comparison
        segment_features = []
        for start, end in singing_segments:
            # Get indices for this segment
            segment_mask = (times >= start) & (times <= end)
            segment_indices = np.where(segment_mask)[0]
            
            if len(segment_indices) > 0:
                # Extract key features that represent this segment
                segment_feats = {
                    'duration': end - start,
                    'features': features_reduced[segment_indices],
                    'states': states[segment_indices],
                    'clusters': cluster_labels[segment_indices],
                    'start': start,
                    'end': end,
                    # Pattern features - useful for comparing verse/chorus
                    'state_pattern': self._extract_pattern(states[segment_indices]),
                    'cluster_pattern': self._extract_pattern(cluster_labels[segment_indices])
                }
                segment_features.append(segment_feats)
        
        # Find potential interludes (regions between singing segments)
        potential_interludes = []
        
        for i in range(len(singing_segments) - 1):
            current_end = singing_segments[i][1]
            next_start = singing_segments[i+1][0]
            
            # Potential interlude if gap is meaningful but not too long
            if next_start > current_end and (next_start - current_end) <= max_interlude_duration:
                interlude_mask = (times >= current_end) & (times <= next_start)
                interlude_indices = np.where(interlude_mask)[0]
                
                if len(interlude_indices) > 0:
                    # Calculate various interlude metrics
                    
                    # 1. State transition activity in the interlude
                    state_transitions = np.diff(states[interlude_indices])
                    transition_activity = np.sum(state_transitions != 0) / max(1, len(interlude_indices) - 1)
                    
                    # 2. Cluster behavior in the interlude
                    singing_cluster_ratio = np.mean(cluster_labels[interlude_indices] == singing_cluster)
                    
                    # 3. Similarity between segments on both sides
                    if i < len(segment_features)-1:
                        segment_similarity = self._compare_segment_patterns(
                            segment_features[i], segment_features[i+1])
                    else:
                        segment_similarity = 0.0
                    
                    # Interlude score based on musical structure
                    # Higher when segments on both sides are similar (suggesting verse-verse or chorus-chorus),
                    # indicating a likely instrumental interlude
                    interlude_score = (
                        0.4 * segment_similarity +      # Higher when similar segments detected
                        0.3 * (1 - singing_cluster_ratio) +  # Higher when less singing in the interlude
                        0.3 * transition_activity        # Higher with more state transitions (musical activity)
                    )
                    
                    potential_interludes.append({
                        'start': current_end,
                        'end': next_start,
                        'score': interlude_score,
                        'before_segment': i,
                        'after_segment': i+1
                    })
        
        # Determine which interludes to include based on score and musical structure
        interludes_to_include = []
        
        for interlude in potential_interludes:
            if interlude['score'] >= interlude_threshold:
                interludes_to_include.append(interlude)
                if verbose:
                    print(f"Identified interlude: {interlude['start']:.2f}s - {interlude['end']:.2f}s "
                          f"(score: {interlude['score']:.2f})")
        
        # Create final segments including the identified interludes
        final_segments = []
        i = 0
        
        while i < len(singing_segments):
            current_start, current_end = singing_segments[i]
            final_segments.append((current_start, current_end))
            
            # Check if this segment is followed by an included interlude
            interlude_found = False
            for interlude in interludes_to_include:
                if interlude['before_segment'] == i:
                    # If the next segment is also included, merge them with the interlude
                    next_idx = i + 1
                    if next_idx < len(singing_segments):
                        next_start, next_end = singing_segments[next_idx]
                        # Create one continuous segment including the interlude
                        final_segments[-1] = (current_start, next_end)
                        i += 1  # Skip the next segment since we've merged it
                        interlude_found = True
                        break
            
            i += 1
        
        # Final clean-up: merge any overlapping segments
        if len(final_segments) > 1:
            final_segments = self._merge_intervals_sweepline(final_segments, verbose)
        
        if verbose:
            print(f"Final segments after structure analysis: {len(final_segments)}")
        
        return final_segments
        
    def _merge_intervals_sweepline(self, segments: List[Tuple[float, float]], verbose=True) -> List[Tuple[float, float]]:
        """
        Merge overlapping intervals using a sweep line algorithm.
        
        Args:
            segments: List of (start, end) tuples
            verbose: Whether to print status
            
        Returns:
            List of merged segments
        """
        if not segments:
            return []

        # Step 1: Sort by start time
        segments.sort()

        merged: List[Tuple[float, float]] = []
        current_start, current_end = segments[0]

        for start, end in segments[1:]:
            if start <= current_end:  # Overlapping
                current_end = max(current_end, end)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        merged.append((current_start, current_end))  # Add last segment
        
        if verbose and len(merged) < len(segments):
            print(f"Merged {len(segments)} segments into {len(merged)} segments")

        return merged
        
    def _extract_pattern(self, sequence, max_length=20):
        """
        Extract a representative pattern from a sequence, useful for verse/chorus detection.
        Returns a downsampled version of the sequence that preserves its overall pattern.
        
        Args:
            sequence: Array of state or cluster values
            max_length: Maximum length of pattern representation
            
        Returns:
            Downsampled pattern array
        """
        if len(sequence) <= max_length:
            return sequence
        
        # Downsample to a fixed length while preserving pattern
        indices = np.linspace(0, len(sequence)-1, max_length, dtype=int)
        return sequence[indices]
        
    def _compare_segment_patterns(self, segment1, segment2):
        """
        Compare two segments to determine if they represent the same musical structure (e.g., two verses).
        Uses dynamic time warping for comparing patterns of different lengths.
        
        Args:
            segment1, segment2: Dictionaries containing segment information
            
        Returns:
            Similarity score between 0 and 1
        """
        # Try to determine if these segments are similar structurally (verse-verse or chorus-chorus)
        
        # 1. Compare state patterns using dynamic time warping
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            
            # Use DTW for state patterns
            state_distance, _ = fastdtw(
                segment1['state_pattern'].reshape(-1, 1), 
                segment2['state_pattern'].reshape(-1, 1),
                dist=euclidean
            )
            # Normalize by length
            state_sim = 1.0 / (1.0 + state_distance / max(len(segment1['state_pattern']), len(segment2['state_pattern'])))
            
            # Use DTW for cluster patterns
            cluster_distance, _ = fastdtw(
                segment1['cluster_pattern'].reshape(-1, 1), 
                segment2['cluster_pattern'].reshape(-1, 1),
                dist=euclidean
            )
            # Normalize by length
            cluster_sim = 1.0 / (1.0 + cluster_distance / max(len(segment1['cluster_pattern']), len(segment2['cluster_pattern'])))
            
            # Duration similarity (verses/choruses often have similar duration)
            duration_ratio = min(segment1['duration'], segment2['duration']) / max(segment1['duration'], segment2['duration'])
            
            # Combined similarity
            combined_sim = 0.4 * state_sim + 0.4 * cluster_sim + 0.2 * duration_ratio
            return combined_sim
            
        except ImportError:
            # Fall back to simpler comparison if fastdtw not available
            
            # Compare durations
            duration_ratio = min(segment1['duration'], segment2['duration']) / max(segment1['duration'], segment2['duration'])
            
            # Compare state distributions
            state1_counts = np.bincount(segment1['states'], minlength=2) / len(segment1['states'])
            state2_counts = np.bincount(segment2['states'], minlength=2) / len(segment2['states'])
            state_sim = 1 - np.mean(np.abs(state1_counts - state2_counts))
            
            # Compare cluster distributions
            cluster1_counts = np.bincount(segment1['clusters'], minlength=2) / len(segment1['clusters'])
            cluster2_counts = np.bincount(segment2['clusters'], minlength=2) / len(segment2['clusters'])
            cluster_sim = 1 - np.mean(np.abs(cluster1_counts - cluster2_counts))
            
            # Combined similarity
            return 0.4 * state_sim + 0.4 * cluster_sim + 0.2 * duration_ratio

    def _find_segments_from_state(self, states, times, target_state, min_duration):
        """
        Find continuous segments for a specific state.
        
        Args:
            states: Array of state assignments
            times: Array of time points
            target_state: The state to find segments for
            min_duration: Minimum segment duration
            
        Returns:
            List of (start, end) tuples
        """
        # Create mask for the target state
        mask = (states == target_state)
        
        # Find transitions
        transitions = np.diff(mask.astype(int))
        segment_starts = np.where(transitions == 1)[0]
        segment_ends = np.where(transitions == -1)[0]
        
        # Handle edge cases
        if mask[0]:
            segment_starts = np.insert(segment_starts, 0, 0)
        if mask[-1]:
            segment_ends = np.append(segment_ends, len(mask) - 1)
        
        # Create segments
        segments = []
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            start_time = times[start_idx]
            end_time = times[end_idx]
            duration = end_time - start_time
            
            if duration >= min_duration:
                segments.append((start_time, end_time))
        
        return segments

    def _visualize_cluster_enhanced_hmm(self,
                                      times: np.ndarray,
                                      cluster_labels: np.ndarray,
                                      states: np.ndarray,
                                      singing_state: int,
                                      segments: List[Tuple[float, float]],
                                      features_df: pd.DataFrame,
                                      singing_ref: Tuple[float, float] = None,
                                      non_singing_ref: Tuple[float, float] = None,
                                      evaluated_segments: List[Tuple[float, float, float]] = None) -> None:
        """Visualize the cluster-enhanced HMM results."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Cluster assignments
        plt.subplot(3, 1, 1)
        
        # Use actual cluster values
        unique_clusters = np.unique(cluster_labels)
        scatter = plt.scatter(times, cluster_labels, c=cluster_labels, cmap='viridis', 
                    alpha=0.6, s=5)
        
        # Highlight reference segments if provided
        if singing_ref:
            sing_start, sing_end = singing_ref
            plt.axvspan(sing_start, sing_end, color='green', alpha=0.2, label='Singing Ref')
        
        if non_singing_ref:
            non_sing_start, non_sing_end = non_singing_ref
            plt.axvspan(non_sing_start, non_sing_end, color='red', alpha=0.2, label='Non-Singing Ref')
        
        # Calculate cluster singing probabilities
        if singing_ref and non_singing_ref:
            singing_ref_mask = (times >= singing_ref[0]) & (times <= singing_ref[1])
            non_singing_ref_mask = (times >= non_singing_ref[0]) & (times <= non_singing_ref[1])
            
            # Add cluster singing probabilities as text
            for cluster_id in unique_clusters:
                # Count occurrences in reference segments
                if np.sum(singing_ref_mask) > 0 and np.sum(non_singing_ref_mask) > 0:
                    cluster_in_singing = np.sum(cluster_labels[singing_ref_mask] == cluster_id) / np.sum(singing_ref_mask)
                    cluster_in_non_singing = np.sum(cluster_labels[non_singing_ref_mask] == cluster_id) / np.sum(non_singing_ref_mask)
                    
                    # Display ratio of singing to non-singing
                    ratio = cluster_in_singing / (cluster_in_non_singing + 0.001)
                    
                    # Find a reasonable position for the text
                    center_idx = len(times) // 2
                    if center_idx < len(times):
                        plt.text(times[center_idx], cluster_id, f"{ratio:.2f}", 
                                fontsize=9, ha='center', va='center',
                                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title("Cluster Assignments and Singing Probabilities")
        plt.ylabel("Cluster ID")
        plt.ylim(min(unique_clusters)-0.5, max(unique_clusters)+0.5)
        plt.legend()
        
        # Plot 2: HMM state assignments
        plt.subplot(3, 1, 2)
        
        # Since we don't have a single singing state anymore, just plot the states
        unique_states = np.unique(states)
        plt.scatter(times, states, c=states, cmap='coolwarm', alpha=0.6, s=5)
        
        # Highlight reference segments again
        if singing_ref:
            plt.axvspan(singing_ref[0], singing_ref[1], color='green', alpha=0.2, label='Singing Ref')
        
        if non_singing_ref:
            plt.axvspan(non_singing_ref[0], non_singing_ref[1], color='red', alpha=0.2, label='Non-Singing Ref')
        
        plt.title("HMM State Assignments")
        plt.ylabel("State")
        plt.yticks(unique_states, [f"State {int(s)}" for s in unique_states])
        plt.legend()
        
        # Plot 3: Singing probabilities and detected segments
        plt.subplot(3, 1, 3)
        
        # Plot singing probabilities if available
        if 'singing_probability' in features_df.columns:
            plt.plot(times, features_df['singing_probability'], alpha=0.6, color='gray')
            plt.ylabel('Singing Probability')
        elif evaluated_segments:
            # Construct probability line from evaluated segments
            probs = np.zeros_like(times)
            for start, end, prob in evaluated_segments:
                segment_mask = (times >= start) & (times <= end)
                probs[segment_mask] = prob
            plt.plot(times, probs, alpha=0.6, color='gray')
            plt.ylabel('Singing Probability')
        else:
            # Plot a relevant feature as fallback
            if 'harmonic_ratio_mean' in features_df.columns:
                plt.plot(times, features_df['harmonic_ratio_mean'], alpha=0.6, color='gray')
                plt.ylabel('Harmonic Ratio')
            elif 'rms_mean' in features_df.columns:
                plt.plot(times, features_df['rms_mean'], alpha=0.6, color='gray')
                plt.ylabel('RMS Energy')
            else:
                plt.plot(times, np.zeros_like(times), alpha=0.6, color='gray')
        
        # Highlight reference segments once more
        if singing_ref:
            plt.axvspan(singing_ref[0], singing_ref[1], color='green', alpha=0.2, label='Singing Ref')
        
        if non_singing_ref:
            plt.axvspan(non_singing_ref[0], non_singing_ref[1], color='red', alpha=0.2, label='Non-Singing Ref')
        
        # Highlight detected segments
        for start, end in segments:
            plt.axvspan(start, end, color='blue', alpha=0.2)
            plt.text((start + end) / 2, 0.8, f"{end-start:.1f}s", 
                    ha='center', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title("Detected Singing Segments")
        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()