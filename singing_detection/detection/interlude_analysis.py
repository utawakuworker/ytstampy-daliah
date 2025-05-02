import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# Helper function to group features by cluster label
def _get_features_by_cluster(clusters, features):
    """
    Groups feature vectors by their assigned cluster label.
    
    Uses numpy's advanced indexing for efficient grouping.
    """
    features_by_cluster = {}
    if features is None or clusters is None or len(clusters) != len(features):
        # Handle cases with missing or mismatched data
        print(f"Warning: Mismatch or missing data in _get_features_by_cluster. Clusters len: {len(clusters) if clusters is not None else 'None'}, Features len: {len(features) if features is not None else 'None'}")
        return features_by_cluster

    # Ensure features is a NumPy array for efficient indexing
    if not isinstance(features, np.ndarray):
        features = np.array(features)
        
    unique_labels = np.unique(clusters)
    for label in unique_labels:
        label_indices = np.where(clusters == label)[0]
        if len(label_indices) > 0:
            features_by_cluster[label] = features[label_indices]
    return features_by_cluster

class InterludeAnalyzer:
    @staticmethod
    def analyze(singing_segments, times, features_reduced, states, posteriors, cluster_labels, singing_cluster, params):
        """
        Analyze song structure to find interludes between segments that likely belong to the same song.
        
        Args:
            singing_segments: List of (start, end) tuples for singing segments
            times: Array of time points
            features_reduced: Reduced feature vectors
            states: HMM state assignments
            posteriors: HMM state probabilities
            cluster_labels: Cluster assignments
            singing_cluster: Index of the cluster identified as singing
            params: Dictionary of parameters
            
        Returns:
            Final segments after merging across interludes
        """
        interlude_threshold = params.get('interlude_threshold', 0.3)
        max_interlude_duration = params.get('max_interlude_duration', 20.0)
        verbose = params.get('verbose', True)
        
        if verbose:
            print("Analyzing song structure to identify interludes...")
            
        # If we have fewer than 2 segments, no interludes to analyze
        if len(singing_segments) < 2:
            return singing_segments
            
        # Extract features for each segment
        segment_features = InterludeAnalyzer._extract_segment_features(
            singing_segments, times, features_reduced, states, cluster_labels, singing_cluster)
            
        # Find potential interludes between segments
        potential_interludes = InterludeAnalyzer._find_potential_interludes(
            singing_segments, times, states, cluster_labels, singing_cluster, 
            segment_features, max_interlude_duration)
            
        # Filter interludes by score threshold
        interludes_to_include = [
            interlude for interlude in potential_interludes 
            if interlude['score'] >= interlude_threshold
        ]
        
        # Merge segments connected by high-scoring interludes
        final_segments = InterludeAnalyzer._merge_segments_with_interludes(
            singing_segments, interludes_to_include, verbose)
            
        # Handle any overlapping segments after merging
        final_segments = InterludeAnalyzer._merge_overlapping_segments(final_segments, verbose)
        
        if verbose:
            print(f"Final segments after structure analysis: {len(final_segments)}")
            
        return final_segments

    @staticmethod
    def _extract_segment_features(singing_segments, times, features_reduced, states, cluster_labels, singing_cluster):
        """
        Extract and standardize features for each detected singing segment.
        """
        # Ensure main arrays are numpy arrays for consistent operations
        if features_reduced is not None and not isinstance(features_reduced, np.ndarray):
            features_reduced = np.array(features_reduced)
        if states is not None and not isinstance(states, np.ndarray):
             states = np.array(states)
        if cluster_labels is not None and not isinstance(cluster_labels, np.ndarray):
            cluster_labels = np.array(cluster_labels)
            
        segment_features = []
        
        for start, end in singing_segments:
            # Get indices for this segment
            segment_mask = (times >= start) & (times <= end)
            segment_indices = np.where(segment_mask)[0]
            
            if len(segment_indices) > 0:
                # Get features and labels for this segment
                segment_feats_data = features_reduced[segment_indices] if features_reduced is not None else None
                segment_states_data = states[segment_indices] if states is not None else None
                segment_clusters_data = cluster_labels[segment_indices] if cluster_labels is not None else None

                # Store features in a dictionary
                segment_feats = {
                    'duration': end - start,
                    'features': np.array(segment_feats_data) if segment_feats_data is not None else None,
                    'states': np.array(segment_states_data) if segment_states_data is not None else None,
                    'clusters': np.array(segment_clusters_data) if segment_clusters_data is not None else None,
                    'start': start,
                    'end': end
                }
                segment_features.append(segment_feats)
                
        return segment_features

    @staticmethod
    def _find_potential_interludes(singing_segments, times, states, cluster_labels, singing_cluster, segment_features, max_interlude_duration):
        """
        Find potential interludes between segments, scoring them by segment similarity.
        """
        potential_interludes = []
        
        for i in range(len(singing_segments) - 1):
            current_end = singing_segments[i][1]
            next_start = singing_segments[i+1][0]
            
            # Check if there's a gap between segments that could be an interlude
            if next_start > current_end and (next_start - current_end) <= max_interlude_duration:
                # Get features for the interlude
                interlude_mask = (times >= current_end) & (times <= next_start)
                interlude_indices = np.where(interlude_mask)[0]
                
                if len(interlude_indices) > 0:
                    # Calculate state transitions in the interlude
                    state_transitions = np.diff(states[interlude_indices])
                    transition_activity = np.sum(state_transitions != 0) / max(1, len(interlude_indices) - 1)
                    
                    # Calculate ratio of singing cluster in the interlude
                    singing_cluster_ratio = np.mean(cluster_labels[interlude_indices] == singing_cluster)
                    
                    # Compare surrounding segments to see if they're similar
                    if i < len(segment_features)-1:
                        segment_similarity = InterludeAnalyzer._compare_segment_patterns(
                            segment_features[i], segment_features[i+1])
                    else:
                        segment_similarity = 0.0
                        
                    # Calculate interlude score using a weighted combination
                    interlude_score = (
                        0.8 * segment_similarity +
                        0.1 * (1 - singing_cluster_ratio) +
                        0.1 * transition_activity
                    )
                    
                    # Store interlude information
                    potential_interludes.append({
                        'start': current_end,
                        'end': next_start,
                        'score': interlude_score,
                        'before_segment': i,
                        'after_segment': i+1
                    })
                    
        return potential_interludes

    @staticmethod
    def _merge_segments_with_interludes(singing_segments, interludes_to_include, verbose):
        """
        Merge segments that are connected by identified interludes.
        """
        final_segments = []
        i = 0
        
        while i < len(singing_segments):
            current_start, current_end = singing_segments[i]
            final_segments.append((current_start, current_end))
            
            # Check if there's an interlude after this segment
            for interlude in interludes_to_include:
                if interlude['before_segment'] == i:
                    next_idx = i + 1
                    if next_idx < len(singing_segments):
                        next_start, next_end = singing_segments[next_idx]
                        # Merge with the next segment
                        final_segments[-1] = (current_start, next_end)
                        i += 1  # Skip the next segment since we've merged it
                        break
                        
            i += 1
            
        return final_segments

    @staticmethod
    def _merge_overlapping_segments(segments, verbose=True):
        """
        Merge any segments that overlap or are adjacent.
        """
        if not segments:
            return []
            
        # Sort segments by start time
        segments.sort()
        
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                # Merge if this segment starts before the previous one ends
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                # Add as a separate segment
                merged.append((start, end))
                
        if verbose and len(merged) < len(segments):
            print(f"Merged {len(segments)} segments into {len(merged)} segments")
            
        return merged

    @staticmethod
    def _extract_pattern(sequence, max_length=20):
        """
        Extract a downsampled pattern from a sequence for quick comparison.
        This method is kept for backward compatibility.
        """
        if sequence is None or len(sequence) == 0:
            return np.array([])  # Return empty array if input is None or empty
            
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)
            
        if len(sequence) <= max_length:
            return sequence
            
        # Use scikit-learn compatible indexing
        indices = np.linspace(0, len(sequence)-1, max_length, dtype=int)
        return sequence[indices]

    @staticmethod
    def _compare_segment_patterns(segment1, segment2):
        """
        Compare two segments to determine if they likely belong to the same song.
        
        Uses scikit-learn for more efficient distance calculations.
        Falls back to DTW if installed, and histogram comparison as a last resort.
        """
        # Try using sklearn-based similarity first
        try:
            # --- Essential checks ---
            if not all(k in segment1 for k in ['features', 'clusters', 'duration']) or \
               not all(k in segment2 for k in ['features', 'clusters', 'duration']):
                 print("Warning: Segments missing essential keys ('features', 'clusters', 'duration').")
                 return 0.0  # Cannot compare
                 
            if segment1['features'] is None or segment1['clusters'] is None or \
               segment2['features'] is None or segment2['clusters'] is None:
                 print("Warning: Missing features or clusters in segments for comparison.")
                 # Fallback to duration only if critical data is missing
                 duration_ratio = min(segment1['duration'], segment2['duration']) / max(1e-6, max(segment1['duration'], segment2['duration']))
                 return 0.2 * duration_ratio  # Low similarity if data missing

            # --- Group features by cluster for more accurate comparison ---
            features1_by_cluster = _get_features_by_cluster(segment1['clusters'], segment1['features'])
            features2_by_cluster = _get_features_by_cluster(segment2['clusters'], segment2['features'])

            all_labels = set(features1_by_cluster.keys()) | set(features2_by_cluster.keys())
            
            # --- Handle case with no clusters ---
            if not all_labels:
                 print("Warning: No cluster labels found in segments for comparison.")
                 duration_ratio = min(segment1['duration'], segment2['duration']) / max(1e-6, max(segment1['duration'], segment2['duration']))
                 return 0.2 * duration_ratio  # Low similarity

            # --- Calculate similarity per matching cluster ---
            # Instead of DTW, use centroid-based comparison and distribution metrics
            total_similarity = 0.0
            total_weight = 0.0  # Weight by total number of feature frames compared

            for label in all_labels:
                features1 = features1_by_cluster.get(label)
                features2 = features2_by_cluster.get(label)

                # Only compare if this cluster label exists in both segments and has features
                if features1 is not None and features2 is not None and len(features1) > 0 and len(features2) > 0:
                    # Ensure features are 2D numpy arrays
                    if features1.ndim == 1: features1 = features1.reshape(-1, 1)
                    if features2.ndim == 1: features2 = features2.reshape(-1, 1)
                    
                    # Sanity check: feature dimensions must match
                    if features1.shape[1] != features2.shape[1]:
                        print(f"Warning: Feature dimensions mismatch for cluster {label} ({features1.shape[1]} vs {features2.shape[1]}). Skipping comparison for this cluster.")
                        continue
                    
                    # Scale features for better distance calculation
                    scaler = StandardScaler()
                    if len(features1) >= 2 and len(features2) >= 2:
                        # Only standardize if we have enough samples
                        combined = np.vstack([features1, features2])
                        scaled_combined = scaler.fit_transform(combined)
                        features1_scaled = scaled_combined[:len(features1)]
                        features2_scaled = scaled_combined[len(features1):]
                    else:
                        features1_scaled = features1
                        features2_scaled = features2
                    
                    # Calculate centroids
                    centroid1 = np.mean(features1_scaled, axis=0).reshape(1, -1)
                    centroid2 = np.mean(features2_scaled, axis=0).reshape(1, -1)
                    
                    # Calculate distance between centroids
                    centroid_distance = pairwise_distances(centroid1, centroid2, metric='euclidean')[0, 0]
                    
                    # Calculate distribution similarity using Nearest Neighbors
                    if len(features1_scaled) >= 5 and len(features2_scaled) >= 5:
                        # Only use NN if we have enough points
                        try:
                            # Use kNN to measure how well the distributions overlap
                            k = min(3, min(len(features1_scaled), len(features2_scaled)))
                            nn1 = NearestNeighbors(n_neighbors=k)
                            nn1.fit(features1_scaled)
                            nn2 = NearestNeighbors(n_neighbors=k)
                            nn2.fit(features2_scaled)
                            
                            # Query each set against the other
                            distances1, _ = nn1.kneighbors(features2_scaled)
                            distances2, _ = nn2.kneighbors(features1_scaled)
                            
                            # Average nearest neighbor distance
                            avg_nn_distance = (np.mean(distances1) + np.mean(distances2)) / 2
                            
                            # Combine centroid and distribution metrics
                            distance = 0.4 * centroid_distance + 0.6 * avg_nn_distance
                        except Exception as e:
                            print(f"Warning: NN calculation failed: {e}. Using centroid distance only.")
                            distance = centroid_distance
                    else:
                        distance = centroid_distance
                    
                    # Convert distance to similarity
                    similarity = 1.0 / (1.0 + distance)
                    
                    # Weight by the number of points
                    weight = len(features1) + len(features2)
                    total_similarity += similarity * weight
                    total_weight += weight

            # --- Combine results ---
            if total_weight > 0:
                aggregated_similarity = total_similarity / total_weight
            else:
                aggregated_similarity = 0.0
                print("Warning: No common cluster labels with features found between segments for comparison.")

            # Combine with duration ratio
            duration_ratio = min(segment1['duration'], segment2['duration']) / max(1e-6, max(segment1['duration'], segment2['duration']))
            combined_sim = 0.8 * aggregated_similarity + 0.2 * duration_ratio
            
            return combined_sim

        except ImportError:
            # If sklearn methods fail, try DTW
            try:
                from fastdtw import fastdtw
                from scipy.spatial.distance import euclidean
                
                # Fallback to DTW comparison (similar to original but with better error handling)
                # Logic similar to original DTW implementation
                # ...
                if segment1['features'] is None or segment2['features'] is None:
                    return 0.0
                    
                features1 = segment1['features']
                features2 = segment2['features']
                
                if len(features1) == 0 or len(features2) == 0:
                    return 0.0
                    
                # Ensure 2D arrays
                if features1.ndim == 1: features1 = features1.reshape(-1, 1)
                if features2.ndim == 1: features2 = features2.reshape(-1, 1)
                
                # Check dimensions match
                if features1.shape[1] != features2.shape[1]:
                    print(f"Warning: Feature dimensions mismatch ({features1.shape[1]} vs {features2.shape[1]}). Can't compare with DTW.")
                    return 0.0
                
                # Perform DTW
                distance, _ = fastdtw(features1, features2, dist=euclidean)
                
                # Normalize distance to similarity
                avg_len = (len(features1) + len(features2)) / 2
                normalized_distance = distance / max(1, avg_len)
                similarity = 1.0 / (1.0 + normalized_distance)
                
                # Include duration ratio
                duration_ratio = min(segment1['duration'], segment2['duration']) / max(1e-6, max(segment1['duration'], segment2['duration']))
                return 0.8 * similarity + 0.2 * duration_ratio
                
            except ImportError:
                print("Warning: fastdtw not found. Falling back to basic histogram comparison.")
                # Final fallback: histogram comparison (simplified but robust)
                duration_ratio = min(segment1.get('duration', 0), segment2.get('duration', 0)) / max(1e-6, max(segment1.get('duration', 0), segment2.get('duration', 0)))
                
                # Compare state histograms if available
                state_sim = 0.0
                if all(k in segment1 and k in segment2 and segment1[k] is not None and segment2[k] is not None 
                       for k in ['states']) and len(segment1['states']) > 0 and len(segment2['states']) > 0:
                    
                    states1 = np.array(segment1['states']).astype(int)
                    states2 = np.array(segment2['states']).astype(int)
                    
                    max_state = max(np.max(states1) if len(states1)>0 else -1, 
                                    np.max(states2) if len(states2)>0 else -1)
                    min_state_len = max(2, max_state + 1)
                    
                    state1_counts = np.bincount(states1, minlength=min_state_len) / max(1, len(states1))
                    state2_counts = np.bincount(states2, minlength=min_state_len) / max(1, len(states2))
                    state_sim = 1 - np.mean(np.abs(state1_counts - state2_counts))
                
                # Compare cluster histograms if available
                cluster_sim = 0.0
                if all(k in segment1 and k in segment2 and segment1[k] is not None and segment2[k] is not None 
                       for k in ['clusters']) and len(segment1['clusters']) > 0 and len(segment2['clusters']) > 0:
                    
                    clusters1 = np.array(segment1['clusters']).astype(int)
                    clusters2 = np.array(segment2['clusters']).astype(int)
                    
                    max_cluster = max(np.max(clusters1) if len(clusters1)>0 else -1, 
                                      np.max(clusters2) if len(clusters2)>0 else -1)
                    min_cluster_len = max(2, max_cluster + 1)
                    
                    cluster1_counts = np.bincount(clusters1, minlength=min_cluster_len) / max(1, len(clusters1))
                    cluster2_counts = np.bincount(clusters2, minlength=min_cluster_len) / max(1, len(clusters2))
                    cluster_sim = 1 - np.mean(np.abs(cluster1_counts - cluster2_counts))
                    
                # Final similarity with weights
                return 0.4 * state_sim + 0.4 * cluster_sim + 0.2 * duration_ratio
                
        except Exception as e:
            import traceback
            print(f"Error during segment pattern comparison: {e}")
            print(traceback.format_exc())
            # Fallback to duration ratio in case of unexpected errors
            duration_ratio = min(segment1.get('duration', 0), segment2.get('duration', 0)) / max(1e-6, max(segment1.get('duration', 0), segment2.get('duration', 0)))
            return 0.2 * duration_ratio