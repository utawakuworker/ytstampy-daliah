import numpy as np

# Helper function to group features by cluster label
def _get_features_by_cluster(clusters, features):
    """Groups feature vectors by their assigned cluster label."""
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
        interlude_threshold = params.get('interlude_threshold', 0.3)
        max_interlude_duration = params.get('max_interlude_duration', 20.0)
        verbose = params.get('verbose', True)
        if verbose:
            print("Analyzing song structure to identify interludes...")
        if len(singing_segments) < 2:
            return singing_segments
        segment_features = InterludeAnalyzer._extract_segment_features(singing_segments, times, features_reduced, states, cluster_labels, singing_cluster)
        potential_interludes = InterludeAnalyzer._find_potential_interludes(singing_segments, times, states, cluster_labels, singing_cluster, segment_features, max_interlude_duration)
        interludes_to_include = [interlude for interlude in potential_interludes if interlude['score'] >= interlude_threshold]
        final_segments = InterludeAnalyzer._merge_segments_with_interludes(singing_segments, interludes_to_include, verbose)
        final_segments = InterludeAnalyzer._merge_overlapping_segments(final_segments, verbose)
        if verbose:
            print(f"Final segments after structure analysis: {len(final_segments)}")
        return final_segments

    @staticmethod
    def _extract_segment_features(singing_segments, times, features_reduced, states, cluster_labels, singing_cluster):
        # Ensure main arrays are numpy arrays if they aren't already for consistency
        if features_reduced is not None and not isinstance(features_reduced, np.ndarray):
            features_reduced = np.array(features_reduced)
        if states is not None and not isinstance(states, np.ndarray):
             states = np.array(states)
        if cluster_labels is not None and not isinstance(cluster_labels, np.ndarray):
            cluster_labels = np.array(cluster_labels)
            
        segment_features = []
        for start, end in singing_segments:
            segment_mask = (times >= start) & (times <= end)
            segment_indices = np.where(segment_mask)[0]
            if len(segment_indices) > 0:
                # Store the actual features and clusters for the segment
                # Ensure segment-level arrays are also numpy arrays
                segment_feats_data = features_reduced[segment_indices] if features_reduced is not None else None
                segment_states_data = states[segment_indices] if states is not None else None
                segment_clusters_data = cluster_labels[segment_indices] if cluster_labels is not None else None

                segment_feats = {
                    'duration': end - start,
                    'features': np.array(segment_feats_data) if segment_feats_data is not None else None,
                    'states': np.array(segment_states_data) if segment_states_data is not None else None,
                    'clusters': np.array(segment_clusters_data) if segment_clusters_data is not None else None,
                    'start': start,
                    'end': end
                    # state_pattern and cluster_pattern removed as they are no longer used here
                }
                segment_features.append(segment_feats)
        return segment_features

    @staticmethod
    def _find_potential_interludes(singing_segments, times, states, cluster_labels, singing_cluster, segment_features, max_interlude_duration):
        potential_interludes = []
        for i in range(len(singing_segments) - 1):
            current_end = singing_segments[i][1]
            next_start = singing_segments[i+1][0]
            if next_start > current_end and (next_start - current_end) <= max_interlude_duration:
                interlude_mask = (times >= current_end) & (times <= next_start)
                interlude_indices = np.where(interlude_mask)[0]
                if len(interlude_indices) > 0:
                    state_transitions = np.diff(states[interlude_indices])
                    transition_activity = np.sum(state_transitions != 0) / max(1, len(interlude_indices) - 1)
                    singing_cluster_ratio = np.mean(cluster_labels[interlude_indices] == singing_cluster)
                    if i < len(segment_features)-1:
                        segment_similarity = InterludeAnalyzer._compare_segment_patterns(
                            segment_features[i], segment_features[i+1])
                    else:
                        segment_similarity = 0.0
                    interlude_score = (
                        0.8 * segment_similarity +
                        0.1 * (1 - singing_cluster_ratio) +
                        0.1 * transition_activity
                    )
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
        final_segments = []
        i = 0
        while i < len(singing_segments):
            current_start, current_end = singing_segments[i]
            final_segments.append((current_start, current_end))
            for interlude in interludes_to_include:
                if interlude['before_segment'] == i:
                    next_idx = i + 1
                    if next_idx < len(singing_segments):
                        next_start, next_end = singing_segments[next_idx]
                        final_segments[-1] = (current_start, next_end)
                        i += 1
                        break
            i += 1
        return final_segments

    @staticmethod
    def _merge_overlapping_segments(segments, verbose=True):
        if not segments:
            return []
        segments.sort()
        merged = [segments[0]]
        for start, end in segments[1:]:
            prev_start, prev_end = merged[-1]
            if start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end))
            else:
                merged.append((start, end))
        if verbose and len(merged) < len(segments):
            print(f"Merged {len(segments)} segments into {len(merged)} segments")
        return merged

    @staticmethod
    def _extract_pattern(sequence, max_length=20):
        # This method is no longer used by the primary path of _compare_segment_patterns,
        # but kept in case it's used elsewhere or as a potential future fallback.
        if sequence is None or len(sequence) == 0:
            return np.array([]) # Return empty array if input is None or empty
        if not isinstance(sequence, np.ndarray):
            sequence = np.array(sequence)
            
        if len(sequence) <= max_length:
            return sequence
        indices = np.linspace(0, len(sequence)-1, max_length, dtype=int)
        return sequence[indices]

    @staticmethod
    def _compare_segment_patterns(segment1, segment2):
        """
        Compares two segments based on the similarity of their acoustic features
        within matching cluster labels, using DTW. Also considers duration ratio.
        """
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            # numpy is imported at the top

            # --- Essential checks ---
            if not all(k in segment1 for k in ['features', 'clusters', 'duration']) or \
               not all(k in segment2 for k in ['features', 'clusters', 'duration']):
                 print("Warning: Segments missing essential keys ('features', 'clusters', 'duration').")
                 return 0.0 # Cannot compare
                 
            if segment1['features'] is None or segment1['clusters'] is None or \
               segment2['features'] is None or segment2['clusters'] is None:
                 print("Warning: Missing features or clusters in segments for comparison.")
                 # Fallback to duration only if critical data is missing
                 duration_ratio = min(segment1['duration'], segment2['duration']) / max(1e-6, segment1['duration'], segment2['duration'])
                 return 0.2 * duration_ratio # Low similarity if data missing

            # --- Group features by cluster --- 
            features1_by_cluster = _get_features_by_cluster(segment1['clusters'], segment1['features'])
            features2_by_cluster = _get_features_by_cluster(segment2['clusters'], segment2['features'])

            all_labels = set(features1_by_cluster.keys()) | set(features2_by_cluster.keys())
            
            # --- Handle case with no clusters --- 
            if not all_labels:
                 print("Warning: No cluster labels found in segments for comparison.")
                 duration_ratio = min(segment1['duration'], segment2['duration']) / max(1e-6, segment1['duration'], segment2['duration'])
                 return 0.2 * duration_ratio # Low similarity

            # --- Calculate DTW similarity per matching cluster --- 
            total_similarity = 0.0
            total_weight = 0.0 # Weight by total number of feature frames compared

            for label in all_labels:
                features1 = features1_by_cluster.get(label)
                features2 = features2_by_cluster.get(label)

                # Only compare if this cluster label exists in both segments and has features
                if features1 is not None and features2 is not None and len(features1) > 0 and len(features2) > 0:
                    # Ensure features are 2D numpy arrays for DTW (usually they are N x D)
                    if features1.ndim == 1: features1 = features1.reshape(-1, 1)
                    if features2.ndim == 1: features2 = features2.reshape(-1, 1)
                    
                    # Sanity check: feature dimensions must match for Euclidean distance
                    if features1.shape[1] != features2.shape[1]:
                        print(f"Warning: Feature dimensions mismatch for cluster {label} ({features1.shape[1]} vs {features2.shape[1]}). Skipping comparison for this cluster.")
                        continue 

                    # Perform DTW on the feature sequences for this cluster
                    distance, path = fastdtw(features1, features2, dist=euclidean)
                    
                    # Normalize distance into similarity (0 to 1)
                    # Normalize distance by average length to make it less length-dependent
                    avg_len = (len(features1) + len(features2)) / 2
                    if avg_len == 0: 
                         similarity = 1.0 # Two empty sequences are perfectly similar?
                    else:
                         normalized_distance = distance / avg_len 
                         similarity = 1.0 / (1.0 + normalized_distance) 

                    # Weight similarity by the total number of frames for this cluster
                    weight = len(features1) + len(features2)
                    total_similarity += similarity * weight
                    total_weight += weight

            # --- Combine results --- 
            if total_weight > 0:
                aggregated_similarity = total_similarity / total_weight
            else:
                # No common clusters found or compared
                aggregated_similarity = 0.0 
                print("Warning: No common cluster labels with features found between segments for comparison.")

            # Combine with duration ratio (80% feature similarity, 20% duration)
            duration_ratio = min(segment1['duration'], segment2['duration']) / max(1e-6, segment1['duration'], segment2['duration']) # Avoid division by zero
            combined_sim = 0.8 * aggregated_similarity + 0.2 * duration_ratio
            
            # Optional: print debug info
            # print(f" Seg1: {segment1['start']:.1f}-{segment1['end']:.1f}, Seg2: {segment2['start']:.1f}-{segment2['end']:.1f}, AggSim: {aggregated_similarity:.3f}, DurRatio: {duration_ratio:.3f}, Combined: {combined_sim:.3f}")
            return combined_sim

        except ImportError:
            print("Warning: fastdtw not found. Falling back to basic histogram comparison.")
            # --- Fallback Logic (Histogram Comparison) ---
            # This remains mostly the same as the original fallback.
            duration_ratio = min(segment1.get('duration', 0), segment2.get('duration', 0)) / max(1e-6, segment1.get('duration', 1e-6), segment2.get('duration', 1e-6))
            
            state_sim = 0.0
            if 'states' in segment1 and 'states' in segment2 and \
               segment1['states'] is not None and segment2['states'] is not None and \
               len(segment1['states']) > 0 and len(segment2['states']) > 0:
                
                states1 = np.array(segment1['states']).astype(int) # Ensure numpy array of int
                states2 = np.array(segment2['states']).astype(int) # Ensure numpy array of int

                max_state = max(np.max(states1) if len(states1)>0 else -1, 
                                np.max(states2) if len(states2)>0 else -1)
                min_state_len = max(2, max_state + 1)
                
                state1_counts = np.bincount(states1, minlength=min_state_len) / len(states1)
                state2_counts = np.bincount(states2, minlength=min_state_len) / len(states2)
                state_sim = 1 - np.mean(np.abs(state1_counts - state2_counts))

            cluster_sim = 0.0
            if 'clusters' in segment1 and 'clusters' in segment2 and \
               segment1['clusters'] is not None and segment2['clusters'] is not None and \
               len(segment1['clusters']) > 0 and len(segment2['clusters']) > 0:
                
                clusters1 = np.array(segment1['clusters']).astype(int) # Ensure numpy array of int
                clusters2 = np.array(segment2['clusters']).astype(int) # Ensure numpy array of int

                max_cluster = max(np.max(clusters1) if len(clusters1)>0 else -1, 
                                  np.max(clusters2) if len(clusters2)>0 else -1)
                min_cluster_len = max(2, max_cluster + 1)
                
                cluster1_counts = np.bincount(clusters1, minlength=min_cluster_len) / len(clusters1)
                cluster2_counts = np.bincount(clusters2, minlength=min_cluster_len) / len(clusters2)
                cluster_sim = 1 - np.mean(np.abs(cluster1_counts - cluster2_counts))
                
            # Original fallback weighting: 0.4 state, 0.4 cluster, 0.2 duration
            return 0.4 * state_sim + 0.4 * cluster_sim + 0.2 * duration_ratio
            
        except Exception as e:
            import traceback
            print(f"Error during segment pattern comparison: {e}")
            print(traceback.format_exc())
            # Fallback to duration ratio only in case of unexpected errors in the main path
            duration_ratio = min(segment1.get('duration', 0), segment2.get('duration', 0)) / max(1e-6, segment1.get('duration', 1e-6), segment2.get('duration', 1e-6))
            return 0.2 * duration_ratio # Return a low similarity based only on duration