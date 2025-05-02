import numpy as np
from hmmlearn import hmm
from sklearn.covariance import empirical_covariance, ledoit_wolf
from sklearn.metrics import pairwise_distances


class DetectionEngine:
    @staticmethod
    def prepare_hmm_features(features, cluster_labels, singing_cluster):
        """
        Prepare features for HMM by adding cluster information.
        """
        return np.column_stack([
            features,
            np.array([1.0 if c == singing_cluster else 0.0 for c in cluster_labels]).reshape(-1, 1)
        ])

    @staticmethod
    def fit_predict_hmm(features_for_hmm, params):
        """
        Fit a Gaussian HMM and predict states and posteriors.
        """
        model = hmm.GaussianHMM(
            n_components=2, 
            covariance_type="full", 
            n_iter=100, 
            random_state=42,
            verbose=params.get('verbose', False)
        )
        model.fit(features_for_hmm)
        states = model.predict(features_for_hmm)
        posteriors = model.predict_proba(features_for_hmm)
        return states, posteriors, model

    @staticmethod
    def find_segments(states, times, target_state, min_duration):
        """
        Find continuous segments of a target state.
        """
        mask = (states == target_state)
        transitions = np.diff(mask.astype(int))
        segment_starts = np.where(transitions == 1)[0]
        segment_ends = np.where(transitions == -1)[0]
        
        # Handle edge cases where the state starts at the beginning or ends at the end
        if mask[0]:
            segment_starts = np.insert(segment_starts, 0, 0)
        if mask[-1]:
            segment_ends = np.append(segment_ends, len(mask) - 1)
            
        segments = []
        for start_idx, end_idx in zip(segment_starts, segment_ends):
            start_time = times[start_idx]
            end_time = times[end_idx]
            duration = end_time - start_time
            if duration >= min_duration:
                segments.append((start_time, end_time))
                
        return segments

    @staticmethod
    def evaluate_segments(segments, features_reduced, times, cluster_labels, singing_cluster,
                         singing_ref_indices, non_singing_ref_indices, singing_centroid, non_singing_centroid):
        """
        Evaluate segments based on their similarity to singing and non-singing references.
        Uses scikit-learn's robust covariance estimation for better Mahalanobis distance.
        """
        evaluated_segments = []
        n_clusters = 2
        
        # Use scikit-learn's empirical_covariance or ledoit_wolf for more stable covariance estimation
        try:
            # Try Ledoit-Wolf estimator first (better for high-dimensional data)
            if len(singing_ref_indices) > 3:  # Need at least n+2 samples for dimension n
                singing_cov = ledoit_wolf(features_reduced[singing_ref_indices])[0]
            else:
                singing_cov = empirical_covariance(features_reduced[singing_ref_indices])
                
            if len(non_singing_ref_indices) > 3:
                non_singing_cov = ledoit_wolf(features_reduced[non_singing_ref_indices])[0]
            else:
                non_singing_cov = empirical_covariance(features_reduced[non_singing_ref_indices])
                
            # Add regularization for numerical stability
            singing_cov += np.eye(singing_cov.shape[0]) * 1e-6
            non_singing_cov += np.eye(non_singing_cov.shape[0]) * 1e-6
            
            # Compute inverse covariance matrices
            singing_cov_inv = np.linalg.pinv(singing_cov)
            non_singing_cov_inv = np.linalg.pinv(non_singing_cov)
        except Exception as e:
            print(f"Warning: Failed to compute covariance matrices: {e}")
            # Fallback to identity matrices
            dim = features_reduced.shape[1]
            singing_cov_inv = np.eye(dim)
            non_singing_cov_inv = np.eye(dim)
        
        for start, end in segments:
            segment_mask = (times >= start) & (times <= end)
            segment_indices = np.where(segment_mask)[0]
            
            if len(segment_indices) == 0:
                continue
                
            segment_features = features_reduced[segment_indices]
            segment_centroid = np.mean(segment_features, axis=0)
            
            # Calculate Mahalanobis distances
            dist_to_singing = DetectionEngine._mahalanobis_distance(
                segment_centroid, singing_centroid.flatten(), singing_cov_inv)
            dist_to_non_singing = DetectionEngine._mahalanobis_distance(
                segment_centroid, non_singing_centroid.flatten(), non_singing_cov_inv)
            
            total_dist = dist_to_singing + dist_to_non_singing
            similarity_score = dist_to_non_singing / total_dist if total_dist > 0 else 0.5
            
            # Cluster-based score
            cluster_counts = np.bincount(cluster_labels[segment_indices], minlength=n_clusters)
            cluster_probs = cluster_counts / np.sum(cluster_counts)
            cluster_score = cluster_probs[singing_cluster]
            
            # Combined probability
            singing_probability = 0.6 * similarity_score + 0.4 * cluster_score
            
            evaluated_segments.append((start, end, float(singing_probability)))
            
        return evaluated_segments

    @staticmethod
    def filter_and_merge_segments(evaluated_segments, threshold, min_gap):
        """
        Filter segments based on probability threshold and merge adjacent segments.
        """
        # Filter by threshold
        singing_segments = [(start, end) for start, end, prob in evaluated_segments if prob >= threshold]
        
        if len(singing_segments) <= 1:
            return singing_segments
            
        # Sort by start time to ensure proper merging
        singing_segments.sort(key=lambda x: x[0])
        
        # Merge adjacent segments
        merged_segments = []
        it = iter(singing_segments)
        current_start, current_end = next(it)
        
        for start, end in it:
            if start - current_end < min_gap:
                # Merge with previous segment
                current_end = max(current_end, end)
            else:
                # Add completed segment and start a new one
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
                
        # Add the last segment
        merged_segments.append((current_start, current_end))
        
        return merged_segments

    @staticmethod
    def _mahalanobis_distance(x, mean, cov_inv):
        """
        Calculate Mahalanobis distance with better error handling.
        """
        try:
            diff = x - mean
            return np.sqrt(np.abs(diff.dot(cov_inv).dot(diff.T)))
        except Exception as e:
            print(f"Warning: Mahalanobis calculation failed: {e}. Using Euclidean distance.")
            return np.linalg.norm(x - mean)

    # Optionally, interlude analysis can be included as a method or sub-class
    # For brevity, not included here, but can be added if needed 