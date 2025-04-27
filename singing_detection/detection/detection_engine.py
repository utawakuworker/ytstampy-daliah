import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import List, Tuple, Dict, Optional, Any

class DetectionEngine:
    @staticmethod
    def prepare_hmm_features(features, cluster_labels, singing_cluster):
        return np.column_stack([
            features,
            np.array([1.0 if c == singing_cluster else 0.0 for c in cluster_labels]).reshape(-1, 1)
        ])

    @staticmethod
    def fit_predict_hmm(features_for_hmm, params):
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100, random_state=42)
        model.fit(features_for_hmm)
        states = model.predict(features_for_hmm)
        posteriors = model.predict_proba(features_for_hmm)
        return states, posteriors, model

    @staticmethod
    def find_segments(states, times, target_state, min_duration):
        mask = (states == target_state)
        transitions = np.diff(mask.astype(int))
        segment_starts = np.where(transitions == 1)[0]
        segment_ends = np.where(transitions == -1)[0]
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
        evaluated_segments = []
        n_clusters = 2
        singing_cov_inv = DetectionEngine._safe_cov_inv(features_reduced[singing_ref_indices])
        non_singing_cov_inv = DetectionEngine._safe_cov_inv(features_reduced[non_singing_ref_indices])
        for start, end in segments:
            segment_mask = (times >= start) & (times <= end)
            segment_indices = np.where(segment_mask)[0]
            if len(segment_indices) == 0:
                continue
            segment_features = features_reduced[segment_indices]
            segment_centroid = np.mean(segment_features, axis=0)
            dist_to_singing = DetectionEngine._mahalanobis_distance(segment_centroid, singing_centroid.flatten(), singing_cov_inv)
            dist_to_non_singing = DetectionEngine._mahalanobis_distance(segment_centroid, non_singing_centroid.flatten(), non_singing_cov_inv)
            total_dist = dist_to_singing + dist_to_non_singing
            similarity_score = dist_to_non_singing / total_dist if total_dist > 0 else 0.5
            cluster_counts = np.bincount(cluster_labels[segment_indices], minlength=n_clusters)
            cluster_probs = cluster_counts / np.sum(cluster_counts)
            cluster_score = cluster_probs[singing_cluster]
            singing_probability = 0.6 * similarity_score + 0.4 * cluster_score
            evaluated_segments.append((start, end, float(singing_probability)))
        return evaluated_segments

    @staticmethod
    def filter_and_merge_segments(evaluated_segments, threshold, min_gap):
        singing_segments = [(start, end) for start, end, prob in evaluated_segments if prob >= threshold]
        if len(singing_segments) <= 1:
            return singing_segments
        merged_segments = []
        it = iter(singing_segments)
        current_start, current_end = next(it)
        for start, end in it:
            if start - current_end < min_gap:
                current_end = end
            else:
                merged_segments.append((current_start, current_end))
                current_start, current_end = start, end
        merged_segments.append((current_start, current_end))
        return merged_segments

    @staticmethod
    def _safe_cov_inv(features):
        cov = np.cov(features.T)
        cov += np.eye(cov.shape[0]) * 1e-6
        try:
            return np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(cov)

    @staticmethod
    def _mahalanobis_distance(x, mean, cov_inv):
        try:
            diff = x - mean
            return np.sqrt(diff.dot(cov_inv).dot(diff.T))
        except Exception:
            return np.linalg.norm(x - mean)

    # Optionally, interlude analysis can be included as a method or sub-class
    # For brevity, not included here, but can be added if needed 