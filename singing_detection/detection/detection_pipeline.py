from singing_detection.detection.detection_engine import DetectionEngine
from singing_detection.detection.feature_engineering import FeatureEngineer
from singing_detection.detection.interlude_analysis import InterludeAnalyzer


class SingingDetectionPipeline:
    @staticmethod
    def run(features_df, singing_ref, non_singing_ref, params):
        # 1. Feature engineering
        features_reduced, times = FeatureEngineer.prepare(features_df, params)
        # 2. Reference info
        (
            singing_ref_indices, non_singing_ref_indices,
            singing_centroid, non_singing_centroid
        ) = FeatureEngineer.get_reference_info(features_reduced, times, singing_ref, non_singing_ref)
        # 3. Clustering
        cluster_labels, singing_cluster = FeatureEngineer.run_clustering(
            features_reduced, singing_centroid, non_singing_centroid, params
        )
        # 4. HMM fit/predict (Using cluster-initialized HMM)
        # No longer need prepare_hmm_features for HMM input
        # features_for_hmm = DetectionEngine.prepare_hmm_features(features_reduced, cluster_labels, singing_cluster)
        states, posteriors, hmm_model, hmm_singing_state_idx = DetectionEngine.fit_predict_hmm(
            features_reduced, 
            cluster_labels, 
            singing_cluster, 
            params=params # Pass params if fit_predict_hmm uses it, otherwise optional
        )

        # Handle potential fitting failure
        if hmm_model is None:
            print("[Pipeline Error] HMM fitting failed. Cannot proceed with segment finding.")
            # Return empty segments and minimal results or raise an exception
            return [], {'error': 'HMM fitting failed'}

        # 5. Segment finding (using the determined singing state)
        min_duration = params.get('min_duration', 2.0)
        singing_segments_raw = DetectionEngine.find_segments(states, times, hmm_singing_state_idx, min_duration)
        
        # Note: Evaluating segments might need adjustment if it relied on the specific HMM states 0/1 before.
        # The current evaluate_segments uses cluster_labels and centroids, so it might be okay.
        # If evaluation needs posteriors for the *singing* state specifically, use hmm_singing_state_idx:
        # singing_posteriors = posteriors[:, hmm_singing_state_idx]

        # 6. Segment evaluation (using the raw segments found for the singing state)
        evaluated_segments = DetectionEngine.evaluate_segments(
            singing_segments_raw, # Evaluate only the segments identified as singing by HMM
            features_reduced, times, cluster_labels, singing_cluster,
            singing_ref_indices, non_singing_ref_indices, singing_centroid, non_singing_centroid
        )
        # 7. Segment merging
        threshold = params.get('threshold', 0.6)
        min_gap = params.get('min_gap', 1.5)
        singing_segments = DetectionEngine.filter_and_merge_segments(evaluated_segments, threshold, min_gap)
        # 8. Interlude analysis (always performed)
        singing_segments = InterludeAnalyzer.analyze(
            singing_segments, times, features_reduced, states, posteriors, cluster_labels, singing_cluster, params
        )
        # 9. Prepare results
        results = {
            'times': times,
            'features_reduced': features_reduced,
            'cluster_labels': cluster_labels,
            'singing_cluster': singing_cluster,
            'states': states,
            'posteriors': posteriors,
            'evaluated_segments': evaluated_segments,
        }
        return singing_segments, results 