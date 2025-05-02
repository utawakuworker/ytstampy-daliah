
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
        # 4. HMM fit/predict
        features_for_hmm = DetectionEngine.prepare_hmm_features(features_reduced, cluster_labels, singing_cluster)
        states, posteriors, hmm_model = DetectionEngine.fit_predict_hmm(features_for_hmm, params)
        # 5. Segment finding (for both states)
        min_duration = params.get('min_duration', 2.0)
        segments_state0 = DetectionEngine.find_segments(states, times, 0, min_duration)
        segments_state1 = DetectionEngine.find_segments(states, times, 1, min_duration)
        # 6. Segment evaluation
        evaluated_segments = DetectionEngine.evaluate_segments(
            segments_state0 + segments_state1, features_reduced, times, cluster_labels, singing_cluster,
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