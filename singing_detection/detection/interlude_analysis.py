import numpy as np


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
        segment_features = []
        for start, end in singing_segments:
            segment_mask = (times >= start) & (times <= end)
            segment_indices = np.where(segment_mask)[0]
            if len(segment_indices) > 0:
                segment_feats = {
                    'duration': end - start,
                    'features': features_reduced[segment_indices],
                    'states': states[segment_indices],
                    'clusters': cluster_labels[segment_indices],
                    'start': start,
                    'end': end,
                    'state_pattern': InterludeAnalyzer._extract_pattern(states[segment_indices]),
                    'cluster_pattern': InterludeAnalyzer._extract_pattern(cluster_labels[segment_indices])
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
                        0.4 * segment_similarity +
                        0.3 * (1 - singing_cluster_ratio) +
                        0.3 * transition_activity
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
        if len(sequence) <= max_length:
            return sequence
        indices = np.linspace(0, len(sequence)-1, max_length, dtype=int)
        return sequence[indices]

    @staticmethod
    def _compare_segment_patterns(segment1, segment2):
        try:
            from fastdtw import fastdtw
            from scipy.spatial.distance import euclidean
            state_distance, _ = fastdtw(
                segment1['state_pattern'].reshape(-1, 1),
                segment2['state_pattern'].reshape(-1, 1),
                dist=euclidean
            )
            state_sim = 1.0 / (1.0 + state_distance / max(len(segment1['state_pattern']), len(segment2['state_pattern'])))
            cluster_distance, _ = fastdtw(
                segment1['cluster_pattern'].reshape(-1, 1),
                segment2['cluster_pattern'].reshape(-1, 1),
                dist=euclidean
            )
            cluster_sim = 1.0 / (1.0 + cluster_distance / max(len(segment1['cluster_pattern']), len(segment2['cluster_pattern'])))
            duration_ratio = min(segment1['duration'], segment2['duration']) / max(segment1['duration'], segment2['duration'])
            combined_sim = 0.4 * state_sim + 0.4 * cluster_sim + 0.2 * duration_ratio
            return combined_sim
        except ImportError:
            duration_ratio = min(segment1['duration'], segment2['duration']) / max(segment1['duration'], segment2['duration'])
            state1_counts = np.bincount(segment1['states'], minlength=2) / len(segment1['states'])
            state2_counts = np.bincount(segment2['states'], minlength=2) / len(segment2['states'])
            state_sim = 1 - np.mean(np.abs(state1_counts - state2_counts))
            cluster1_counts = np.bincount(segment1['clusters'], minlength=2) / len(segment1['clusters'])
            cluster2_counts = np.bincount(segment2['clusters'], minlength=2) / len(segment2['clusters'])
            cluster_sim = 1 - np.mean(np.abs(cluster1_counts - cluster2_counts))
            return 0.4 * state_sim + 0.4 * cluster_sim + 0.2 * duration_ratio 