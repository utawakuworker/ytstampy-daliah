import numpy as np
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal
import logging # Added for logging

# Setup logging (optional, but good practice)
logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO) # Configure level as needed

class DetectionEngine:
    @staticmethod
    def prepare_hmm_features(features, cluster_labels, singing_cluster):
        # This method might still be useful elsewhere, but not directly for HMM input now
        return np.column_stack([
            features,
            np.array([1.0 if c == singing_cluster else 0.0 for c in cluster_labels]).reshape(-1, 1)
        ])

    # --- Helper Methods for HMM --- 

    @staticmethod
    def _standardize_features(features):
        """Standardizes features to zero mean and unit variance."""
        if features.ndim == 1:
            features = features.reshape(-1, 1)
        # Check for constant features before calculating std dev
        std = np.std(features, axis=0)
        is_constant = std < 1e-8
        if np.any(is_constant):
            logger.warning("Some features are constant or near-constant.")
        
        mean = np.mean(features, axis=0)
        epsilon = 1e-8
        # Avoid division by zero for constant features by adding epsilon
        features_std = (features - mean) / (std + epsilon)
        return features_std

    @staticmethod
    def _initialize_hmm_distributions(features_std, cluster_labels, singing_cluster):
        """Initializes HMM distributions based on cluster statistics."""
        dims = features_std.shape[1]
        cov_epsilon = 1e-6

        non_singing_indices = np.where(cluster_labels != singing_cluster)[0]
        singing_indices = np.where(cluster_labels == singing_cluster)[0]

        def create_distribution(indices, cluster_name):
            """Safely compute mean/covariance and create a Normal distribution."""
            if len(indices) < dims + 1:
                logger.warning(
                    f'Cluster "{cluster_name}" has too few points ({len(indices)}) ' \
                    f'for stable covariance calculation (dims={dims}). Using default Normal().'
                )
                return Normal() # Fallback to uninitialized

            cluster_features = features_std[indices]
            mean = torch.from_numpy(np.mean(cluster_features, axis=0)).float()
            cov = np.cov(cluster_features.T)
            cov += np.eye(dims) * cov_epsilon # Regularize
            cov = torch.from_numpy(cov).float()
            return Normal(means=mean, covs=cov)

        d_non_singing = create_distribution(non_singing_indices, "non-singing")
        d_singing = create_distribution(singing_indices, "singing")

        # Return distributions (non-singing=state 0, singing=state 1) and indices
        return d_non_singing, d_singing, non_singing_indices, singing_indices

    @staticmethod
    def _determine_final_singing_state(model, features_std, singing_indices):
        """Determines the HMM state index corresponding to singing after fitting."""
        if not singing_indices.size > 0:
            logger.warning("Initial singing cluster was empty, cannot determine final singing state reliably. Defaulting to state 1.")
            return 1

        initial_singing_centroid = np.mean(features_std[singing_indices], axis=0)

        try:
            # Check if model and distributions exist
            if not model or not hasattr(model, 'distributions') or len(model.distributions) < 2:
                 raise AttributeError("Model or distributions not properly initialized.")
                 
            learned_mean0 = model.distributions[0].means.detach().numpy()
            learned_mean1 = model.distributions[1].means.detach().numpy()

            dist_to_mean0 = np.linalg.norm(initial_singing_centroid - learned_mean0)
            dist_to_mean1 = np.linalg.norm(initial_singing_centroid - learned_mean1)

            final_singing_state = 1 if dist_to_mean1 < dist_to_mean0 else 0
            return final_singing_state
        except (AttributeError, IndexError, TypeError) as e:
            logger.warning(f"Could not access learned means to determine singing state ({e}). Defaulting to state 1.")
            return 1 # Default if means are not accessible

    # --- Main HMM Fitting Method --- 

    @staticmethod
    def fit_predict_hmm(features, cluster_labels, singing_cluster, params=None):
        """Fits an HMM using initial cluster labels to initialize parameters."""
        # Use provided params dict or default if None
        if params is None:
            params = {}
            
        torch.manual_seed(params.get('random_seed', 42))
        np.random.seed(params.get('random_seed', 42))

        logger.info("Starting HMM fitting and prediction.")

        # 1. Preprocess Features
        try:
            features_std = DetectionEngine._standardize_features(features)
        except Exception as e:
            logger.error(f"Error during feature standardization: {e}", exc_info=True)
            return np.zeros(features.shape[0]), np.ones((features.shape[0], 2)) * 0.5, None, -1

        # 2. Initialize Distributions based on Clusters
        try:
             d_non_singing, d_singing, non_singing_indices, singing_indices = \
                 DetectionEngine._initialize_hmm_distributions(features_std, cluster_labels, singing_cluster)
        except Exception as e:
            logger.error(f"Error during HMM distribution initialization: {e}", exc_info=True)
            return np.zeros(features.shape[0]), np.ones((features.shape[0], 2)) * 0.5, None, -1

        # 3. Define HMM parameters
        hmm_distributions = [d_non_singing, d_singing] # state 0=non-singing, state 1=singing
        initial_starts = torch.tensor([0.5, 0.5], dtype=torch.float32)
        initial_edges = torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
        hmm_config = {
            'max_iter': params.get('hmm_max_iter', 100),
            'tol': params.get('hmm_tol', 1e-3),
            'verbose': params.get('hmm_verbose', False)
        }

        # 4. Create and Fit HMM
        model = DenseHMM(
            distributions=hmm_distributions,
            edges=initial_edges,
            starts=initial_starts,
            **hmm_config
        )
        
        X = torch.from_numpy(features_std).float().unsqueeze(0) # Prepare input tensor
        try:
            logger.info(f"Fitting DenseHMM with config: {hmm_config}")
            model.fit(X)
            logger.info("HMM fitting completed.")
        except torch._C._LinAlgError as e:
            logger.error(f"LinAlgError during HMM fitting: {e}. Check feature scaling/variance or cluster separation.", exc_info=True)
            return np.zeros(features.shape[0]), np.ones((features.shape[0], 2)) * 0.5, None, -1
        except Exception as e:
             logger.error(f"Unexpected error during HMM fitting: {e}", exc_info=True)
             return np.zeros(features.shape[0]), np.ones((features.shape[0], 2)) * 0.5, None, -1

        # 5. Predict States and Posteriors
        try:
            states_tensor = model.predict(X)
            posteriors_tensor = model.predict_proba(X)
            states = states_tensor[0].numpy()
            posteriors = posteriors_tensor[0].numpy()
        except Exception as e:
            logger.error(f"Error during HMM prediction: {e}", exc_info=True)
            return np.zeros(features.shape[0]), np.ones((features.shape[0], 2)) * 0.5, model, -1 # Return model but indicate prediction failure

        # 6. Determine Final Singing State
        final_singing_state = DetectionEngine._determine_final_singing_state(model, features_std, singing_indices)
        logger.info(f"Determined final singing state index: {final_singing_state}")

        return states, posteriors, model, final_singing_state


    # --- Other Static Methods --- 

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