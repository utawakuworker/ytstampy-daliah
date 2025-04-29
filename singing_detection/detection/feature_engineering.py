import numpy as np
from scipy.cluster.vq import kmeans2


class FeatureEngineer:
    @staticmethod
    def prepare(features_df, params):
        feature_cols = [col for col in features_df.columns if col != 'time' and not col.endswith('_cluster')]
        features = features_df[feature_cols].values
        times = features_df['time'].values
        # Manual standard scaling
        features_scaled = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        dim_reduction = params.get('dim_reduction', 'pca')
        n_components = params.get('n_components', 10)
        verbose = params.get('verbose', True)

        features_reduced = features_scaled  # Default: no reduction
        reduction_method = None

        if dim_reduction == 'pca':
            # PCA using numpy SVD
            X_centered = features_scaled - features_scaled.mean(axis=0)
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            features_reduced = np.dot(X_centered, Vt[:n_components].T)
            reduction_method = "PCA"
        elif dim_reduction == 'umap':
            try:
                import umap
                reducer = umap.UMAP(n_components=min(n_components, features_scaled.shape[1]))
                features_reduced = reducer.fit_transform(features_scaled)
                reduction_method = "UMAP"
            except ImportError:
                reduction_method = None

        if verbose:
            if reduction_method:
                print(f"Reduced features from {features_scaled.shape[1]} to {features_reduced.shape[1]} dimensions using {reduction_method}")
            else:
                print(f"Using {features_scaled.shape[1]} original features without reduction")

        return features_reduced, times

    @staticmethod
    def get_reference_info(features, times, singing_ref, non_singing_ref):
        sing_start, sing_end = singing_ref
        non_sing_start, non_sing_end = non_singing_ref
        singing_ref_mask = (times >= sing_start) & (times <= sing_end)
        non_singing_ref_mask = (times >= non_sing_start) & (times <= non_sing_end)
        singing_ref_indices = np.where(singing_ref_mask)[0]
        non_singing_ref_indices = np.where(non_singing_ref_mask)[0]
        singing_centroid = np.mean(features[singing_ref_indices], axis=0).reshape(1, -1)
        non_singing_centroid = np.mean(features[non_singing_ref_indices], axis=0).reshape(1, -1)
        return singing_ref_indices, non_singing_ref_indices, singing_centroid, non_singing_centroid

    @staticmethod
    def run_clustering(features, singing_centroid, non_singing_centroid, params):
        n_clusters = 2
        initial_centroids = np.vstack([singing_centroid, non_singing_centroid])
        # Use scipy's kmeans2 for clustering
        centroids, cluster_labels = kmeans2(features, initial_centroids, minit='matrix', iter=20)
        singing_ref_mask = (np.linalg.norm(features - singing_centroid, axis=1) < np.linalg.norm(features - non_singing_centroid, axis=1))
        singing_cluster_votes = np.bincount(cluster_labels[singing_ref_mask], minlength=n_clusters)
        non_singing_cluster_votes = np.bincount(cluster_labels[~singing_ref_mask], minlength=n_clusters)
        singing_ratio = singing_cluster_votes / np.sum(singing_cluster_votes)
        non_singing_ratio = non_singing_cluster_votes / np.sum(non_singing_cluster_votes)
        singing_scores = singing_ratio / (non_singing_ratio + 0.001)
        singing_cluster = np.argmax(singing_scores)
        if params.get('verbose', True):
            print(f"Identified cluster {singing_cluster} as likely singing cluster")
        return cluster_labels, singing_cluster 