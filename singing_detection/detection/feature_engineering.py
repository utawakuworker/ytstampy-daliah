import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    @staticmethod
    def prepare(features_df, params):
        """
        Prepare features for analysis by scaling and applying dimensionality reduction.
        
        Uses scikit-learn's StandardScaler and PCA/UMAP for better consistency and performance.
        """
        feature_cols = [col for col in features_df.columns if col != 'time' and not col.endswith('_cluster')]
        features = features_df[feature_cols].values
        times = features_df['time'].values
        
        # Use scikit-learn's StandardScaler instead of manual scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        dim_reduction = params.get('dim_reduction', 'pca')
        verbose = params.get('verbose', True)

        features_reduced = features_scaled  # Default: no reduction
        reduction_method = None

        if dim_reduction == 'pca':
            # Use scikit-learn's PCA with variance coverage threshold instead of fixed n_components
            # Get the desired variance coverage from params (default to 0.95 if not specified)
            variance_coverage = params.get('pca_variance_coverage', 0.95)
            
            # Ensure variance_coverage is between 0 and 1
            variance_coverage = max(0.1, min(0.99, variance_coverage))
            
            pca = PCA(n_components=variance_coverage)
            features_reduced = pca.fit_transform(features_scaled)
            
            if verbose:
                explained_variance = pca.explained_variance_ratio_.sum()
                n_components_used = pca.n_components_
                print(f"PCA automatically selected {n_components_used} components to capture {explained_variance:.2%} of variance")
                print(f"Target variance coverage was {variance_coverage:.2%}")
                
            reduction_method = "PCA"
            
        elif dim_reduction == 'umap':
            try:
                import umap
                # For UMAP, we still need to specify n_components
                n_components = min(params.get('n_components', 10), features.shape[1])
                reducer = umap.UMAP(n_components=n_components)
                features_reduced = reducer.fit_transform(features_scaled)
                reduction_method = "UMAP"
            except ImportError:
                if verbose:
                    print("UMAP not available, falling back to PCA")
                # Fallback to PCA if UMAP is not available
                variance_coverage = params.get('pca_variance_coverage', 0.95)
                variance_coverage = max(0.1, min(0.99, variance_coverage))
                pca = PCA(n_components=variance_coverage)
                features_reduced = pca.fit_transform(features_scaled)
                reduction_method = "PCA (fallback from UMAP)"

        if verbose:
            if reduction_method:
                print(f"Reduced features from {features_scaled.shape[1]} to {features_reduced.shape[1]} dimensions using {reduction_method}")
            else:
                print(f"Using {features_scaled.shape[1]} original features without reduction")

        return features_reduced, times

    @staticmethod
    def get_reference_info(features, times, singing_ref, non_singing_ref):
        """
        Extract reference information from the provided reference segments.
        """
        sing_start, sing_end = singing_ref
        non_sing_start, non_sing_end = non_singing_ref
        
        singing_ref_mask = (times >= sing_start) & (times <= sing_end)
        non_singing_ref_mask = (times >= non_sing_start) & (times <= non_sing_end)
        
        singing_ref_indices = np.where(singing_ref_mask)[0]
        non_singing_ref_indices = np.where(non_singing_ref_mask)[0]
        
        # Calculate centroids for reference segments
        singing_centroid = np.mean(features[singing_ref_indices], axis=0).reshape(1, -1)
        non_singing_centroid = np.mean(features[non_singing_ref_indices], axis=0).reshape(1, -1)
        
        return singing_ref_indices, non_singing_ref_indices, singing_centroid, non_singing_centroid

    @staticmethod
    def run_clustering(features, singing_centroid, non_singing_centroid, params):
        """
        Run KMeans clustering to identify singing and non-singing segments.
        
        Uses scikit-learn's KMeans instead of scipy's kmeans2 for better consistency with
        the rest of the scikit-learn ecosystem.
        """
        n_clusters = 2
        initial_centroids = np.vstack([singing_centroid, non_singing_centroid])
        
        # Use scikit-learn's KMeans instead of scipy's kmeans2
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=initial_centroids,
            n_init=1,  # Use the provided centroids exactly
            max_iter=100,
            random_state=42
        )
        
        # Fit and get cluster labels
        cluster_labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_
        
        # Determine which cluster corresponds to singing
        singing_ref_mask = (np.linalg.norm(features - singing_centroid, axis=1) < 
                          np.linalg.norm(features - non_singing_centroid, axis=1))
        
        # Count votes for each cluster in the areas we expect to be singing/non-singing
        singing_cluster_votes = np.bincount(cluster_labels[singing_ref_mask], minlength=n_clusters)
        non_singing_cluster_votes = np.bincount(cluster_labels[~singing_ref_mask], minlength=n_clusters)
        
        # Calculate ratios to determine which cluster is more associated with singing
        singing_ratio = singing_cluster_votes / (np.sum(singing_cluster_votes) + 1e-8)
        non_singing_ratio = non_singing_cluster_votes / (np.sum(non_singing_cluster_votes) + 1e-8)
        
        # Calculate a score for each cluster based on its association with singing vs non-singing
        singing_scores = singing_ratio / (non_singing_ratio + 0.001)
        singing_cluster = np.argmax(singing_scores)
        
        if params.get('verbose', True):
            print(f"Identified cluster {singing_cluster} as likely singing cluster")
            print(f"Singing score: {singing_scores[singing_cluster]:.2f}")
            
        return cluster_labels, singing_cluster 