"""ClusterService: sweeps k-range, computes silhouette scores, returns Plotly figure JSON."""

from typing import Dict

import numpy as np

from umap_retrieval.data import apply_metadata_filters
from umap_retrieval.embedding import cluster_embeddings

from umap_retrieval.visualization import plot_clustering_results

from backend.models import ClusterParams, ClusterResponse


class ClusterService:
    """Runs K-sweep clustering on patient embeddings and returns a Plotly dashboard."""

    def run_clustering(self, params: ClusterParams, data_service) -> ClusterResponse:
        """Sweep k from k_min to k_max, compute silhouette scores, and build a Plotly figure.

        Applies metadata filters before clustering to restrict the slide pool.
        Uses mean-pooled patient embeddings in the original 1536-d space with
        cosine-equivalent KMeans (L2-normalise + euclidean).

        Args:
            params: ClusterParams with k_min, k_max, cross_tabulate_cols, filters.
            data_service: DataService instance providing embeddings and metadata.

        Returns:
            ClusterResponse with plotly_json, best_k, and best_silhouette.

        Raises:
            ValueError: If data is not loaded or filters are too restrictive.
        """
        embeddings: Dict[str, np.ndarray] = dict(data_service.get_embeddings())
        metadata_df = data_service.get_metadata()

        if embeddings is None or metadata_df is None:
            raise ValueError("Dataset not loaded. Call POST /api/data/load first.")

        # Apply metadata filters to restrict the slide pool (Requirement 3.3)
        if params.filters:
            filtered_meta = apply_metadata_filters(metadata_df, params.filters)
            filtered_names = set(filtered_meta["slide_name"])
            embeddings = {k: v for k, v in embeddings.items() if k in filtered_names}
            metadata_df = filtered_meta

        k_range = range(params.k_min, params.k_max + 1)

        # cluster_embeddings sweeps k, computes silhouette scores, returns best-K cluster labels
        # (Requirement 3.1, 3.2)
        silhouette_df, cluster_df = cluster_embeddings(
            embeddings=embeddings,
            metadata_df=metadata_df,
            k_range=k_range,
            cross_tabulate_cols=params.cross_tabulate_cols,
        )

        best_k = int(silhouette_df.loc[silhouette_df["silhouette_score"].idxmax(), "k"])
        best_silhouette = float(
            silhouette_df.loc[silhouette_df["k"] == best_k, "silhouette_score"].iloc[0]
        )

        fig = plot_clustering_results(
            silhouette_df=silhouette_df,
            cluster_df=cluster_df,
            cross_tabulate_cols=params.cross_tabulate_cols,
        )

        return ClusterResponse(
            plotly_json=fig.to_json(),
            best_k=best_k,
            best_silhouette=best_silhouette,
        )


# Module-level singleton shared by all routers
cluster_service = ClusterService()
