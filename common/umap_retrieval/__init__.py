"""UMAP Slide Clustering & KNN Retrieval package.

Provides dimensionality reduction, clustering, nearest-neighbor retrieval,
metadata filtering, and interactive visualization for slide embeddings.
"""

from umap_retrieval.config import (
    FILTERABLE_COLUMNS,
    NONFILTERABLE_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
    METADATA_COLUMNS,
    SEED,
)
from umap_retrieval.data import (
    load_and_validate_csv,
    extract_metadata_and_embeddings,
    apply_metadata_filters,
    fetch_patch_h5_files,
    fetch_all_h5_files,
    load_patch_features,
)
from umap_retrieval.embedding import run_umap, run_umap_3d, assign_clusters, assign_clusters_by_metadata, select_n_clusters, cluster_embeddings
from umap_retrieval.retrieval import (
    find_k_nearest,
    compare_umap_vs_cosine_neighbors,
    aggregate_embeddings_by_patient,
    embedding_silhouette_scores,
    cosine_similarity_null_distribution,
    format_knn_results,
    composite_patient_similarity,
    rank_patients_by_composite,
)
from umap_retrieval.composition import (
    build_composition_profiles,
    build_composition_profiles_from_metadata,
    sample_patch_features,
    profile_slide,
    patient_composition_profile,
    build_patient_profiles,
    composition_silhouette,
    save_profiles,
    load_profiles,
    update_profiles,
)
from umap_retrieval.visualization import (
    plot_cluster_bars,
    plot_umap_interactive,
    plot_umap_3d,
    plot_knn_radial_3d,
    plot_umap_vs_cosine,
    query_by_patient_id,
    plot_clustering_results,
)

__all__ = [
    "FILTERABLE_COLUMNS",
    "NONFILTERABLE_COLUMNS",
    "REQUIRED_METADATA_COLUMNS",
    "METADATA_COLUMNS",
    "SEED",
    "load_and_validate_csv",
    "extract_metadata_and_embeddings",
    "apply_metadata_filters",
    "fetch_patch_h5_files",
    "fetch_all_h5_files",
    "load_patch_features",
    "run_umap",
    "run_umap_3d",
    "assign_clusters",
    "assign_clusters_by_metadata",
    "select_n_clusters",
    "cluster_embeddings",
    "find_k_nearest",
    "compare_umap_vs_cosine_neighbors",
    "aggregate_embeddings_by_patient",
    "embedding_silhouette_scores",
    "cosine_similarity_null_distribution",
    "format_knn_results",
    "composite_patient_similarity",
    "rank_patients_by_composite",
    "build_composition_profiles",
    "build_composition_profiles_from_metadata",
    "sample_patch_features",
    "profile_slide",
    "patient_composition_profile",
    "build_patient_profiles",
    "composition_silhouette",
    "save_profiles",
    "load_profiles",
    "update_profiles",
    "plot_cluster_bars",
    "plot_umap_interactive",
    "plot_umap_3d",
    "plot_knn_radial_3d",
    "plot_umap_vs_cosine",
    "query_by_patient_id",
    "plot_clustering_results",
]
