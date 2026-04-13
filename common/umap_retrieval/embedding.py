"""UMAP projection and KMeans clustering."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import umap as umap_lib
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from umap_retrieval.config import SEED


def run_umap(
    embeddings: Dict[str, np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Project slide embeddings to 2D using UMAP with cosine distance.

    Args:
        embeddings: Mapping of slide name to 1536-d numpy array.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns slide_name, umap_x, umap_y.
    """
    names = list(embeddings.keys())
    matrix = np.stack([embeddings[n] for n in names])

    reducer = umap_lib.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(matrix)

    return pd.DataFrame({
        "slide_name": names,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
    })


def run_umap_3d(
    embeddings: Dict[str, np.ndarray],
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = SEED,
) -> pd.DataFrame:
    """Project slide embeddings to 3D using UMAP with cosine distance.

    Args:
        embeddings: Mapping of slide name to 1536-d numpy array.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with columns slide_name, umap_x, umap_y, umap_z.
    """
    names = list(embeddings.keys())
    matrix = np.stack([embeddings[n] for n in names])

    reducer = umap_lib.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    coords = reducer.fit_transform(matrix)

    return pd.DataFrame({
        "slide_name": names,
        "umap_x": coords[:, 0],
        "umap_y": coords[:, 1],
        "umap_z": coords[:, 2],
    })


def assign_clusters_by_metadata(
    umap_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_by: str = "primary_tumor_site",
    n_clusters: int = 5,
    seed: int = SEED,
) -> pd.DataFrame:
    """Run KMeans clustering on 2D UMAP coordinates and annotate each cluster
    with the dominant value of a metadata column.

    This performs unsupervised clustering (KMeans on the 2D UMAP layout) and
    then labels each discovered cluster by the most common ``color_by`` value
    among its members.  This is distinct from simply coloring by metadata: the
    cluster boundaries are determined entirely by embedding geometry, and the
    metadata label is added afterwards for interpretability.

    Args:
        umap_df: DataFrame with columns slide_name, umap_x, umap_y.
        metadata_df: DataFrame with slide_name and the ``color_by`` column.
        color_by: Metadata column used to annotate clusters after assignment.
        n_clusters: Number of KMeans clusters.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with added ``cluster`` (int) and ``cluster_label`` (str)
        columns, where ``cluster_label`` is the dominant ``color_by`` value
        within each KMeans cluster.
    """
    coords = umap_df[["umap_x", "umap_y"]].values
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=15)
    cluster_ids = km.fit_predict(coords)

    result = umap_df.copy()
    result["cluster"] = cluster_ids

    # Annotate each cluster with the dominant metadata value
    merged = result.merge(
        metadata_df[["slide_name", color_by]], on="slide_name", how="left"
    )
    merged[color_by] = merged[color_by].fillna("unknown")

    dominant = (
        merged.groupby("cluster")[color_by]
        .agg(lambda x: x.value_counts().idxmax())
        .rename("cluster_label")
    )
    result = result.merge(dominant, on="cluster", how="left")

    summary = result.groupby(["cluster", "cluster_label"]).size().reset_index(name="count")
    print(f"KMeans clustering ({n_clusters} clusters) annotated by '{color_by}':")
    for _, row in summary.iterrows():
        print(f"  Cluster {row['cluster']} (dominant: {row['cluster_label']}): {row['count']} slides")

    return result


def assign_clusters(
    umap_df: pd.DataFrame,
    n_clusters: int = 3,
    seed: int = SEED,
) -> pd.DataFrame:
    """Assign cluster labels using KMeans on 2D UMAP coordinates.

    Args:
        umap_df: DataFrame with columns slide_name, umap_x, umap_y.
        n_clusters: Number of clusters for KMeans.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with an added 'cluster' column containing integer labels.
    """
    coords = umap_df[["umap_x", "umap_y"]].values
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=15)
    labels = km.fit_predict(coords)
    result = umap_df.copy()
    result["cluster"] = labels

    summary = result["cluster"].value_counts().sort_index()
    print(f"Cluster assignment complete ({n_clusters} clusters):")
    for cluster_id, count in summary.items():
        print(f"  Cluster {cluster_id}: {count} slides")

    return result


def select_n_clusters(
    features: np.ndarray,
    k_range: range = range(2, 11),
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = SEED,
    show_plot: bool = True,
) -> Tuple[int, List[float], List[float], np.ndarray]:
    """Select optimal K for KMeans via UMAP projection + silhouette scoring.

    Projects patch features to 2D with UMAP (cosine metric, L2-normalised),
    then sweeps k_range computing inertia and silhouette score for each K.
    Returns the K with the highest silhouette score and optionally shows the
    silhouette plot.

    Args:
        features: (N, D) float array of patch embeddings.
        k_range: Range of K values to evaluate.
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        seed: Random seed for reproducibility.
        show_plot: Whether to display the silhouette score plot.

    Returns:
        Tuple of (best_k, inertias, silhouettes, umap_xy) where umap_xy is
        the (N, 2) UMAP projection used for scoring.
    """
    features_norm = normalize(features, norm="l2")
    reducer = umap_lib.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
    )
    umap_xy = reducer.fit_transform(features_norm)
    print(f"UMAP projection complete: {umap_xy.shape}")

    inertias: List[float] = []
    silhouettes: List[float] = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(umap_xy)
        inertias.append(km.inertia_)
        silhouettes.append(
            silhouette_score(umap_xy, labels, sample_size=min(1000, len(umap_xy)))
        )

    best_k = list(k_range)[silhouettes.index(max(silhouettes))]
    print(f"Best K by silhouette: {best_k}")

    if show_plot:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(k_range), y=silhouettes, mode="lines+markers", name="Silhouette"
        ))
        fig.update_layout(
            title=f"Silhouette Score (patch UMAP, {features.shape[0]} patches)",
            xaxis_title="K",
            yaxis_title="Score",
        )
        fig.show()

    return best_k, inertias, silhouettes, umap_xy


def cluster_embeddings(
    embeddings: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    k_range: range = range(2, 11),
    cross_tabulate_cols: Optional[List[str]] = None,
    seed: int = SEED,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster patients in the original 1536-d embedding space and cross-tabulate
    the best-K labels against clinical metadata.

    Embeddings are mean-pooled per patient before clustering, consistent with
    embedding_silhouette_scores. This avoids inflating silhouette scores with
    near-identical same-patient slides.

    Clustering is performed on L2-normalised embeddings with cosine-equivalent
    KMeans (normalise + euclidean ≡ cosine KMeans). Silhouette scores are
    computed in the same normalised space with cosine metric to select K.

    Args:
        embeddings: Mapping of slide_name to 1536-d numpy array.
        metadata_df: DataFrame with slide_name and metadata columns.
        k_range: Range of K values to sweep.
        cross_tabulate_cols: Metadata columns to cross-tabulate against cluster
            labels. Defaults to primary_tumor_site, histologic_type,
            hpv_association_p16.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of:
        - silhouette_df: DataFrame with columns [k, silhouette_score] for all
          K values tried, sorted by k.
        - cluster_df: one row per patient with columns [patient_id, cluster]
          plus the cross_tabulate_cols, with printed cross-tabulation tables.
    """
    from umap_retrieval.retrieval import aggregate_embeddings_by_patient, _patient_id_from_slide

    if cross_tabulate_cols is None:
        cross_tabulate_cols = [
            "primary_tumor_site",
            "histologic_type",
            "hpv_association_p16",
        ]

    # Mean-pool to one vector per patient
    patient_embs = aggregate_embeddings_by_patient(embeddings)
    patient_to_slide: Dict[str, str] = {}
    for slide_name in embeddings:
        pid = _patient_id_from_slide(slide_name) or "__unknown__"
        if pid not in patient_to_slide:
            patient_to_slide[pid] = slide_name

    pids = list(patient_embs.keys())
    matrix = np.stack([patient_embs[p] for p in pids])
    normed = normalize(matrix, norm="l2")
    meta = metadata_df.set_index("slide_name")

    if verbose:
        print(f"Sweeping K={list(k_range)} on {len(pids)} patients (1536-d, cosine)...")
    sil_rows = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=seed, n_init="auto").fit_predict(normed)
        sil = silhouette_score(normed, labels, metric="cosine")
        sil_rows.append({"k": k, "silhouette_score": round(float(sil), 4)})
        if verbose:
            print(f"  K={k}: silhouette={sil:.4f}")

    silhouette_df = pd.DataFrame(sil_rows)
    best_k = int(silhouette_df.loc[silhouette_df["silhouette_score"].idxmax(), "k"])
    if verbose:
        print(f"\nBest K by silhouette: {best_k}")

    best_labels = KMeans(n_clusters=best_k, random_state=seed, n_init="auto").fit_predict(normed)

    rows = []
    for pid, label in zip(pids, best_labels):
        slide = patient_to_slide.get(pid)
        row: Dict = {"patient_id": pid, "cluster": int(label)}
        for col in cross_tabulate_cols:
            row[col] = (
                str(meta.loc[slide, col])
                if slide and slide in meta.index and col in meta.columns
                else "unknown"
            )
        rows.append(row)
    cluster_df = pd.DataFrame(rows)

    if verbose:
        print(f"\nCluster sizes (K={best_k}):")
        print(cluster_df["cluster"].value_counts().sort_index().to_string())
        for col in cross_tabulate_cols:
            ct = pd.crosstab(cluster_df["cluster"], cluster_df[col])
            print(f"\nCross-tabulation: cluster × {col}")
            print(ct.to_string())

    return silhouette_df, cluster_df
