"""Interactive Plotly visualizations for UMAP projections and KNN results."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

from umap_retrieval.config import FILTERABLE_COLUMNS
from umap_retrieval.data import apply_metadata_filters
from umap_retrieval.retrieval import find_k_nearest


def _bin_age(age_value) -> str:
    """Bin an age value into a decade-based group."""
    try:
        age = int(float(age_value))
    except (ValueError, TypeError):
        return "Unknown"
    if age >= 80:
        return "80+"
    decade_start = (age // 10) * 10
    return f"{decade_start}-{decade_start + 9}"


def _build_color_map(unique_groups: list[str]) -> dict[str, str]:
    """Build a color mapping, reserving gray for Unknown."""
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map = {}
    color_idx = 0
    for group in unique_groups:
        if group == "Unknown":
            color_map[group] = "lightgray"
        else:
            color_map[group] = palette[color_idx % len(palette)]
            color_idx += 1
    return color_map


def _prepare_plot_df(
    umap_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_by: str,
    metadata_filters: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """Merge UMAP coords with metadata, apply filters, compute color groups."""
    plot_df = umap_df.merge(metadata_df, on="slide_name", how="inner")

    if metadata_filters:
        filtered_meta = apply_metadata_filters(
            metadata_df, metadata_filters, FILTERABLE_COLUMNS
        )
        filtered_names = set(filtered_meta["slide_name"])
        plot_df = plot_df[plot_df["slide_name"].isin(filtered_names)].copy()
        print(f"{len(plot_df)} slides remaining after filtering")

    plot_df = plot_df.copy()
    if color_by == "age_at_initial_diagnosis":
        plot_df["_color_group"] = plot_df[color_by].apply(_bin_age)
    else:
        plot_df["_color_group"] = plot_df[color_by].fillna("unknown").astype(str)
        plot_df.loc[plot_df["_color_group"] == "unknown", "_color_group"] = "Unknown"

    return plot_df


def plot_umap_interactive(
    umap_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_by: str = "primary_tumor_site",
    query_slide: Optional[str] = None,
    neighbors: Optional[List[str]] = None,
    metadata_filters: Optional[Dict[str, List[str]]] = None,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
    k: int = 15,
    title: Optional[str] = None,
) -> go.Figure:
    """Render an interactive Plotly scatter plot of the 2D UMAP projection.

    Colors points by the specified metadata field. Supports categorical
    coloring and decade-based age binning. Highlights query slide (star)
    and KNN neighbors (diamonds) when provided.

    When both `embeddings` and `metadata_filters` are supplied and
    `neighbors` is not pre-computed, KNN is run against the filtered
    embedding subset so that only slides passing the filter are considered
    as candidates.

    Args:
        umap_df: DataFrame with slide_name, umap_x, umap_y columns.
        metadata_df: DataFrame with slide_name and metadata columns.
        color_by: Metadata field name to color points by.
        query_slide: Slide name to highlight as query (star marker).
        neighbors: Pre-computed neighbor slide names (diamond markers).
            When None and both query_slide and embeddings are provided,
            KNN is computed automatically against the filtered pool.
        metadata_filters: Active filters — restricts both the displayed
            points AND the KNN candidate pool.
        embeddings: Full embedding dict. Required for automatic KNN
            computation when neighbors is None.
        k: Number of neighbors to retrieve when computing KNN automatically.
        title: Optional plot title override. Defaults to
            "UMAP — colored by {color_by}".

    Returns:
        plotly.graph_objects.Figure
    """
    plot_df = _prepare_plot_df(umap_df, metadata_df, color_by, metadata_filters)

    # Auto-compute neighbors against the filtered pool when not pre-supplied
    if query_slide and neighbors is None and embeddings is not None:
        filtered_names = set(plot_df["slide_name"])
        # Always include the query slide in the pool even if filtered out
        filtered_names.add(query_slide)
        candidate_embeddings = {
            name: vec for name, vec in embeddings.items()
            if name in filtered_names
        }
        if metadata_filters:
            print(
                f"KNN restricted to {len(candidate_embeddings)} filtered candidates "
                f"(filters: {metadata_filters})"
            )
        try:
            knn_results = find_k_nearest(
                embeddings, query_slide, k=k,
                candidate_embeddings=candidate_embeddings,
            )
            neighbors = [name for name, _ in knn_results]
        except (KeyError, ValueError) as e:
            print(f"KNN skipped: {e}")
            neighbors = []
    unique_groups = sorted(plot_df["_color_group"].unique())
    color_map = _build_color_map(unique_groups)

    fig = go.Figure()
    query_set = {query_slide} if query_slide else set()
    neighbor_set = set(neighbors) if neighbors else set()

    def _hover_text(row: pd.Series, prefix: str = "") -> str:
        """Build tooltip text with all required metadata fields."""
        parts = [
            f"{prefix}{row['slide_name']}",
            f"{color_by}: {row['_color_group']}",
        ]
        # Always include the core metadata fields per requirement 2.7
        for col, label in [
            ("primary_tumor_site", "Tumor Site"),
            ("histologic_type", "Histologic Type"),
            ("hpv_association_p16", "HPV Status"),
            ("age_at_initial_diagnosis", "Age at Diagnosis"),
            ("year_of_initial_diagnosis", "Year of Diagnosis"),
        ]:
            # Skip if this field is already shown as color_by to avoid duplication
            if col == color_by:
                continue
            val = row.get(col, "N/A")
            if val is None or (isinstance(val, float) and pd.isna(val)):
                val = "N/A"
            parts.append(f"{label}: {val}")
        return "<br>".join(parts)

    for group in unique_groups:
        group_df = plot_df[plot_df["_color_group"] == group]
        regular = group_df[~group_df["slide_name"].isin(query_set | neighbor_set)]
        if len(regular) > 0:
            fig.add_trace(go.Scatter(
                x=regular["umap_x"], y=regular["umap_y"],
                mode="markers",
                marker=dict(size=7, color=color_map[group]),
                name=str(group),
                text=[_hover_text(row) for _, row in regular.iterrows()],
                hoverinfo="text",
                legendgroup=group,
            ))

    if neighbors:
        nbr_df = plot_df[plot_df["slide_name"].isin(neighbor_set)]
        if len(nbr_df) > 0:
            fig.add_trace(go.Scatter(
                x=nbr_df["umap_x"], y=nbr_df["umap_y"],
                mode="markers",
                marker=dict(size=12, symbol="diamond", color="orange",
                            line=dict(width=1, color="black")),
                name="KNN Neighbor",
                text=[_hover_text(row) for _, row in nbr_df.iterrows()],
                hoverinfo="text",
            ))

    if query_slide:
        q_df = plot_df[plot_df["slide_name"] == query_slide]
        if len(q_df) > 0:
            fig.add_trace(go.Scatter(
                x=q_df["umap_x"], y=q_df["umap_y"],
                mode="markers",
                marker=dict(size=16, symbol="star", color="red",
                            line=dict(width=1, color="black")),
                name="Query Slide",
                text=[_hover_text(row, prefix="QUERY: ") for _, row in q_df.iterrows()],
                hoverinfo="text",
            ))

    base_title = title if title is not None else f"UMAP — colored by {color_by}"
    subtitle = ""
    if metadata_filters:
        filter_parts = [f"{k}: {', '.join(v)}" for k, v in metadata_filters.items()]
        subtitle = f"<br><sub>Filters: {' | '.join(filter_parts)}</sub>"

    fig.update_layout(
        title=base_title + subtitle,
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        legend_title=color_by, template="plotly_white",
        width=900, height=650,
    )
    return fig

def plot_cluster_bars(
    umap_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_by: str = "primary_tumor_site",
) -> go.Figure:
    """Bar chart showing metadata composition of each UMAP cluster.

    Each cluster gets a group of stacked bars (one color per metadata
    category), making it easy to see which clinical attributes dominate
    each cluster.

    Args:
        umap_df: DataFrame with slide_name and cluster columns.
        color_by: Metadata field to break down by.
        metadata_df: DataFrame with slide_name and metadata columns.

    Returns:
        plotly.graph_objects.Figure
    """
    cluster_cols = ["slide_name", "cluster"]
    if "cluster_label" in umap_df.columns:
        cluster_cols.append("cluster_label")

    merged = umap_df[cluster_cols].merge(
        metadata_df[["slide_name", color_by]], on="slide_name", how="inner",
    )
    if color_by == "age_at_initial_diagnosis":
        merged["_group"] = merged[color_by].apply(_bin_age)
    else:
        merged["_group"] = merged[color_by].fillna("unknown").astype(str)
        merged.loc[merged["_group"] == "unknown", "_group"] = "Unknown"

    # Always use "Cluster N (dominant: label)" on x-axis so the axis represents
    # geometry-derived cluster identity, not the metadata category itself.
    if "cluster_label" in merged.columns:
        cluster_index_map = (
            merged[["cluster", "cluster_label"]]
            .drop_duplicates()
            .set_index("cluster")["cluster_label"]
        )
        x_labels = lambda idx: [
            f"Cluster {c} ({cluster_index_map.get(c, '?')})" for c in idx
        ]
    else:
        x_labels = lambda idx: [f"Cluster {c}" for c in idx]

    ct = pd.crosstab(merged["cluster"], merged["_group"])
    unique_groups = sorted(ct.columns)
    color_map = _build_color_map(unique_groups)

    fig = go.Figure()
    for group in unique_groups:
        fig.add_trace(go.Bar(
            x=x_labels(ct.index),
            y=ct[group],
            name=str(group),
            marker_color=color_map[group],
        ))

    fig.update_layout(
        barmode="stack",
        title=f"Cluster Composition — by {color_by}",
        xaxis_title="Cluster (dominant metadata value)",
        yaxis_title="Slide Count",
        legend_title=color_by,
        template="plotly_white",
        width=900,
        height=500,
    )
    return fig




def plot_umap_3d(
    umap3d_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_by: str = "primary_tumor_site",
    query_slide: Optional[str] = None,
    neighbors: Optional[List[str]] = None,
    metadata_filters: Optional[Dict[str, List[str]]] = None,
) -> go.Figure:
    """Render an interactive 3D Plotly scatter of the UMAP projection."""
    plot_df = _prepare_plot_df(umap3d_df, metadata_df, color_by, metadata_filters)
    unique_groups = sorted(plot_df["_color_group"].unique())
    color_map = _build_color_map(unique_groups)

    fig = go.Figure()
    query_set = {query_slide} if query_slide else set()
    neighbor_set = set(neighbors) if neighbors else set()

    for group in unique_groups:
        group_df = plot_df[plot_df["_color_group"] == group]
        regular = group_df[~group_df["slide_name"].isin(query_set | neighbor_set)]
        if len(regular) > 0:
            fig.add_trace(go.Scatter3d(
                x=regular["umap_x"], y=regular["umap_y"], z=regular["umap_z"],
                mode="markers",
                marker=dict(size=4, color=color_map[group]),
                name=str(group),
                text=[
                    f"{row['slide_name']}<br>{color_by}: {row['_color_group']}"
                    for _, row in regular.iterrows()
                ],
                hoverinfo="text", legendgroup=group,
            ))

    if neighbors:
        nbr_df = plot_df[plot_df["slide_name"].isin(neighbor_set)]
        if len(nbr_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=nbr_df["umap_x"], y=nbr_df["umap_y"], z=nbr_df["umap_z"],
                mode="markers",
                marker=dict(size=7, symbol="diamond", color="orange",
                            line=dict(width=1, color="black")),
                name="KNN Neighbor",
                text=[
                    f"{row['slide_name']}<br>{color_by}: {row['_color_group']}"
                    for _, row in nbr_df.iterrows()
                ],
                hoverinfo="text",
            ))

    if query_slide:
        q_df = plot_df[plot_df["slide_name"] == query_slide]
        if len(q_df) > 0:
            fig.add_trace(go.Scatter3d(
                x=q_df["umap_x"], y=q_df["umap_y"], z=q_df["umap_z"],
                mode="markers",
                marker=dict(size=10, symbol="diamond", color="red",
                            line=dict(width=1, color="black")),
                name="Query Slide",
                text=[
                    f"QUERY: {row['slide_name']}<br>{color_by}: {row['_color_group']}"
                    for _, row in q_df.iterrows()
                ],
                hoverinfo="text",
            ))

    title = f"3D UMAP — colored by {color_by}"
    subtitle = ""
    if metadata_filters:
        filter_parts = [f"{k}: {', '.join(v)}" for k, v in metadata_filters.items()]
        subtitle = f"<br><sub>Filters: {' | '.join(filter_parts)}</sub>"

    fig.update_layout(
        title=title + subtitle,
        scene=dict(xaxis_title="UMAP 1", yaxis_title="UMAP 2", zaxis_title="UMAP 3"),
        legend_title=color_by, template="plotly_white",
        width=950, height=700,
    )
    return fig


def plot_knn_radial_3d(
    embeddings: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    query_slide: str,
    knn_results: List[Tuple[str, float]],
    color_by: str = "histologic_type",
) -> go.Figure:
    """Radial 3D plot with query slide at origin and neighbors around it.

    Uses PCA on the neighbor embeddings to derive 3 spatial axes,
    then scales each neighbor's position by (1 - cosine_similarity)
    so more similar slides are closer to the center.
    """
    neighbor_names = [name for name, _ in knn_results]
    neighbor_sims = {name: sim for name, sim in knn_results}

    query_vec = embeddings[query_slide]
    diff_matrix = np.stack([embeddings[name] - query_vec for name in neighbor_names])

    n_components = min(3, len(neighbor_names))
    pca = PCA(n_components=n_components, random_state=42)
    coords_raw = pca.fit_transform(diff_matrix)

    if n_components < 3:
        pad = np.zeros((coords_raw.shape[0], 3 - n_components))
        coords_raw = np.hstack([coords_raw, pad])

    norms = np.linalg.norm(coords_raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    directions = coords_raw / norms

    coords_3d = np.zeros((len(neighbor_names), 3))
    for i, name in enumerate(neighbor_names):
        distance = 1.0 - neighbor_sims[name]
        coords_3d[i] = directions[i] * distance

    meta_lookup = metadata_df.set_index("slide_name")

    all_names = [query_slide] + neighbor_names
    color_values = [
        str(meta_lookup.loc[n, color_by]) if n in meta_lookup.index else "Unknown"
        for n in all_names
    ]
    unique_groups = sorted(set(color_values))
    color_map = _build_color_map(unique_groups)

    fig = go.Figure()

    # Lines from origin to each neighbor
    for i, name in enumerate(neighbor_names):
        fig.add_trace(go.Scatter3d(
            x=[0, coords_3d[i, 0]], y=[0, coords_3d[i, 1]], z=[0, coords_3d[i, 2]],
            mode="lines", line=dict(color="lightgray", width=2),
            showlegend=False, hoverinfo="skip",
        ))

    # Neighbors colored by metadata group
    for group in unique_groups:
        idxs = [
            i for i, name in enumerate(neighbor_names)
            if str(meta_lookup.loc[name, color_by]) == group
        ]
        if not idxs:
            continue
        fig.add_trace(go.Scatter3d(
            x=[coords_3d[i, 0] for i in idxs],
            y=[coords_3d[i, 1] for i in idxs],
            z=[coords_3d[i, 2] for i in idxs],
            mode="markers",
            marker=dict(size=6, color=color_map[group], symbol="diamond",
                        line=dict(width=1, color="black")),
            name=group,
            text=[
                f"{neighbor_names[i]}<br>{color_by}: {group}<br>"
                f"similarity: {neighbor_sims[neighbor_names[i]]:.4f}"
                for i in idxs
            ],
            hoverinfo="text", legendgroup=group,
        ))

    # Query at origin
    q_color_val = (
        str(meta_lookup.loc[query_slide, color_by])
        if query_slide in meta_lookup.index else "Unknown"
    )
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0], mode="markers",
        marker=dict(size=12, symbol="diamond", color="red",
                    line=dict(width=2, color="black")),
        name=f"Query: {query_slide.split('_')[-1]}",
        text=[f"QUERY: {query_slide}<br>{color_by}: {q_color_val}"],
        hoverinfo="text",
    ))

    var_explained = pca.explained_variance_ratio_
    fig.update_layout(
        title=(
            f"Radial KNN 3D — {query_slide.split('_')[-1]} as centroid<br>"
            f"<sub>colored by {color_by} | K={len(knn_results)} | "
            f"PCA variance: {var_explained[0]:.1%}, {var_explained[1]:.1%}, "
            f"{var_explained[2]:.1%}</sub>"
        ),
        scene=dict(
            xaxis_title=f"PC1 ({var_explained[0]:.0%})",
            yaxis_title=f"PC2 ({var_explained[1]:.0%})",
            zaxis_title=f"PC3 ({var_explained[2]:.0%})",
        ),
        legend_title=color_by, template="plotly_white",
        width=950, height=700,
    )
    return fig


def _plot_group(fig, plot_df, name_set, comparison_df, color, name, symbol):
    """Helper to add a neighbor group trace with similarity hover text."""
    if not name_set:
        return
    sub = plot_df[plot_df["slide_name"].isin(name_set)]
    sim_map = dict(zip(comparison_df["slide_name"], comparison_df["cosine_similarity"]))
    fig.add_trace(go.Scatter(
        x=sub["umap_x"], y=sub["umap_y"],
        mode="markers",
        marker=dict(size=11, symbol=symbol, color=color,
                    line=dict(width=1, color="black")),
        name=name,
        text=[
            f"{row['slide_name']}<br>cosine sim: {sim_map.get(row['slide_name'], 0):.4f}"
            for _, row in sub.iterrows()
        ],
        hoverinfo="text",
    ))


def plot_umap_vs_cosine(
    query_slide: str,
    comparison_df: pd.DataFrame,
    umap_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    color_by: str = "primary_tumor_site",
) -> go.Figure:
    """Plot UMAP scatter highlighting cosine-only, umap-only, and shared neighbors.

    - Green diamonds: in both UMAP and cosine KNN (trustworthy)
    - Orange diamonds: UMAP-only (misleading visual proximity)
    - Blue diamonds: cosine-only (true neighbors UMAP placed far away)
    """
    plot_df = umap_df.merge(metadata_df, on="slide_name", how="inner")

    both_names = set(comparison_df[comparison_df["source"] == "both"]["slide_name"])
    cosine_only_names = set(
        comparison_df[comparison_df["source"] == "cosine_only"]["slide_name"]
    )
    umap_only_names = set(
        comparison_df[comparison_df["source"] == "umap_only"]["slide_name"]
    )
    special = {query_slide} | both_names | cosine_only_names | umap_only_names

    fig = go.Figure()

    bg = plot_df[~plot_df["slide_name"].isin(special)]
    fig.add_trace(go.Scatter(
        x=bg["umap_x"], y=bg["umap_y"], mode="markers",
        marker=dict(size=5, color="lightgray", opacity=0.4),
        name="Other slides", hoverinfo="skip",
    ))

    _plot_group(fig, plot_df, both_names, comparison_df,
                color="green", name="Both (cosine + UMAP)", symbol="diamond")
    _plot_group(fig, plot_df, cosine_only_names, comparison_df,
                color="dodgerblue", name="Cosine-only (true KNN)", symbol="diamond")
    _plot_group(fig, plot_df, umap_only_names, comparison_df,
                color="orange", name="UMAP-only (misleading)", symbol="diamond")

    q_df = plot_df[plot_df["slide_name"] == query_slide]
    if len(q_df) > 0:
        fig.add_trace(go.Scatter(
            x=q_df["umap_x"], y=q_df["umap_y"], mode="markers",
            marker=dict(size=16, symbol="star", color="red",
                        line=dict(width=1, color="black")),
            name="Query Slide",
            text=[f"QUERY: {query_slide}"], hoverinfo="text",
        ))

    overlap = len(both_names)
    total = len(both_names) + len(cosine_only_names)
    fig.update_layout(
        title=(
            f"UMAP vs Cosine KNN — {query_slide.split('_')[-1]}"
            f"<br><sub>Overlap: {overlap}/{total} | "
            f"Green=both, Blue=cosine-only, Orange=UMAP-only</sub>"
        ),
        xaxis_title="UMAP 1", yaxis_title="UMAP 2",
        template="plotly_white", width=950, height=700,
    )
    return fig


def query_by_patient_id(
    patient_id: str,
    umap_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    embeddings: Dict[str, np.ndarray],
    color_by: str = "primary_tumor_site",
    k: int = 15,
) -> Optional[go.Figure]:
    """Highlight all slides for a patient and show KNN from their first slide.

    Args:
        patient_id: Numeric patient ID string (e.g. '492', '054').
        umap_df: UMAP projection DataFrame.
        metadata_df: Metadata DataFrame.
        embeddings: Embedding dictionary.
        color_by: Metadata field to color by.
        k: Number of nearest neighbors to find.

    Returns:
        Plotly Figure with patient slides highlighted, or None if not found.
    """
    pattern = f"patient{patient_id.zfill(3)}"
    patient_slides = [
        name for name in metadata_df["slide_name"] if pattern in name
    ]
    if not patient_slides:
        print(f"No slides found for patient {patient_id}")
        return None

    print(f"Patient {patient_id}: {len(patient_slides)} slide(s)")
    for s in patient_slides:
        print(f"  {s}")

    query_slide = patient_slides[0]
    knn_results = find_k_nearest(
        embeddings, query_slide, k=k, exclude_same_patient=True
    )
    neighbor_names = [name for name, score in knn_results]

    print(f"\nKNN from {query_slide} (k={k}):")
    for name, score in knn_results:
        print(f"  {name} (similarity: {score:.4f})")

    return plot_umap_interactive(
        umap_df, metadata_df,
        color_by=color_by,
        query_slide=query_slide,
        neighbors=neighbor_names,
    )


def plot_clustering_results(
    silhouette_df: pd.DataFrame,
    cluster_df: pd.DataFrame,
    cross_tabulate_cols: Optional[List[str]] = None,
) -> go.Figure:
    """Visualize unsupervised clustering results as a dashboard figure.

    Renders:
    - Top row: silhouette score bar chart with best-K highlighted
    - Bottom rows: one heatmap per cross-tabulation column showing
      cluster × metadata value counts

    Args:
        silhouette_df: DataFrame with columns [k, silhouette_score] from
            cluster_embeddings.
        cluster_df: DataFrame with columns [patient_id, cluster, ...metadata]
            from cluster_embeddings.
        cross_tabulate_cols: Metadata columns to show as heatmaps. Defaults
            to all non-id columns in cluster_df.

    Returns:
        plotly.graph_objects.Figure
    """
    from plotly.subplots import make_subplots

    if cross_tabulate_cols is None:
        cross_tabulate_cols = [
            c for c in cluster_df.columns
            if c not in ("patient_id", "cluster")
        ]

    n_heatmaps = len(cross_tabulate_cols)
    n_rows = 1 + n_heatmaps
    row_heights = [0.25] + [0.75 / n_heatmaps] * n_heatmaps

    subplot_titles = ["Silhouette Score by K"] + [
        f"Cluster × {col}" for col in cross_tabulate_cols
    ]

    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        vertical_spacing=0.08,
    )

    # --- Silhouette bar chart ---
    best_k = int(silhouette_df.loc[silhouette_df["silhouette_score"].idxmax(), "k"])
    colors = [
        "#1f77b4" if k != best_k else "#d62728"
        for k in silhouette_df["k"]
    ]
    fig.add_trace(
        go.Bar(
            x=silhouette_df["k"].astype(str),
            y=silhouette_df["silhouette_score"],
            marker_color=colors,
            text=[f"{s:.3f}" for s in silhouette_df["silhouette_score"]],
            textposition="outside",
            name="Silhouette",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=0, line_dash="dash", line_color="gray", line_width=1, row=1, col=1
    )
    fig.update_yaxes(title_text="Score", row=1, col=1)
    fig.update_xaxes(title_text="K", row=1, col=1)

    # --- Cross-tabulation heatmaps ---
    for i, col in enumerate(cross_tabulate_cols, start=2):
        if col not in cluster_df.columns:
            continue
        ct = pd.crosstab(cluster_df["cluster"], cluster_df[col])
        # Sort columns by total count descending for readability
        ct = ct[ct.sum().sort_values(ascending=False).index]

        fig.add_trace(
            go.Heatmap(
                z=ct.values,
                x=[str(c) for c in ct.columns],
                y=[f"Cluster {k}" for k in ct.index],
                colorscale="Blues",
                text=ct.values,
                texttemplate="%{text}",
                showscale=False,
                hovertemplate="Cluster %{y}<br>%{x}: %{z} patients<extra></extra>",
            ),
            row=i, col=1,
        )
        fig.update_xaxes(tickangle=-30, row=i, col=1)

    best_sil = float(silhouette_df.loc[silhouette_df["k"] == best_k, "silhouette_score"].iloc[0])
    n_patients = len(cluster_df)
    fig.update_layout(
        title=(
            f"Unsupervised Clustering — {n_patients} patients, 1536-d cosine<br>"
            f"<sub>Best K={best_k} (silhouette={best_sil:.3f}, highlighted red) | "
            f"Heatmap values = patient counts per cluster</sub>"
        ),
        height=300 + 280 * n_heatmaps,
        template="plotly_white",
        margin=dict(t=100, b=60),
    )
    return fig
