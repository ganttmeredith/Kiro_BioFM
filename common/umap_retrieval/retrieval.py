"""K-nearest-neighbor retrieval and UMAP vs cosine comparison."""

import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity


def _patient_id_from_slide(slide_name: str) -> Optional[str]:
    """Extract patient ID from a slide name (e.g. 'patient492' -> '492')."""
    m = re.search(r'patient(\d+)', slide_name)
    return m.group(1) if m else None


def aggregate_embeddings_by_patient(
    embeddings: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Mean-pool all slide embeddings per patient.

    Slides whose names contain no 'patientNNN' token are grouped under the
    key '__unknown__'.

    Returns:
        Dict mapping patient_id -> mean embedding vector.
    """
    buckets: Dict[str, List[np.ndarray]] = {}
    for slide_name, vec in embeddings.items():
        pid = _patient_id_from_slide(slide_name) or "__unknown__"
        buckets.setdefault(pid, []).append(vec)
    return {pid: np.mean(np.stack(vecs), axis=0) for pid, vecs in buckets.items()}


def find_k_nearest(
    embeddings: Dict[str, np.ndarray],
    query_slide: str,
    k: int,
    exclude_same_patient: bool = True,
    deduplicate_by_patient: bool = True,
    candidate_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> List[Tuple[str, float]]:
    """Find k nearest slides by cosine similarity in original embedding space.

    Args:
        embeddings: Full embedding dict (used to resolve the query vector and
            to define the candidate pool when candidate_embeddings is None).
        query_slide: Slide name to use as the query.
        k: Number of neighbors to return.
        exclude_same_patient: When True, slides sharing the same patientNNN
            token as the query are excluded from results. Defaults to True.
        deduplicate_by_patient: When True, only the highest-scoring slide per
            patient is kept in the final results. Defaults to True.
        candidate_embeddings: Optional restricted pool of embeddings to search
            against (e.g. a pre-filtered subset). When provided, KNN is run
            only over these candidates.

    Returns:
        List of (slide_name, similarity_score) sorted descending by similarity.

    Raises:
        KeyError: If query_slide not in embeddings.
        ValueError: If k < 1 or k >= number of valid candidates.
    """
    if query_slide not in embeddings:
        raise KeyError(f"Slide '{query_slide}' not found in embeddings")

    pool = candidate_embeddings if candidate_embeddings is not None else embeddings

    query_patient = _patient_id_from_slide(query_slide) if exclude_same_patient else None

    candidates = {
        name: vec for name, vec in pool.items()
        if name != query_slide
        and not (
            exclude_same_patient
            and query_patient is not None
            and _patient_id_from_slide(name) == query_patient
        )
    }

    n = len(candidates)
    if k < 1 or k > n:
        raise ValueError(f"k must be in [1, {n}], got {k}")

    names = list(candidates.keys())
    matrix = np.stack([candidates[name] for name in names])
    query_vec = embeddings[query_slide].reshape(1, -1)
    sims = cosine_similarity(query_vec, matrix).ravel()

    scored = [(name, float(sims[i])) for i, name in enumerate(names)]
    scored.sort(key=lambda x: x[1], reverse=True)

    if deduplicate_by_patient:
        seen_patients: set = set()
        deduped = []
        for name, sim in scored:
            pid = _patient_id_from_slide(name)
            key = pid if pid is not None else name
            if key not in seen_patients:
                seen_patients.add(key)
                deduped.append((name, sim))
        scored = deduped

    return scored[:k]


def compare_umap_vs_cosine_neighbors(
    query_slide: str,
    embeddings: Dict[str, np.ndarray],
    umap_df: pd.DataFrame,
    k: int = 15,
    exclude_same_patient: bool = True,
) -> pd.DataFrame:
    """Compare UMAP 2D neighbors with true cosine KNN in 1536-d space.

    Both neighbor searches are restricted to slides present in the
    embeddings dict, so filtered embedding subsets work correctly.

    Args:
        query_slide: Slide to use as the query.
        embeddings: Embedding dict (may be a pre-filtered subset).
        umap_df: DataFrame with umap_x, umap_y, slide_name columns.
        k: Number of neighbors.
        exclude_same_patient: Exclude same-patient slides from KNN results.

    Returns:
        DataFrame with one row per unique neighbor (union of both sets),
        showing UMAP rank, cosine rank, cosine similarity, UMAP distance,
        and which set(s) the slide belongs to.
    """
    embed_names = set(embeddings.keys())

    # Cosine KNN in original space (same-patient exclusion applied)
    cosine_knn = find_k_nearest(
        embeddings, query_slide, k=k, exclude_same_patient=exclude_same_patient
    )
    cosine_names = [name for name, _ in cosine_knn]

    # UMAP 2D neighbors (restricted to slides in embeddings)
    umap_subset = umap_df[umap_df["slide_name"].isin(embed_names)].copy()
    query_row = umap_subset[umap_subset["slide_name"] == query_slide]
    q_xy = query_row[["umap_x", "umap_y"]].values
    other = umap_subset[umap_subset["slide_name"] != query_slide].copy()
    dists = cdist(q_xy, other[["umap_x", "umap_y"]].values, metric="euclidean").ravel()
    other["umap_dist"] = dists
    other = other.sort_values("umap_dist")
    umap_neighbors = other.head(k)
    umap_names = umap_neighbors["slide_name"].tolist()
    umap_dists = dict(zip(umap_neighbors["slide_name"], umap_neighbors["umap_dist"]))

    # Compute cosine similarity for ALL slides in embeddings
    all_names = list(embeddings.keys())
    matrix = np.stack([embeddings[n] for n in all_names])
    query_vec = embeddings[query_slide].reshape(1, -1)
    all_sims = cosine_similarity(query_vec, matrix).ravel()
    full_sim_map = dict(zip(all_names, all_sims.tolist()))

    # Build comparison table
    cosine_set = set(cosine_names)
    umap_set = set(umap_names)
    union_names = list(dict.fromkeys(cosine_names + umap_names))
    rows = []
    for name in union_names:
        in_cosine = name in cosine_set
        in_umap = name in umap_set
        if in_cosine and in_umap:
            source = "both"
        elif in_cosine:
            source = "cosine_only"
        else:
            source = "umap_only"
        rows.append({
            "slide_name": name,
            "cosine_similarity": full_sim_map[name],
            "cosine_rank": (cosine_names.index(name) + 1) if in_cosine else None,
            "umap_distance": umap_dists.get(name, np.nan),
            "umap_rank": (umap_names.index(name) + 1) if in_umap else None,
            "source": source,
        })

    return pd.DataFrame(rows).sort_values("cosine_similarity", ascending=False)


def embedding_silhouette_scores(
    embeddings: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    label_columns: List[str],
    sample_n: Optional[int] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute silhouette scores in the original embedding space per label column.

    Embeddings are deduplicated to one vector per patient (mean-pooled across
    all slides for that patient) before scoring. This prevents same-patient
    near-duplicate slides from artificially inflating the silhouette score.

    A silhouette score near 1.0 means the embeddings cleanly separate that
    clinical label; near 0 means no separation; negative means worse than random.

    Args:
        embeddings: Full slide embedding dict.
        metadata_df: DataFrame with slide_name and label columns.
        label_columns: Metadata fields to evaluate (e.g. ['primary_tumor_site', 'hpv_association_p16']).
        sample_n: If set, subsample to this many *patients* for speed. Defaults
            to None (use all patients). With ~700 patients cosine silhouette
            completes in a few seconds.
        random_state: Seed for subsampling reproducibility.

    Returns:
        DataFrame with columns [label, silhouette_score, n_slides, n_classes,
        min_class_size] sorted by silhouette_score descending.
    """
    from sklearn.metrics import silhouette_score as _silhouette

    # Deduplicate to one slide per patient (mean-pooled) to avoid inflating
    # silhouette scores with near-identical same-patient slides.
    patient_embeddings = aggregate_embeddings_by_patient(embeddings)
    meta = metadata_df.set_index("slide_name")

    # Build a patient -> representative slide_name map for metadata lookup.
    # Use the first slide encountered for each patient.
    patient_to_slide: Dict[str, str] = {}
    for slide_name in embeddings:
        pid = _patient_id_from_slide(slide_name) or "__unknown__"
        if pid not in patient_to_slide:
            patient_to_slide[pid] = slide_name

    names = list(patient_embeddings.keys())  # patient IDs
    matrix = np.stack([patient_embeddings[pid] for pid in names])

    rng = np.random.default_rng(random_state)
    if sample_n is not None and len(names) > sample_n:
        idx = rng.choice(len(names), size=sample_n, replace=False)
        names = [names[i] for i in idx]
        matrix = matrix[idx]

    rows = []
    for col in label_columns:
        labels = [
            str(meta.loc[patient_to_slide[pid], col])
            if pid in patient_to_slide and patient_to_slide[pid] in meta.index
            else "unknown"
            for pid in names
        ]
        label_series = pd.Series(labels)
        counts = label_series.value_counts()
        # silhouette requires >= 2 classes each with >= 2 members
        valid_classes = counts[counts >= 2].index.tolist()
        mask = label_series.isin(valid_classes).values
        if mask.sum() < 2 or len(valid_classes) < 2:
            rows.append({
                "label": col, "silhouette_score": float("nan"),
                "n_slides": int(mask.sum()), "n_classes": len(valid_classes),
                "min_class_size": int(counts.min()),
            })
            continue
        score = _silhouette(matrix[mask], label_series[mask].values, metric="cosine")
        rows.append({
            "label": col,
            "silhouette_score": round(float(score), 4),
            "n_slides": int(mask.sum()),
            "n_classes": len(valid_classes),
            "min_class_size": int(counts[valid_classes].min()),
        })

    return pd.DataFrame(rows).sort_values("silhouette_score", ascending=False)


def cosine_similarity_null_distribution(
    embeddings: Dict[str, np.ndarray],
    sample_n: int = 300,
    random_state: int = 42,
) -> Dict[str, float]:
    """Estimate the null distribution of pairwise cosine similarities.

    Samples a random subset of slides and computes all pairwise cosine
    similarities (excluding self-pairs) to characterise the baseline.
    Use this to contextualise KNN similarity scores: a neighbor at 0.80
    is only meaningful if the population mean is substantially lower.

    Args:
        embeddings: Full slide embedding dict.
        sample_n: Number of slides to sample (full pairwise is O(n²)).
        random_state: Seed for reproducibility.

    Returns:
        Dict with keys: mean, std, p5, p25, median, p75, p95, n_pairs.
    """
    names = list(embeddings.keys())
    rng = np.random.default_rng(random_state)
    n = min(sample_n, len(names))
    idx = rng.choice(len(names), size=n, replace=False)
    sample_names = [names[i] for i in idx]
    matrix = np.stack([embeddings[n] for n in sample_names])

    sim_matrix = cosine_similarity(matrix)
    # Upper triangle only, excluding diagonal
    upper = sim_matrix[np.triu_indices(n, k=1)]

    return {
        "mean":   round(float(np.mean(upper)), 4),
        "std":    round(float(np.std(upper)), 4),
        "p5":     round(float(np.percentile(upper, 5)), 4),
        "p25":    round(float(np.percentile(upper, 25)), 4),
        "median": round(float(np.median(upper)), 4),
        "p75":    round(float(np.percentile(upper, 75)), 4),
        "p95":    round(float(np.percentile(upper, 95)), 4),
        "n_pairs": len(upper),
    }


def format_knn_results(
    knn_results: List[Tuple[str, float]],
    metadata_df: pd.DataFrame,
    null_dist: Optional[Dict[str, float]] = None,
    display_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Format KNN results as a readable match table with metadata columns.

    Args:
        knn_results: Output of find_k_nearest — list of (slide_name, similarity).
        metadata_df: DataFrame with slide_name and metadata columns.
        null_dist: Optional output of cosine_similarity_null_distribution. When
            provided, adds a 'vs_baseline' column showing how far each score is
            above the population mean.
        display_columns: Metadata columns to include. Defaults to a standard
            clinical set if None.

    Returns:
        DataFrame with rank, patient_id, slide_name, similarity, vs_baseline
        (if null_dist provided), and the requested metadata columns.
    """
    if display_columns is None:
        display_columns = [
            "primary_tumor_site",
            "histologic_type",
            "hpv_association_p16",
            "pT_stage",
            "smoking_status",
            "sex",
            "age_at_initial_diagnosis",
        ]

    meta = metadata_df.set_index("slide_name")
    rows = []
    for rank, (slide_name, sim) in enumerate(knn_results, start=1):
        row: dict = {"rank": rank, "slide_name": slide_name, "similarity": round(sim, 4)}
        row["patient_id"] = _patient_id_from_slide(slide_name) or ""
        if null_dist:
            row["vs_baseline"] = round(sim - null_dist["mean"], 4)
        for col in display_columns:
            row[col] = meta.loc[slide_name, col] if slide_name in meta.index and col in meta.columns else ""
        rows.append(row)

    col_order = ["rank", "patient_id", "similarity"]
    if null_dist:
        col_order.append("vs_baseline")
    col_order += ["slide_name"] + [c for c in display_columns if c not in col_order]
    return pd.DataFrame(rows)[col_order]


def _get_patient_slides(patient_id: str, embeddings: Dict[str, np.ndarray]) -> List[str]:
    """Return all slide names in embeddings that belong to patient_id."""
    pattern = f"patient{patient_id.zfill(3)}"
    return [name for name in embeddings if pattern in name]


def composite_patient_similarity(
    patient_a: str,
    patient_b: str,
    embeddings: Dict[str, np.ndarray],
    composition_profiles: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    metadata_fields: Optional[List[str]] = None,
    slide_aggregation: str = "max",
) -> Dict[str, float]:
    """Weighted combination of slide-level, composition, and metadata similarity.

    Slide-level similarity is computed directly from the per-slide ABMIL-aggregated
    embeddings stored in the CSV. When a patient has multiple slides, the cross-patient
    slide similarity matrix is computed and aggregated according to ``slide_aggregation``:

    - ``"max"``: maximum pairwise cosine similarity across all (slide_a, slide_b) pairs.
      Captures the closest morphological match between any two slides, which is the most
      clinically relevant signal for trial matching.
    - ``"mean"``: mean of all pairwise cosine similarities across the (n_a × n_b) cross-patient
      slide matrix. Note: this is NOT generally equivalent to cosine(mean(slides_a), mean(slides_b))
      unless the embeddings are L2-normalised — the two are only equal when all vectors are unit-norm.

    The composition profile similarity (beta term) already operates at the patient level
    (mean-pooled tissue proportions) and is unaffected by this parameter.

    Args:
        patient_a: Numeric patient ID string (e.g. '727').
        patient_b: Numeric patient ID string to compare against.
        embeddings: Full slide embedding dict (per-slide ABMIL-aggregated vectors).
        composition_profiles: Patient-level composition profiles from
            umap_retrieval.composition.build_patient_profiles.
        metadata_df: DataFrame with slide_name and metadata columns.
        alpha: Weight for slide-level cosine similarity.
        beta: Weight for composition profile similarity.
        gamma: Weight for metadata field match fraction.
        metadata_fields: Clinical fields to compare for metadata match.
            Defaults to primary_tumor_site, histologic_type,
            hpv_association_p16, pT_stage, pN_stage.
        slide_aggregation: How to aggregate cross-patient slide similarities.
            ``"max"`` (default) or ``"mean"``.

    Returns:
        Dict with keys: composite, slide_sim, comp_sim, meta_sim,
        slide_aggregation, and the input alpha/beta/gamma weights.

    Raises:
        ValueError: If either patient has no slides in embeddings, no
            composition profile, or slide_aggregation is not recognised.
    """
    if metadata_fields is None:
        metadata_fields = [
            "primary_tumor_site", "histologic_type",
            "hpv_association_p16", "pT_stage", "pN_stage",
        ]

    if slide_aggregation not in ("max", "mean"):
        raise ValueError(f"slide_aggregation must be 'max' or 'mean', got '{slide_aggregation}'")

    # 1. Slide-level cosine similarity — computed over the per-slide ABMIL embeddings
    #    directly, without a second mean-pool across slides.
    slides_a = _get_patient_slides(patient_a, embeddings)
    slides_b = _get_patient_slides(patient_b, embeddings)
    if not slides_a:
        raise ValueError(f"No slides found for patient {patient_a}")
    if not slides_b:
        raise ValueError(f"No slides found for patient {patient_b}")

    matrix_a = np.stack([embeddings[s] for s in slides_a])  # (n_a, D)
    matrix_b = np.stack([embeddings[s] for s in slides_b])  # (n_b, D)
    # Full cross-patient pairwise similarity matrix: shape (n_a, n_b)
    cross_sim = cosine_similarity(matrix_a, matrix_b)

    if slide_aggregation == "max":
        slide_sim = float(cross_sim.max())
    else:  # "mean" — mean of all (n_a × n_b) pairwise similarities; not equal to cosine(mean_a, mean_b) on unnormalised vectors
        slide_sim = float(cross_sim.mean())

    # 2. Composition profile similarity
    if patient_a not in composition_profiles:
        raise ValueError(f"No composition profile for patient {patient_a}")
    if patient_b not in composition_profiles:
        raise ValueError(f"No composition profile for patient {patient_b}")

    comp_sim = float(cosine_similarity(
        composition_profiles[patient_a].reshape(1, -1),
        composition_profiles[patient_b].reshape(1, -1),
    )[0, 0])

    # 3. Metadata field match fraction (ignores unknown values)
    pattern_a = f"patient{patient_a.zfill(3)}"
    pattern_b = f"patient{patient_b.zfill(3)}"
    rows_a = metadata_df[metadata_df["slide_name"].str.contains(pattern_a, na=False)]
    rows_b = metadata_df[metadata_df["slide_name"].str.contains(pattern_b, na=False)]

    meta_sim = 0.0
    if not rows_a.empty and not rows_b.empty:
        meta_a = rows_a.iloc[0]
        meta_b = rows_b.iloc[0]
        valid_fields = [
            f for f in metadata_fields
            if f in meta_a.index and f in meta_b.index
            and str(meta_a[f]) not in ("unknown", "nan", "")
            and str(meta_b[f]) not in ("unknown", "nan", "")
        ]
        if valid_fields:
            matches = sum(str(meta_a[f]) == str(meta_b[f]) for f in valid_fields)
            meta_sim = matches / len(valid_fields)

    composite = alpha * slide_sim + beta * comp_sim + gamma * meta_sim

    return {
        "composite": round(composite, 4),
        "slide_sim": round(slide_sim, 4),
        "comp_sim": round(comp_sim, 4),
        "meta_sim": round(meta_sim, 4),
        "slide_aggregation": slide_aggregation,
        "alpha": alpha, "beta": beta, "gamma": gamma,
    }


def rank_patients_by_composite(
    query_patient: str,
    candidate_patients: List[str],
    embeddings: Dict[str, np.ndarray],
    composition_profiles: Dict[str, np.ndarray],
    metadata_df: pd.DataFrame,
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
    metadata_fields: Optional[List[str]] = None,
    display_columns: Optional[List[str]] = None,
    slide_aggregation: str = "max",
) -> pd.DataFrame:
    """Rank candidate patients by composite similarity to a query patient.

    Args:
        query_patient: Numeric patient ID string.
        candidate_patients: List of patient IDs to rank.
        embeddings: Full slide embedding dict (per-slide ABMIL-aggregated vectors).
        composition_profiles: Patient-level composition profiles.
        metadata_df: DataFrame with slide_name and metadata columns.
        alpha: Slide-level similarity weight.
        beta: Composition profile similarity weight.
        gamma: Metadata match weight.
        metadata_fields: Fields to use for metadata match score.
        display_columns: Metadata columns to append to the output table.
            Defaults to primary_tumor_site, histologic_type,
            hpv_association_p16, pT_stage, smoking_status.
        slide_aggregation: How to aggregate cross-patient slide similarities.
            ``"max"`` (default) captures the closest morphological match between
            any two slides. ``"mean"`` averages all pairwise similarities.
            See composite_patient_similarity for details.

    Returns:
        DataFrame sorted by composite score descending, with columns
        [rank, patient_id, composite, slide_sim, comp_sim, meta_sim,
        ...display_columns]. A header row for the query patient is
        prepended with rank=0 for easy comparison.
    """
    if display_columns is None:
        display_columns = [
            "primary_tumor_site",
            "histologic_type",
            "hpv_association_p16",
            "pT_stage",
            "pN_stage",       # included in meta_sim score — shown for transparency
            "smoking_status",
        ]

    # Verify query patient has slides in embeddings
    query_slides = _get_patient_slides(query_patient, embeddings)
    if not query_slides:
        raise ValueError(
            f"Query patient '{query_patient}' has a composition profile but no slides "
            f"in the embeddings dict. This patient's slides may have been dropped during "
            f"zero-embedding filtering. Choose a different PATIENT_ID."
        )
    meta = metadata_df.set_index("slide_name")
    pid_meta: Dict[str, Dict] = {}
    for slide_name in metadata_df["slide_name"]:
        pid = _patient_id_from_slide(slide_name)
        if pid and pid not in pid_meta:
            pid_meta[pid] = {
                col: str(meta.loc[slide_name, col])
                if col in meta.columns else ""
                for col in display_columns
            }

    rows = []
    first_error: Optional[str] = None
    for pid in candidate_patients:
        if pid == query_patient:
            continue
        try:
            scores = composite_patient_similarity(
                query_patient, pid, embeddings, composition_profiles,
                metadata_df, alpha, beta, gamma, metadata_fields,
                slide_aggregation=slide_aggregation,
            )
            scores["patient_id"] = pid
            scores.update(pid_meta.get(pid, {col: "" for col in display_columns}))
            rows.append(scores)
        except ValueError as e:
            if first_error is None:
                first_error = str(e)
            continue

    if not rows:
        raise ValueError(
            f"No candidates could be scored for query patient '{query_patient}'. "
            f"First error: {first_error}. "
            f"Check that PATIENT_ID matches a key in patient_profiles "
            f"(sample keys: {list(composition_profiles.keys())[:5]})"
        )

    df = pd.DataFrame(rows).sort_values("composite", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    col_order = ["rank", "patient_id", "composite", "slide_sim", "comp_sim", "meta_sim"] + display_columns
    df = df[col_order]

    # Prepend query patient as rank-0 reference row
    query_row = {"rank": 0, "patient_id": f"{query_patient} (query)",
                 "composite": 1.0, "slide_sim": 1.0, "comp_sim": 1.0, "meta_sim": 1.0}
    query_row.update(pid_meta.get(query_patient, {col: "" for col in display_columns}))
    df = pd.concat([pd.DataFrame([query_row])[col_order], df], ignore_index=True)

    return df
