"""Patch-level composition profiles from H5 files.

Builds a fixed-length tissue composition vector per slide by clustering
patch embeddings into tissue types and recording the proportions. This
captures morphological heterogeneity that slide-level mean-pooling destroys.

Supports both local file paths and S3 URIs (s3://bucket/key). S3 access
uses s3fs with the ambient IAM role — no credentials or downloads needed
when running on SageMaker or any IAM-authenticated environment.
"""

import contextlib
import os
import re
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize


@contextlib.contextmanager
def _open_h5(path: str):
    """Open an H5 file from a local path or S3 URI transparently.

    Args:
        path: Local filesystem path or s3://bucket/key URI.

    Yields:
        Open h5py.File object in read mode.
    """
    if path.startswith("s3://"):
        try:
            import s3fs
        except ImportError:
            raise ImportError(
                "s3fs is required for S3 URI support. "
                "Install it with: pip install s3fs"
            )
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(path, "rb") as f:
            with h5py.File(f, "r") as hf:
                yield hf
    else:
        with h5py.File(path, "r") as hf:
            yield hf


def _load_features_parallel(
    paths: List[str],
    max_workers: int = 8,
) -> List[Tuple[str, np.ndarray]]:
    """Load and L2-normalise patch features from H5 files in parallel.

    Uses a thread pool — optimal for I/O-bound S3 reads. Each thread
    opens its own s3fs connection so there are no shared-state issues.

    Args:
        paths: List of local paths or S3 URIs.
        max_workers: Number of concurrent reader threads.

    Returns:
        List of (slide_name, features_array) in the same order as paths.
    """
    import concurrent.futures
    try:
        from tqdm.auto import tqdm
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    def _read_one(path: str) -> Tuple[str, np.ndarray]:
        with _open_h5(path) as f:
            feat_key = "features" if "features" in f else "embeddings"
            feats = normalize(f[feat_key][:].astype(np.float32))
        slide_name = os.path.basename(path.rstrip("/")).replace(".h5", "")
        return slide_name, feats

    results: List[Optional[Tuple[str, np.ndarray]]] = [None] * len(paths)
    failed: List[str] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(_read_one, p): i for i, p in enumerate(paths)}
        iterator = concurrent.futures.as_completed(future_to_idx)
        if _tqdm is not None:
            iterator = _tqdm(iterator, total=len(paths), desc="Loading H5 files", unit="file")
        for future in iterator:
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                failed.append(paths[idx])
                if _tqdm is None:
                    print(f"  Warning: failed to read {paths[idx]}: {e}")

    if failed:
        print(f"  Warning: {len(failed)} file(s) failed to load")

    return [r for r in results if r is not None]


def build_composition_profiles(
    h5_paths: List[str],
    n_tissue_types: int = 10,
    seed: int = 42,
    max_workers: int = 8,
) -> Tuple[Dict[str, np.ndarray], MiniBatchKMeans]:
    """Fit a shared patch codebook and compute per-slide composition profiles.

    Step 1: Collect all patches across all slides and fit a global KMeans
    codebook (n_tissue_types clusters). Each cluster represents a recurring
    morphological pattern (e.g. tumor epithelium, stroma, necrosis).

    Step 2: For each slide, assign patches to clusters and record the
    proportion vector — a (n_tissue_types,) array summing to 1.

    Args:
        h5_paths: List of local H5 file paths or S3 URIs.
        n_tissue_types: Number of tissue type clusters (codebook size).
        seed: Random seed for reproducibility.
        max_workers: Number of parallel reader threads for H5/S3 I/O.

    Returns:
        Tuple of:
        - profiles: Dict mapping slide_name -> (n_tissue_types,) proportion array.
        - codebook: Fitted MiniBatchKMeans model (use to profile new slides).
    """
    loaded = _load_features_parallel(h5_paths, max_workers=max_workers)

    all_patches: List[np.ndarray] = []
    slide_boundaries: List[Tuple[int, int, str]] = []
    cursor = 0
    for slide_name, feats in loaded:
        all_patches.append(feats)
        slide_boundaries.append((cursor, cursor + len(feats), slide_name))
        cursor += len(feats)

    if not all_patches:
        raise RuntimeError(
            f"No H5 files could be loaded from {len(h5_paths)} path(s). "
            "Check that paths are valid, AWS credentials are configured, "
            "and the files exist."
        )
    stacked = np.vstack(all_patches)
    print(f"Fitting codebook on {len(stacked)} patches from {len(h5_paths)} slides "
          f"(n_tissue_types={n_tissue_types})")
    codebook = MiniBatchKMeans(
        n_clusters=n_tissue_types,
        random_state=seed,
        batch_size=1024,
        n_init="auto",
    )
    all_labels = codebook.fit_predict(stacked)

    profiles: Dict[str, np.ndarray] = {}
    for start, end, name in slide_boundaries:
        slide_labels = all_labels[start:end]
        proportions = np.bincount(slide_labels, minlength=n_tissue_types) / len(slide_labels)
        profiles[name] = proportions.astype(np.float32)

    return profiles, codebook


def profile_slide(
    h5_path: str,
    codebook: MiniBatchKMeans,
) -> np.ndarray:
    """Compute a composition profile for a single slide using an existing codebook.

    Args:
        h5_path: Local path to the H5 file.
        codebook: Fitted MiniBatchKMeans from build_composition_profiles.

    Returns:
        (n_tissue_types,) proportion array.
    """
    with _open_h5(h5_path) as f:
        feat_key = "features" if "features" in f else "embeddings"
        feats = normalize(f[feat_key][:].astype(np.float32))
    labels = codebook.predict(feats)
    return (np.bincount(labels, minlength=codebook.n_clusters) / len(labels)).astype(np.float32)


def patient_composition_profile(
    slide_profiles: Dict[str, np.ndarray],
    patient_id: str,
) -> Optional[np.ndarray]:
    """Mean-pool composition profiles across all slides for one patient.

    Args:
        slide_profiles: Dict from build_composition_profiles.
        patient_id: Numeric patient ID string (e.g. '727').

    Returns:
        Mean proportion vector, or None if no slides found.
    """
    pattern = f"patient{patient_id.zfill(3)}"
    matching = {k: v for k, v in slide_profiles.items() if pattern in k}
    if not matching:
        return None
    return np.mean(np.stack(list(matching.values())), axis=0).astype(np.float32)


def build_patient_profiles(
    slide_profiles: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build patient-level composition profiles from slide-level profiles.

    Extracts patient IDs from slide names and mean-pools all slides per patient.

    Args:
        slide_profiles: Dict mapping slide_name -> proportion array.

    Returns:
        Dict mapping patient_id -> mean proportion array.
    """
    buckets: Dict[str, List[np.ndarray]] = {}
    for slide_name, profile in slide_profiles.items():
        m = re.search(r'patient(\d+)', slide_name)
        pid = m.group(1) if m else "__unknown__"
        buckets.setdefault(pid, []).append(profile)
    return {
        pid: np.mean(np.stack(vecs), axis=0).astype(np.float32)
        for pid, vecs in buckets.items()
    }


def composition_silhouette(
    patient_profiles: Dict[str, np.ndarray],
    metadata_df,
    label_column: str,
) -> float:
    """Compute silhouette score for composition profiles against a clinical label.

    Use this to validate whether composition profiles separate clinical labels
    better than raw slide-level embeddings (baseline from embedding_silhouette_scores).

    Args:
        patient_profiles: Dict from build_patient_profiles.
        metadata_df: DataFrame with slide_name and metadata columns.
        label_column: Metadata field to evaluate.

    Returns:
        Silhouette score (cosine metric).
    """
    from sklearn.metrics import silhouette_score

    meta = metadata_df.set_index("slide_name")

    # Map patient_id -> label via first matching slide
    pid_to_label: Dict[str, str] = {}
    for slide_name in metadata_df["slide_name"]:
        m = re.search(r'patient(\d+)', str(slide_name))
        if not m:
            continue
        pid = m.group(1)
        if pid not in pid_to_label and slide_name in meta.index:
            pid_to_label[pid] = str(meta.loc[slide_name, label_column])

    pids = [p for p in patient_profiles if p in pid_to_label]
    labels = [pid_to_label[p] for p in pids]
    matrix = np.stack([patient_profiles[p] for p in pids])

    label_series = __import__('pandas').Series(labels)
    counts = label_series.value_counts()
    valid = counts[counts >= 2].index.tolist()
    mask = label_series.isin(valid).values

    if mask.sum() < 2 or len(valid) < 2:
        return float("nan")

    return float(silhouette_score(matrix[mask], label_series[mask].values, metric="cosine"))


def build_composition_profiles_from_metadata(
    metadata_df,
    n_tissue_types: int = 10,
    seed: int = 42,
    local_dir: Optional[str] = None,
    max_workers: int = 8,
) -> Tuple[Dict[str, np.ndarray], MiniBatchKMeans]:
    """Build composition profiles from local H5 files referenced in metadata_df.

    Reads the 'h5file' column for filenames and looks them up in local_dir
    by basename. Files not present locally are skipped — no S3 downloads
    are attempted.

    Args:
        metadata_df: DataFrame with 'slide_name' and 'h5file' columns.
        n_tissue_types: Codebook size (number of tissue type clusters).
        seed: Random seed.
        local_dir: Local cache directory containing pre-downloaded H5 files.
            Defaults to '../data/h5_cache'.
        max_workers: Number of parallel reader threads.

    Returns:
        Tuple of (profiles dict, fitted codebook).
    """
    if local_dir is None:
        local_dir = "../data/h5_cache"

    paths: List[str] = []
    skipped = 0
    for _, row in metadata_df.iterrows():
        h5file = row.get("h5file", "")
        if not h5file or (isinstance(h5file, float) and __import__('math').isnan(h5file)):
            continue
        local_path = os.path.join(local_dir, os.path.basename(str(h5file)))
        if os.path.exists(local_path):
            paths.append(local_path)
        else:
            skipped += 1

    if skipped:
        print(f"Skipped {skipped} slides not found in local cache '{local_dir}'")
    print(f"Building profiles for {len(paths)} slides from local cache ({max_workers} workers)")
    return build_composition_profiles(paths, n_tissue_types=n_tissue_types,
                                      seed=seed, max_workers=max_workers)


def sample_patch_features(
    metadata_df,
    n_patients: int = 5,
    local_dir: Optional[str] = None,
    max_workers: int = 8,
) -> np.ndarray:
    """Load patch features for a sample of patients from local H5 files.

    Uses the 'h5file' column in metadata_df to derive filenames, then looks
    them up in local_dir by basename. Files not present locally are skipped —
    no S3 downloads are attempted.

    Args:
        metadata_df: DataFrame with 'slide_name' and 'h5file' columns.
        n_patients: Number of distinct patients to sample.
        local_dir: Local cache directory containing pre-downloaded H5 files.
            Defaults to '../data/h5_cache'.
        max_workers: Parallel reader threads.

    Returns:
        (N, D) float32 array of L2-normalised patch features.
    """
    if local_dir is None:
        local_dir = "../data/h5_cache"

    seen_pids: list = []
    paths: List[str] = []
    for _, row in metadata_df.iterrows():
        h5file = row.get("h5file", "")
        if not h5file or (isinstance(h5file, float) and __import__('math').isnan(h5file)):
            continue
        m = re.search(r'patient(\d+)', str(row.get("slide_name", "")))
        pid = m.group(1) if m else None
        if pid and pid not in seen_pids:
            seen_pids.append(pid)
        if len(seen_pids) > n_patients:
            break
        local_path = os.path.join(local_dir, os.path.basename(str(h5file)))
        if os.path.exists(local_path):
            paths.append(local_path)
        else:
            print(f"  Skipping (not in local cache): {os.path.basename(str(h5file))}")

    print(f"Loading patch features from {len(paths)} local H5 files "
          f"({n_patients} patients requested) for K-sweep...")
    loaded = _load_features_parallel(paths, max_workers=max_workers)
    if not loaded:
        raise RuntimeError(
            f"No H5 files could be loaded from {len(paths)} path(s) in '{local_dir}'. "
            "Check that the files exist in the local cache directory."
        )
    return np.vstack([feats for _, feats in loaded])


def save_profiles(
    patient_profiles: Dict[str, np.ndarray],
    codebook: MiniBatchKMeans,
    output_dir: str = "./profiles",
) -> None:
    """Persist patient profiles and codebook to disk or S3.

    If output_dir is an S3 URI, attempts to write there first. If S3 write
    fails (e.g. no write permissions), falls back to a local directory with
    the same name as the S3 prefix basename.

    Args:
        patient_profiles: Dict from build_patient_profiles.
        codebook: Fitted MiniBatchKMeans from build_composition_profiles.
        output_dir: Local directory or S3 URI (s3://bucket/prefix).
    """
    import joblib
    import tempfile

    def _save_local(directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        np.savez(os.path.join(directory, "patient_profiles.npz"), **patient_profiles)
        joblib.dump(codebook, os.path.join(directory, "codebook.joblib"))
        print(f"Saved {len(patient_profiles)} patient profiles → {directory}/patient_profiles.npz")
        print(f"Saved codebook (K={codebook.n_clusters}) → {directory}/codebook.joblib")

    if output_dir.startswith("s3://"):
        try:
            import s3fs
            fs = s3fs.S3FileSystem(anon=False)
            profiles_path = output_dir.rstrip("/") + "/patient_profiles.npz"
            codebook_path = output_dir.rstrip("/") + "/codebook.joblib"
            with tempfile.TemporaryDirectory() as tmp:
                local_npz = os.path.join(tmp, "patient_profiles.npz")
                local_cb = os.path.join(tmp, "codebook.joblib")
                np.savez(local_npz, **patient_profiles)
                joblib.dump(codebook, local_cb)
                fs.put(local_npz, profiles_path)
                fs.put(local_cb, codebook_path)
            print(f"Saved {len(patient_profiles)} patient profiles → {profiles_path}")
            print(f"Saved codebook (K={codebook.n_clusters}) → {codebook_path}")
        except Exception as e:
            local_fallback = "./" + output_dir.rstrip("/").split("/")[-1]
            print(f"S3 write failed ({e}) — saving locally to {local_fallback}")
            _save_local(local_fallback)
    else:
        _save_local(output_dir)


def load_profiles(
    output_dir: str = "./profiles",
) -> Tuple[Dict[str, np.ndarray], MiniBatchKMeans]:
    """Load previously saved patient profiles and codebook from disk or S3.

    Args:
        output_dir: Local directory or S3 URI (s3://bucket/prefix) written
            by save_profiles.

    Returns:
        Tuple of (patient_profiles dict, fitted codebook).

    Raises:
        FileNotFoundError: If profiles or codebook files are missing.
    """
    import joblib
    import tempfile

    is_s3 = output_dir.startswith("s3://")
    profiles_path = (output_dir.rstrip("/") + "/patient_profiles.npz")
    codebook_path = (output_dir.rstrip("/") + "/codebook.joblib")

    if is_s3:
        try:
            import s3fs
        except ImportError:
            raise ImportError("s3fs is required for S3 input. pip install s3fs")
        fs = s3fs.S3FileSystem(anon=False)
        if not fs.exists(profiles_path):
            raise FileNotFoundError(
                f"Profiles not found: {profiles_path}\n"
                f"First-run: build profiles with build_composition_profiles_from_metadata, "
                f"then call save_profiles(patient_profiles, codebook, '{output_dir}')."
            )
        with tempfile.TemporaryDirectory() as tmp:
            local_npz = os.path.join(tmp, "patient_profiles.npz")
            local_cb = os.path.join(tmp, "codebook.joblib")
            fs.get(profiles_path, local_npz)
            fs.get(codebook_path, local_cb)
            data = np.load(local_npz)
            patient_profiles = {pid: data[pid] for pid in data.files}
            codebook = joblib.load(local_cb)
    else:
        if not os.path.exists(profiles_path):
            raise FileNotFoundError(
                f"Profiles not found: {profiles_path}\n"
                f"First-run: build profiles with build_composition_profiles_from_metadata, "
                f"then call save_profiles(patient_profiles, codebook, '{output_dir}')."
            )
        if not os.path.exists(codebook_path):
            raise FileNotFoundError(
                f"Codebook not found: {codebook_path}\n"
                f"Re-run build_composition_profiles_from_metadata and save_profiles."
            )
        data = np.load(profiles_path)
        patient_profiles = {pid: data[pid] for pid in data.files}
        codebook = joblib.load(codebook_path)

    print(f"Loaded {len(patient_profiles)} patient profiles from {profiles_path}")
    print(f"Loaded codebook (K={codebook.n_clusters}) from {codebook_path}")
    return patient_profiles, codebook


def update_profiles(
    patient_profiles: Dict[str, np.ndarray],
    codebook: MiniBatchKMeans,
    metadata_df,
    output_dir: str = "./profiles",
    local_dir: Optional[str] = None,
    max_workers: int = 8,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Profile new patients against an existing codebook and update saved profiles.

    Detects patients in metadata_df that are not yet in patient_profiles,
    streams their H5 files, assigns patches to the existing codebook clusters,
    and merges the new profiles in. The updated profiles are saved to disk.

    The codebook is NOT retrained — new patients are projected into the
    existing tissue-type space. This ensures consistency across runs.

    Args:
        patient_profiles: Existing profiles dict (from load_profiles).
        codebook: Fitted codebook (from load_profiles).
        metadata_df: DataFrame with 'slide_name' and 'h5file' columns,
            including both existing and new patients.
        output_dir: Directory to save updated profiles (same as load_profiles).
        local_dir: Optional local cache directory for H5 files.
        max_workers: Parallel reader threads.

    Returns:
        Tuple of:
        - updated patient_profiles dict (existing + new patients)
        - list of newly added patient IDs
    """
    # Find patient IDs in metadata not yet profiled
    existing_pids = set(patient_profiles.keys())
    new_paths: List[str] = []
    new_slide_names: List[str] = []

    for _, row in metadata_df.iterrows():
        s3_path = row.get("h5file", "")
        if not s3_path or (isinstance(s3_path, float) and __import__('math').isnan(s3_path)):
            continue
        s3_path = str(s3_path)
        slide_name = str(row.get("slide_name", ""))
        m = re.search(r'patient(\d+)', slide_name)
        if not m:
            continue
        pid = m.group(1)
        if pid in existing_pids:
            continue
        # Only add one path per new patient (we'll mean-pool slides after)
        if local_dir:
            local_path = os.path.join(local_dir, os.path.basename(s3_path))
            path = local_path if os.path.exists(local_path) else s3_path
        else:
            path = s3_path
        new_paths.append(path)
        new_slide_names.append(slide_name)

    if not new_paths:
        print("No new patients found — profiles are up to date.")
        return patient_profiles, []

    print(f"Found {len(new_paths)} new slides to profile...")
    loaded = _load_features_parallel(new_paths, max_workers=max_workers)

    # Build slide-level profiles using existing codebook
    new_slide_profiles: Dict[str, np.ndarray] = {}
    for slide_name, feats in loaded:
        labels = codebook.predict(feats)
        proportions = np.bincount(labels, minlength=codebook.n_clusters) / len(labels)
        new_slide_profiles[slide_name] = proportions.astype(np.float32)

    # Mean-pool to patient level
    new_patient_profiles = build_patient_profiles(new_slide_profiles)
    new_pids = list(new_patient_profiles.keys())

    # Merge and save
    updated = {**patient_profiles, **new_patient_profiles}
    save_profiles(updated, codebook, output_dir)
    print(f"Added {len(new_pids)} new patient(s): {new_pids}")

    return updated, new_pids
