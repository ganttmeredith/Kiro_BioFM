"""Data loading, validation, and metadata filtering."""

import os
import re
import subprocess
import warnings
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd

from umap_retrieval.config import (
    FILTERABLE_COLUMNS,
    METADATA_COLUMNS,
    OPTIONAL_METADATA_COLUMNS,
    REQUIRED_METADATA_COLUMNS,
)


def load_and_validate_csv(csv_path: str) -> pd.DataFrame:
    """Load slide metadata CSV and validate structural integrity.

    Validates:
    - slide_name is the first column
    - All 10 filterable and 5 non-filterable metadata columns are present
    - All slide_name values are unique
    - No slide_name value is empty or null

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Validated pandas DataFrame.

    Raises:
        ValueError: With a message identifying the specific violation.
    """
    df = pd.read_csv(csv_path)

    if df.columns[0] != "slide_name":
        raise ValueError(
            f"Expected 'slide_name' as first column, got '{df.columns[0]}'"
        )

    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")

    duplicates = df["slide_name"][df["slide_name"].duplicated()].tolist()
    if duplicates:
        raise ValueError(f"Duplicate slide_name values: {duplicates}")

    null_mask = df["slide_name"].isna() | (
        df["slide_name"].astype(str).str.strip() == ""
    )
    if null_mask.any():
        bad_rows = df.index[null_mask].tolist()
        raise ValueError(f"Empty or null slide_name at row(s): {bad_rows}")

    return df


def extract_metadata_and_embeddings(
    csv_df: pd.DataFrame,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Split a validated CSV DataFrame into metadata and embeddings.

    Args:
        csv_df: DataFrame from :func:`load_and_validate_csv`.

    Returns:
        Tuple of (metadata_df, embeddings_dict) where embeddings_dict maps
        slide_name to a 1536-d numpy array.
    """
    metadata_df = csv_df[
        METADATA_COLUMNS + [c for c in OPTIONAL_METADATA_COLUMNS if c in csv_df.columns]
    ].copy()
    embedding_cols = csv_df.columns[-1536:]
    embeddings: Dict[str, np.ndarray] = {
        row["slide_name"]: row[embedding_cols].values.astype(np.float64)
        for _, row in csv_df.iterrows()
    }
    return metadata_df, embeddings


def apply_metadata_filters(
    metadata_df: pd.DataFrame,
    metadata_filters: Optional[Dict[str, List[str]]],
    filterable_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter the metadata DataFrame by filterable metadata criteria.

    Applies AND logic across all filter criteria: each field must match one
    of the accepted values for a row to be included.

    Args:
        metadata_df: DataFrame containing slide metadata.
        metadata_filters: Dictionary mapping filterable field names to lists
            of accepted values. If None or empty, returns the full DataFrame.
        filterable_columns: Allowed filter keys. Defaults to FILTERABLE_COLUMNS.

    Returns:
        Filtered DataFrame subset.

    Raises:
        KeyError: If a filter key is not in filterable_columns or not in the
            DataFrame columns.
    """
    if not metadata_filters:
        return metadata_df

    if filterable_columns is None:
        filterable_columns = FILTERABLE_COLUMNS

    for key in metadata_filters:
        if key not in metadata_df.columns:
            raise KeyError(f"Filter field '{key}' does not exist in metadata columns")
        if key not in filterable_columns:
            raise KeyError(
                f"Field '{key}' is not a filterable column. "
                f"Filterable columns: {filterable_columns}"
            )

    mask = pd.Series(True, index=metadata_df.index)
    for field, accepted_values in metadata_filters.items():
        mask = mask & metadata_df[field].isin(accepted_values)

    filtered = metadata_df[mask].copy()

    if len(filtered) < 2:
        warnings.warn(
            f"Only {len(filtered)} slide(s) match the filter criteria — "
            f"filtering may be too restrictive. Filters: {metadata_filters}"
        )

    return filtered


def fetch_patch_h5_files(
    metadata_df: pd.DataFrame,
    anchor_patient_id: str,
    n_patients: int = 3,
    local_dir: str = "../data/h5_cache",
) -> List[Tuple[str, str]]:
    """Download H5 patch files for N patients starting from an anchor patient.

    Searches metadata_df for slides matching anchor_patient_id, then expands
    to n_patients consecutive patients (by order of appearance in metadata_df).
    All slides for each selected patient are downloaded via `aws s3 cp`.

    Args:
        metadata_df: DataFrame with at least 'slide_name' and 'h5file' columns.
        anchor_patient_id: Patient ID substring to anchor the search (e.g. "727").
        n_patients: Number of distinct patients to include starting from anchor.
        local_dir: Local directory to cache downloaded H5 files.

    Returns:
        List of (slide_name, local_path) tuples for all successfully fetched files.

    Raises:
        ValueError: If no slide matches anchor_patient_id.
    """
    os.makedirs(local_dir, exist_ok=True)

    anchor_rows = metadata_df[metadata_df["slide_name"].str.contains(anchor_patient_id, na=False)]
    if anchor_rows.empty:
        raise ValueError(f"No slide found matching '{anchor_patient_id}' in metadata_df")

    # Extract ordered unique patient IDs from all slide names
    all_pids = list(dict.fromkeys(
        m.group(1)
        for name in metadata_df["slide_name"]
        if (m := re.search(r'patient(\d+)', str(name)))
    ))

    anchor_match = re.search(r'patient(\d+)', anchor_rows.iloc[0]["slide_name"])
    anchor_pid = anchor_match.group(1) if anchor_match else None
    start = all_pids.index(anchor_pid) if anchor_pid in all_pids else 0
    selected_pids = all_pids[start:start + n_patients]

    patch_h5_files: List[Tuple[str, str]] = []
    for pid in selected_pids:
        pid_rows = metadata_df[metadata_df["slide_name"].str.contains(f"patient{pid}", na=False)]
        for _, row in pid_rows.iterrows():
            s3_path = row.get("h5file", "") if "h5file" in row.index else ""
            if not s3_path or (isinstance(s3_path, float) and pd.isna(s3_path)):
                continue
            local_path = os.path.join(local_dir, os.path.basename(s3_path))
            if os.path.exists(local_path):
                print(f"  cached : {os.path.basename(local_path)}")
            else:
                print(f"  downloading: {s3_path}")
                result = subprocess.run(
                    ["aws", "s3", "cp", s3_path, local_path],
                    capture_output=True, text=True,
                )
                if result.returncode != 0:
                    print(f"  FAILED: {result.stderr.strip()}")
                    continue
            patch_h5_files.append((row["slide_name"], local_path))

    print(f"\nReady: {len(patch_h5_files)} H5 files across {n_patients} patients")
    for slide_name, local_path in patch_h5_files:
        print(f"  {slide_name}")

    return patch_h5_files


def load_patch_features(
    patch_h5_files: List[Tuple[str, str]],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and stack patch coordinates and features from a list of H5 files.

    Args:
        patch_h5_files: List of (slide_name, local_path) as returned by
            :func:`fetch_patch_h5_files`.

    Returns:
        Tuple of (coords, features, sources) where:
        - coords: (N, 2) array of patch coordinates
        - features: (N, D) float32 array of patch embeddings
        - sources: list of slide_name repeated per patch (length N)
    """
    coords_list, feats_list, sources = [], [], []
    for slide_name, path in patch_h5_files:
        with h5py.File(path, "r") as f:
            c = f["coords"][:]
            feat_key = "features" if "features" in f else "embeddings"
            ft = f[feat_key][:].astype(np.float32)
        coords_list.append(c)
        feats_list.append(ft)
        sources.extend([slide_name] * len(ft))
        print(f"  {slide_name}: {len(ft)} patches, dim={ft.shape[1]}")

    coords = np.vstack(coords_list)
    features = np.vstack(feats_list)
    print(f"\nTotal patches : {features.shape[0]}, embedding dim: {features.shape[1]}")
    return coords, features, sources


def fetch_all_h5_files(
    metadata_df: pd.DataFrame,
    local_dir: str = "../data/h5_cache",
    max_workers: int = 8,
    patient_ids: Optional[List[str]] = None,
) -> List[Tuple[str, str]]:
    """Download H5 patch files for all (or a specified subset of) patients.

    Uses a thread pool for parallel downloads. Already-cached files are
    skipped. Progress is printed as downloads complete.

    Args:
        metadata_df: DataFrame with 'slide_name' and 'h5file' columns.
        local_dir: Local directory to cache downloaded H5 files.
        max_workers: Number of parallel download threads.
        patient_ids: Optional list of numeric patient ID strings to restrict
            the download (e.g. ['727', '261']). If None, downloads all patients.

    Returns:
        List of (slide_name, local_path) tuples for all successfully fetched files.
    """
    import concurrent.futures

    os.makedirs(local_dir, exist_ok=True)

    # Build list of (slide_name, s3_path) to fetch
    rows_to_fetch = []
    for _, row in metadata_df.iterrows():
        s3_path = row.get("h5file", "")
        if not s3_path or (isinstance(s3_path, float) and pd.isna(s3_path)):
            continue
        if patient_ids is not None:
            pid_match = re.search(r'patient(\d+)', str(row["slide_name"]))
            if not pid_match or pid_match.group(1) not in patient_ids:
                continue
        rows_to_fetch.append((row["slide_name"], str(s3_path)))

    total = len(rows_to_fetch)
    print(f"Fetching {total} H5 files to {local_dir} ({max_workers} workers)...")

    results: List[Tuple[str, str]] = []
    failed: List[str] = []

    def _download(slide_name: str, s3_path: str) -> Optional[Tuple[str, str]]:
        local_path = os.path.join(local_dir, os.path.basename(s3_path))
        if os.path.exists(local_path):
            return (slide_name, local_path)
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, local_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        return (slide_name, local_path)

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download, slide_name, s3_path): slide_name
            for slide_name, s3_path in rows_to_fetch
        }
        for future in concurrent.futures.as_completed(futures):
            done += 1
            slide_name = futures[future]
            result = future.result()
            if result:
                results.append(result)
            else:
                failed.append(slide_name)
            if done % 50 == 0 or done == total:
                print(f"  {done}/{total} — {len(results)} ok, {len(failed)} failed")

    if failed:
        print(f"\nFailed ({len(failed)}): {failed[:10]}{'...' if len(failed) > 10 else ''}")
    print(f"\nReady: {len(results)} H5 files")
    return results
