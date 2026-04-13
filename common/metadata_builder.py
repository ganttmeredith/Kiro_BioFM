"""Helper functions for the Metadata Builder notebook.

Extracted here so they can be tested independently of the notebook.
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# Ordered metadata columns for the enriched embeddings CSV.
ENRICHED_METADATA_COLUMNS: list[str] = [
    "primary_tumor_site",
    "pT_stage",
    "pN_stage",
    "grading_hpv",
    "hpv_association_p16",
    "histologic_type",
    "survival_status",
    "recurrence",
    "smoking_status",
    "sex",
    "perineural_invasion_Pn",
    "lymphovascular_invasion",
    "perinodal_invasion",
    "year_of_initial_diagnosis",
    "age_at_initial_diagnosis",
    "primarily_metastasis",
    "h5file",
]

# Reverse map: output column name -> actual DataFrame column name.
# For columns not in FIELD_NAME_MAP the name is used as-is.
_ENRICHED_COL_MAP: dict[str, str] = {
    "grading_hpv": "grading",
    "lymphovascular_invasion": "lymphovascular_invasion_L",
}


def _to_str(value) -> str:
    """Convert a value to string, replacing null/NaN with ``"unknown"``."""
    if pd.isna(value):
        return "unknown"
    return str(value)


def _extract_patient_id_from_slide(slide_name: str) -> str | None:
    """Extract the zero-padded patient ID from a slide name.

    Expects a trailing ``_patient<NNN>`` segment (with optional file
    extension).  Returns the zero-padded ID string or ``None`` if the
    pattern is not found.
    """
    m = re.search(r"_patient(\d+)", slide_name)
    if m is None:
        return None
    return m.group(1).zfill(3)


def _build_h5file_column(
    slide_names: pd.Series,
    h5_s3_base: str = "",
    local_h5_dir: str = "",
) -> list[str]:
    """Build h5file paths for each slide.

    Priority:
    1. If ``local_h5_dir`` is set, look for ``<local_h5_dir>/<slide_name>.h5``
       and return the absolute local path when found, ``"not_found"`` otherwise.
       No S3 calls are made.
    2. If ``h5_s3_base`` is set, validate against S3 via a single
       ``list_objects_v2`` scan and return the full S3 URI or ``"not_found"``.
    3. If neither is set, return ``"unknown"`` for all rows.
    """
    if local_h5_dir:
        import os
        results: list[str] = []
        for slide_name in slide_names:
            local_path = os.path.join(local_h5_dir, f"{slide_name}.h5")
            results.append(local_path if os.path.exists(local_path) else "not_found")
        found = sum(1 for r in results if r != "not_found")
        logger.info("Local h5 lookup: %d found, %d not_found", found, len(results) - found)
        return results

    if not h5_s3_base:
        return ["unknown"] * len(slide_names)

    import boto3

    base = h5_s3_base.rstrip("/")
    m = re.match(r"s3://([^/]+)(?:/(.*))?", base)
    if not m:
        logger.warning("Cannot parse h5_s3_base as S3 URI: %s", base)
        return ["unknown"] * len(slide_names)

    bucket = m.group(1)
    prefix = (m.group(2) or "").rstrip("/")
    prefix_with_slash = f"{prefix}/" if prefix else ""

    s3 = boto3.client("s3")
    logger.info("Listing S3 objects under s3://%s/%s", bucket, prefix_with_slash)
    paginator = s3.get_paginator("list_objects_v2")
    existing_keys: set[str] = set()
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix_with_slash):
        for obj in page.get("Contents", []):
            existing_keys.add(obj["Key"])
    logger.info("Found %d objects in S3 prefix", len(existing_keys))

    results = []
    for slide_name in slide_names:
        key = f"{prefix_with_slash}{slide_name}.h5"
        full_uri = f"s3://{bucket}/{key}"
        if key in existing_keys:
            results.append(full_uri)
        else:
            logger.warning("H5 file not found: %s", full_uri)
            results.append("not_found")
    return results


def build_enriched_embeddings(
    embeddings_path: Path,
    data_dir: Path,
    output_path: Path,
    h5_s3_base: str = "",
    local_h5_dir: str = "",
) -> pd.DataFrame:
    """Merge embeddings CSV with clinical/pathological metadata and write output.

    Parameters
    ----------
    embeddings_path:
        Path to the input CSV with ``slide_name``, ``embedding_dim``, and
        ``e_0`` … ``e_1535`` columns.
    data_dir:
        Directory containing ``clinical_data.json`` and
        ``pathological_data.json``.
    output_path:
        Where to write the enriched CSV.
    local_h5_dir:
        Local directory containing pre-downloaded H5 files. When set, the
        ``h5file`` column is populated with absolute local paths (no S3 calls).
        Takes priority over ``h5_s3_base``.
    h5_s3_base:
        S3 base URL for H5 files. Used only when ``local_h5_dir`` is not set.
        Leave both empty to write ``"unknown"`` for all rows.

    Returns the enriched DataFrame.
    """
    # Load embeddings
    emb_df = pd.read_csv(embeddings_path)

    # Remove duplicate slide_name rows, keeping the first occurrence
    dup_count = emb_df["slide_name"].duplicated().sum()
    if dup_count:
        logger.warning(
            "Dropped %d duplicate slide_name row(s) from %s",
            dup_count,
            embeddings_path,
        )
        emb_df = emb_df.drop_duplicates(subset="slide_name", keep="first")

    # Reset index after dedup so that emb_df and the lookup frame share
    # a contiguous 0-based index.  Without this, pd.concat(axis=1) will
    # mis-align rows and produce orphan metadata-only or embedding-only
    # rows wherever the original index had gaps from dropped duplicates.
    emb_df = emb_df.reset_index(drop=True)

    # Extract patient_id from slide_name into a separate Series to avoid
    # fragmentation warnings on the wide embeddings DataFrame.
    patient_ids = emb_df["slide_name"].apply(_extract_patient_id_from_slide)

    # Build a slim lookup frame for joining
    lookup = pd.DataFrame({"patient_id": patient_ids, "_row_idx": range(len(emb_df))})

    # Load clinical & pathological JSON
    with open(data_dir / "clinical_data.json", encoding="utf-8") as f:
        clinical_df = pd.DataFrame(json.load(f))
    with open(data_dir / "pathological_data.json", encoding="utf-8") as f:
        pathological_df = pd.DataFrame(json.load(f))

    # Left-join the slim lookup with metadata (avoids merging wide emb_df)
    lookup = lookup.merge(clinical_df, on="patient_id", how="left")
    lookup = lookup.merge(pathological_df, on="patient_id", how="left")

    # Build metadata columns in the desired order, applying name mapping
    meta_series: list[pd.Series] = []
    for out_col in ENRICHED_METADATA_COLUMNS:
        if out_col == "h5file":
            continue  # handled separately below
        src_col = _ENRICHED_COL_MAP.get(out_col, out_col)
        if src_col in lookup.columns:
            meta_series.append(lookup[src_col].apply(_to_str).rename(out_col))
        else:
            logger.warning("Column %s not found in merged data", src_col)
            meta_series.append(pd.Series("unknown", index=lookup.index, name=out_col))

    # Build h5file column: local path or S3 URI, validated at build time
    h5file_values = _build_h5file_column(emb_df["slide_name"], h5_s3_base, local_h5_dir)
    meta_series.append(pd.Series(h5file_values, name="h5file"))

    # Identify embedding columns
    embedding_cols = ["embedding_dim"] + [
        c for c in emb_df.columns if c.startswith("e_")
    ]

    # Assemble final DataFrame via concat (avoids fragmentation)
    result = pd.concat(
        [emb_df[["slide_name"]]] + meta_series + [emb_df[embedding_cols]],
        axis=1,
    )

    result.to_csv(output_path, index=False)
    logger.info("Wrote enriched embeddings to %s (%d rows)", output_path, len(result))
    return result
