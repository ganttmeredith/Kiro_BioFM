"""Dataset and DataLoader utilities for the HANCOCK ABMIL pipeline."""

import io
import re
from concurrent.futures import as_completed

import boto3
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def extract_patient_id(slide_name: str) -> str:
    """Extract patient ID from slide name, e.g. 'TumorCenter_HE_..._patient492' -> 'patient492'."""
    match = re.search(r'(patient\d+)', slide_name)
    return match.group(1) if match else slide_name


class HancockS3PatchDataset(Dataset):
    """
    Dataset for HANCOCK slides. Uses all slides regardless of label availability.

    Data source priority:
      1. S3 .h5 patch embeddings (N x 1536) — full multi-patch bag
      2. Pre-computed CSV embeddings (e_0..e_1535) — single-vector fallback (1 x 1536)

    When return_metadata=True, each item includes a dict of clinical metadata
    for use in SupCon pair mask construction.
    """

    EMBEDDING_COLS = [f'e_{i}' for i in range(1536)]
    METADATA_COLS = [
        "slide_name", "patient_id", "primary_tumor_site", "histologic_type",
        "hpv_association_p16", "pT_stage", "pN_stage", "grading_hpv",
        "survival_status", "recurrence", "smoking_status", "sex",
    ]

    def __init__(self, metadata_df: pd.DataFrame, return_metadata: bool = False, require_s3: bool = False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.return_metadata = return_metadata
        self.require_s3 = require_s3  # if True, raises on S3 failure instead of falling back
        self.s3 = None
        self._has_csv_embeddings = all(c in metadata_df.columns for c in self.EMBEDDING_COLS)

    def _init_s3(self):
        if self.s3 is None:
            # Use the lab profile if running locally
            import os
            profile = os.environ.get("AWS_PROFILE", None)
            session = boto3.Session(profile_name=profile) if profile else boto3.Session()
            self.s3 = session.client('s3')

    def _load_from_s3(self, s3_uri: str) -> np.ndarray:
        match = re.match(r's3://([^/]+)/(.*)', s3_uri)
        if not match:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")
        bucket, key = match.groups()
        self._init_s3()
        response = self.s3.get_object(Bucket=bucket, Key=key)
        stream = io.BytesIO(response['Body'].read())
        with h5py.File(stream, 'r') as f:
            if 'embeddings' in f:
                dataset_key = 'embeddings'
            elif 'features' in f:
                dataset_key = 'features'
            else:
                raise KeyError(f"No 'embeddings' or 'features' key in {s3_uri}. Keys: {list(f.keys())}")
            data = f[dataset_key][:]
            if data.shape[0] == 0:
                raise ValueError(f"Empty patch bag in {s3_uri} (0 patches)")
            return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # --- Load patch features ---
        features = None
        s3_uri = row.get('h5file', '')
        s3_error = None
        if isinstance(s3_uri, str) and s3_uri.startswith('s3://'):
            try:
                features = self._load_from_s3(s3_uri)
            except Exception as e:
                s3_error = e
                features = None

        # Fall back to CSV embeddings if S3 failed (e.g. corrupt/empty H5)
        if features is None and self._has_csv_embeddings:
            if s3_error:
                self._fallback_count = getattr(self, '_fallback_count', 0) + 1
            features = row[self.EMBEDDING_COLS].values.astype(np.float32).reshape(1, 1536)

        if features is None:
            raise RuntimeError(
                f"No features available for {row['slide_name']} — "
                f"S3 failed ({s3_error}) and no CSV embeddings present."
            )

        feature_tensor = torch.from_numpy(features.astype(np.float32))

        # --- Label (for any downstream classification use) ---
        hpv = str(row.get('hpv_association_p16', 'not_tested')).lower()
        label = torch.tensor(1.0 if hpv == 'positive' else 0.0).unsqueeze(0)

        if not self.return_metadata:
            return feature_tensor, label, row['slide_name']

        # --- Metadata dict for SupCon pair mask ---
        meta_row = {
            col: row[col] if col in row.index else ""
            for col in self.METADATA_COLS
        }
        return feature_tensor, label, row['slide_name'], meta_row


def identify_corrupt_slides(
    metadata_df: pd.DataFrame,
    size_threshold_bytes: int = 10000,
    max_workers: int = 32,
) -> set:
    """
    Identify slides whose H5 files are likely corrupt (under size_threshold_bytes).
    Uses parallel boto3 head_object calls — no download required.
    Returns a set of slide_names to exclude from training.
    """
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    profile = os.environ.get("AWS_PROFILE", None)
    session = boto3.Session(profile_name=profile) if profile else boto3.Session()
    s3 = session.client('s3')

    s3_rows = metadata_df[metadata_df["h5file"].str.startswith("s3://", na=False)]
    print(f"Checking {len(s3_rows)} H5 files for corruption ({max_workers} parallel workers)...")

    def _check(slide_name, uri):
        m = re.match(r's3://([^/]+)/(.*)', uri)
        if not m:
            return slide_name, True
        bucket, key = m.groups()
        try:
            resp = s3.head_object(Bucket=bucket, Key=key)
            return slide_name, resp["ContentLength"] < size_threshold_bytes
        except Exception:
            return slide_name, True

    corrupt = set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_check, row["slide_name"], row["h5file"]): row["slide_name"]
                   for _, row in s3_rows.iterrows()}
        for f in as_completed(futures):
            slide_name, is_corrupt = f.result()
            if is_corrupt:
                corrupt.add(slide_name)

    print(f"Found {len(corrupt)} corrupt/empty H5 files — excluded from training.")
    return corrupt


def load_hancock_manifest(csv_path: str = 'enriched_slide_embeddings.csv') -> pd.DataFrame:
    """Load the enriched slide embeddings CSV and add a patient_id column."""
    df = pd.read_csv(csv_path)
    df = df.copy()  # defragment before adding columns
    df['patient_id'] = df['slide_name'].apply(extract_patient_id)
    return df
