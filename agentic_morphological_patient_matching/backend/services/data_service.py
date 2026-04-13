"""DataService: loads, validates, and caches the patient embedding CSV."""

import re
import threading
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from umap_retrieval.config import FILTERABLE_COLUMNS
from umap_retrieval.data import extract_metadata_and_embeddings, load_and_validate_csv
from umap_retrieval.retrieval import (
    cosine_similarity_null_distribution,
    embedding_silhouette_scores,
)

from backend.models import (
    AppStatus,
    ColumnMeta,
    DataSummary,
    NullDistStats,
    SilhouetteRow,
)

# Project root is four levels up from this file:
# backend/services/data_service.py -> backend/services/ -> backend/ -> 03_create_agent/ -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _validate_path(csv_path: str) -> Path:
    """Validate csv_path against path traversal attacks.

    Raises:
        ValueError: If the path contains traversal sequences or resolves
            outside the project root.
    """
    if "../" in csv_path or "..\\" in csv_path:
        raise ValueError(
            f"Path traversal detected in csv_path: '{csv_path}'"
        )

    resolved = Path(csv_path).resolve()
    if not str(resolved).startswith(str(_PROJECT_ROOT)):
        raise ValueError(
            f"csv_path resolves outside project root: '{resolved}' "
            f"(project root: '{_PROJECT_ROOT}')"
        )

    return resolved


def _count_patients(slide_names) -> int:
    """Count distinct patient IDs extracted from slide names via regex."""
    patient_ids = set()
    for name in slide_names:
        m = re.search(r"patient(\d+)", str(name))
        if m:
            patient_ids.add(m.group(1))
    return len(patient_ids)


class DataService:
    """Singleton service that loads and caches the patient embedding dataset."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._metadata_df = None
        self._embeddings: Optional[Dict[str, np.ndarray]] = None
        self._summary: Optional[DataSummary] = None
        self._csv_path: Optional[str] = None

    def load(self, csv_path: str) -> DataSummary:
        """Load, validate, and cache the CSV dataset.

        Args:
            csv_path: Path to the enriched slide embeddings CSV.

        Returns:
            DataSummary with slide/patient counts, filterable column values,
            null distribution stats, and silhouette scores.

        Raises:
            ValueError: On path traversal or CSV validation failure.
        """
        resolved = _validate_path(csv_path)
        return self._load_resolved(str(resolved), display_path=csv_path)

    def load_from_temp(self, tmp_path: str, display_path: str = "") -> DataSummary:
        """Load from an already-validated temp file path (skips traversal check)."""
        return self._load_resolved(tmp_path, display_path=display_path)

    def _load_resolved(self, resolved_path: str, display_path: str = "") -> DataSummary:

        with self._lock:
            # Load and validate
            df = load_and_validate_csv(resolved_path)

            # Drop slides with all-zero embeddings (failed quality check at embedding time)
            embedding_cols = [col for col in df.columns if col.startswith("e_")]
            df = df.loc[~(df[embedding_cols] == 0).all(axis=1)].copy()

            metadata_df, embeddings = extract_metadata_and_embeddings(df)

            # Null distribution
            null_dist = cosine_similarity_null_distribution(embeddings, sample_n=300)

            # Silhouette scores
            sil_df = embedding_silhouette_scores(
                embeddings, metadata_df, FILTERABLE_COLUMNS
            )

            # Build NullDistStats
            null_stats = NullDistStats(
                mean=null_dist["mean"],
                std=null_dist["std"],
                percentile_5=null_dist["p5"],
                percentile_95=null_dist["p95"],
            )

            # Build filterable column metadata
            filterable_columns = []
            for col in FILTERABLE_COLUMNS:
                if col not in metadata_df.columns:
                    continue
                values = sorted(str(v) for v in metadata_df[col].dropna().unique())
                filterable_columns.append(
                    ColumnMeta(name=col, label=col.replace("_", " ").title(), values=values)
                )

            # Silhouette rows (skip NaN scores)
            silhouette_scores = [
                SilhouetteRow(column=row["label"], score=float(row["silhouette_score"]))
                for _, row in sil_df.iterrows()
                if not (isinstance(row["silhouette_score"], float) and
                        row["silhouette_score"] != row["silhouette_score"])
            ]

            n_slides = len(metadata_df)
            n_patients = _count_patients(metadata_df["slide_name"])

            summary = DataSummary(
                n_slides=n_slides,
                n_patients=n_patients,
                filterable_columns=filterable_columns,
                null_distribution=null_stats,
                silhouette_scores=silhouette_scores,
            )

            # Cache
            self._metadata_df = metadata_df
            self._embeddings = embeddings
            self._summary = summary
            self._csv_path = display_path or resolved_path

        return summary

    def get_summary(self) -> Optional[DataSummary]:
        """Return the cached DataSummary, or None if not loaded."""
        return self._summary

    def get_metadata(self):
        """Return the cached metadata DataFrame, or None if not loaded."""
        return self._metadata_df

    def get_embeddings(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the cached embeddings dict, or None if not loaded."""
        return self._embeddings

    def get_status(self) -> AppStatus:
        """Return current load status."""
        if self._metadata_df is None:
            return AppStatus(loaded=False)

        n_slides = len(self._metadata_df)
        n_patients = _count_patients(self._metadata_df["slide_name"])
        return AppStatus(
            loaded=True,
            n_slides=n_slides,
            n_patients=n_patients,
            csv_path=self._csv_path,
        )

    def update_embeddings(self, new_embeddings: Dict[str, np.ndarray]) -> None:
        """Merge new_embeddings into the cached embeddings dict in-place."""
        with self._lock:
            if self._embeddings is not None:
                self._embeddings.update(new_embeddings)


# Module-level singleton shared by all routers
data_service = DataService()
