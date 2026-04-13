"""RetrievalService: composite patient similarity retrieval."""

import re
import threading
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from umap_retrieval.data import apply_metadata_filters
from umap_retrieval.retrieval import rank_patients_by_composite

from backend.models import PatientMatch, RetrievalParams, RetrievalResponse

# Metadata columns surfaced in PatientMatch
_DISPLAY_COLUMNS = [
    "primary_tumor_site",
    "histologic_type",
    "hpv_association_p16",
    "pT_stage",
    "pN_stage",
    "smoking_status",
    "age_at_initial_diagnosis",
    "year_of_initial_diagnosis",
]

# Map from DataFrame column names to PatientMatch field names
_COL_TO_FIELD = {
    "primary_tumor_site": "primary_tumor_site",
    "histologic_type": "histologic_type",
    "hpv_association_p16": "hpv_association_p16",
    "pT_stage": "pt_stage",
    "pN_stage": "pn_stage",
    "smoking_status": "smoking_status",
    "age_at_initial_diagnosis": "age_at_initial_diagnosis",
    "year_of_initial_diagnosis": "year_of_initial_diagnosis",
}

# Columns that should be cast to int rather than str
_INT_COLUMNS = {"age_at_initial_diagnosis", "year_of_initial_diagnosis"}


def _patient_id_from_slide(slide_name: str) -> Optional[str]:
    """Extract numeric patient ID from a slide name."""
    m = re.search(r"patient(\d+)", str(slide_name))
    return m.group(1) if m else None


def _extract_patient_ids(metadata_df: pd.DataFrame) -> List[str]:
    """Return sorted unique patient IDs from the metadata DataFrame."""
    ids = set()
    for name in metadata_df["slide_name"]:
        pid = _patient_id_from_slide(name)
        if pid:
            ids.add(pid)
    return sorted(ids)


def _row_to_patient_match(row: pd.Series) -> PatientMatch:
    """Convert a ranked DataFrame row to a PatientMatch model."""
    # Strip the " (query)" suffix that rank_patients_by_composite appends
    raw_pid = str(row["patient_id"])
    patient_id = raw_pid.replace(" (query)", "").strip()

    kwargs: dict = {
        "rank": int(row["rank"]),
        "patient_id": patient_id,
        "composite": float(row["composite"]),
        "slide_sim": float(row["slide_sim"]),
        "comp_sim": float(row["comp_sim"]),
        "meta_sim": float(row["meta_sim"]),
    }

    for col, field in _COL_TO_FIELD.items():
        val = row.get(col, "")
        if val == "" or str(val) in ("nan", "unknown"):
            kwargs[field] = None
        elif col in _INT_COLUMNS:
            try:
                kwargs[field] = int(float(val))
            except (ValueError, TypeError):
                kwargs[field] = None
        else:
            kwargs[field] = str(val)

    return PatientMatch(**kwargs)


class RetrievalService:
    """Composite patient similarity retrieval service.

    Wraps ``rank_patients_by_composite`` from the ``umap_retrieval`` package,
    applying metadata filters, validating the query patient, and truncating
    results to the requested top-k.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        params: RetrievalParams,
        data_service,
        composition_service,
    ) -> RetrievalResponse:
        """Run composite retrieval for a query patient.

        Args:
            params: Validated retrieval parameters.
            data_service: DataService providing embeddings and metadata.
            composition_service: CompositionService providing patient profiles.

        Returns:
            RetrievalResponse with ranked matches and rank-0 query patient.

        Raises:
            RuntimeError: If data or profiles are not loaded.
            ValueError: If the query patient is not in the filtered candidate pool.
        """
        metadata_df = data_service.get_metadata()
        if metadata_df is None:
            raise RuntimeError(
                "Dataset not loaded. Call POST /api/data/load first."
            )

        embeddings: Optional[Dict[str, np.ndarray]] = data_service.get_embeddings()
        if embeddings is None:
            raise RuntimeError(
                "Embeddings not loaded. Call POST /api/data/load first."
            )

        profiles = composition_service.get_profiles()
        if profiles is None:
            raise RuntimeError(
                "Composition profiles not built. Call POST /api/profiles/build first."
            )

        # Apply metadata filters to restrict the candidate pool (Req 4.4)
        if params.filters:
            filtered_meta = apply_metadata_filters(metadata_df, params.filters)
        else:
            filtered_meta = metadata_df

        candidate_patients = _extract_patient_ids(filtered_meta)

        # Validate query patient is in the filtered pool (Req 4.6)
        if params.patient_id not in candidate_patients:
            raise ValueError(
                f"Patient '{params.patient_id}' not found in the filtered candidate pool. "
                f"Check that the patient ID is correct and that the active filters include "
                f"at least one slide for this patient."
            )

        # Run composite ranking
        ranked_df = rank_patients_by_composite(
            query_patient=params.patient_id,
            candidate_patients=candidate_patients,
            embeddings=embeddings,
            composition_profiles=profiles,
            metadata_df=metadata_df,
            alpha=params.alpha,
            beta=params.beta,
            gamma=params.gamma,
            display_columns=_DISPLAY_COLUMNS,
            slide_aggregation=params.slide_aggregation,
        )

        # Deduplicate by patient if requested (Req 4.5)
        # rank_patients_by_composite already operates at patient level, but
        # the deduplication flag is honoured here for explicitness.
        if params.deduplicate_by_patient:
            # patient_id column may have " (query)" suffix for rank-0 row;
            # strip it for dedup comparison
            clean_pids = ranked_df["patient_id"].str.replace(
                r"\s*\(query\)", "", regex=True
            ).str.strip()
            ranked_df = ranked_df[~clean_pids.duplicated(keep="first")].copy()

        # Separate query row (rank == 0) and match rows (rank > 0)
        query_rows = ranked_df[ranked_df["rank"] == 0]
        match_rows = ranked_df[ranked_df["rank"] > 0]

        # Truncate to top-k (Req 4.1)
        top_k_matches = match_rows.head(params.k)

        # Convert to Pydantic models
        matches = [_row_to_patient_match(row) for _, row in top_k_matches.iterrows()]
        query_patient = _row_to_patient_match(query_rows.iloc[0])

        # Find representative slide names for UMAP highlighting
        # Use the slide most similar to the query patient's best slide
        query_slides = [s for s in embeddings if _patient_id_from_slide(s) == params.patient_id]
        
        def _best_slide_for_patient(patient_id: str) -> Optional[str]:
            slides = [s for s in embeddings if _patient_id_from_slide(s) == patient_id]
            if not slides:
                return None
            if query_slides:
                # Pick the slide most similar to any query slide
                best, best_sim = slides[0], -1.0
                for s in slides:
                    for qs in query_slides:
                        v1, v2 = embeddings[s], embeddings[qs]
                        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                        if n1 > 0 and n2 > 0:
                            sim = float(np.dot(v1, v2) / (n1 * n2))
                            if sim > best_sim:
                                best_sim = sim
                                best = s
                return best
            return slides[0]

        query_slide_name = query_slides[0] if query_slides else None
        neighbor_slide_names = [
            s for pid in [m.patient_id for m in matches]
            for s in [_best_slide_for_patient(pid)] if s is not None
        ]

        return RetrievalResponse(
            matches=matches,
            query_patient=query_patient,
            query_slide_name=query_slide_name,
            neighbor_slide_names=neighbor_slide_names,
        )
    def list_patients(self, data_service) -> List[str]:
        """Return a sorted list of all patient IDs from loaded metadata.

        Args:
            data_service: DataService providing metadata.

        Returns:
            Sorted list of patient ID strings.

        Raises:
            RuntimeError: If data is not loaded.
        """
        metadata_df = data_service.get_metadata()
        if metadata_df is None:
            raise RuntimeError(
                "Dataset not loaded. Call POST /api/data/load first."
            )
        return _extract_patient_ids(metadata_df)


# Module-level singleton shared by all routers
retrieval_service = RetrievalService()
