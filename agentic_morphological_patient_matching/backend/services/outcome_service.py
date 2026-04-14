"""OutcomeService: classifies patients into Non_Responder / Responder cohorts
based on researcher-selected clinical outcome criteria and provides cohort summaries.
"""

from __future__ import annotations

import csv
import io
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import plotly.graph_objects as go
import plotly.io as pio

import re

import plotly.express as px
import umap as umap_lib
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from backend.models import (
    AnalyteComparison,
    BiomarkerRequest,
    BiomarkerResponse,
    BoxPlotRequest,
    BoxPlotResponse,
    ClassifyRequest,
    ClassifyResponse,
    CohortSummary,
    DeviationCell,
    InterpretRequest,
    InterpretResponse,
    OutcomeCriteria,
    OutcomeUMAPRequest,
    OutcomeUMAPResponse,
)

# Project root: backend/services/outcome_service.py -> services/ -> backend/ -> app/ -> <root>
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


class OutcomeService:
    """Singleton service for outcome-based cohort classification and biomarker analysis."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._clinical_df: Optional[pd.DataFrame] = None
        self._blood_df: Optional[pd.DataFrame] = None
        self._reference_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Data loading (with caching)
    # ------------------------------------------------------------------

    def _load_clinical_data(self) -> pd.DataFrame:
        """Load and cache clinical_data.json from the project root."""
        if self._clinical_df is not None:
            return self._clinical_df

        path = _PROJECT_ROOT / "clinical_data.json"
        try:
            df = pd.read_json(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load clinical data: {exc}") from exc

        # Ensure patient_id is string for consistent joins
        df["patient_id"] = df["patient_id"].astype(str)
        self._clinical_df = df
        return df

    def _load_blood_data(self) -> pd.DataFrame:
        """Load and cache blood_data.json from the project root."""
        if self._blood_df is not None:
            return self._blood_df

        path = _PROJECT_ROOT / "blood_data.json"
        try:
            df = pd.read_json(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load biomarker data: {exc}") from exc

        df["patient_id"] = df["patient_id"].astype(str)
        self._blood_df = df
        return df

    def _load_reference_ranges(self) -> pd.DataFrame:
        """Load and cache blood_data_reference_ranges.json from the project root."""
        if self._reference_df is not None:
            return self._reference_df

        path = _PROJECT_ROOT / "blood_data_reference_ranges.json"
        try:
            df = pd.read_json(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load reference ranges: {exc}") from exc

        self._reference_df = df
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _filter_insufficient_data(blood_df: pd.DataFrame, min_analytes: int = 5) -> Set[str]:
        """Return patient IDs with fewer than *min_analytes* distinct analyte measurements."""
        counts = blood_df.groupby("patient_id")["analyte_name"].nunique()
        return set(counts[counts < min_analytes].index)

    @staticmethod
    def _apply_outcome_criteria(df: pd.DataFrame, criteria: OutcomeCriteria) -> pd.Series:
        """Return a boolean mask where True = Non_Responder for each patient row.

        A patient is Non_Responder if **at least one** selected criterion matches.
        """
        mask = pd.Series(False, index=df.index)

        if criteria.deceased:
            mask = mask | (df["survival_status"] == "deceased")

        if criteria.tumor_caused_death:
            mask = mask | (df["survival_status_with_cause"] == "deceased_due_to_tumor")

        if criteria.recurrence:
            mask = mask | (df["recurrence"] == "yes")

        if criteria.progression:
            mask = mask | (df["progress_1"] == "yes") | (df["progress_2"] == "yes")

        if criteria.metastasis:
            mask = mask | df["metastasis_1_locations"].notna()

        return mask

    @staticmethod
    def _build_cohort_summary(df: pd.DataFrame) -> CohortSummary:
        """Compute count, mean age, and sex distribution for a patient DataFrame."""
        count = len(df)
        mean_age: Optional[float] = None
        sex_distribution: Dict[str, int] = {}

        if count > 0:
            if "age_at_initial_diagnosis" in df.columns:
                ages = df["age_at_initial_diagnosis"].dropna()
                if len(ages) > 0:
                    mean_age = float(np.round(ages.mean(), 2))

            if "sex" in df.columns:
                sex_distribution = df["sex"].value_counts().to_dict()
                # Ensure values are plain ints for JSON serialization
                sex_distribution = {str(k): int(v) for k, v in sex_distribution.items()}

        return CohortSummary(
            count=count,
            mean_age=mean_age,
            sex_distribution=sex_distribution,
        )

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_mann_whitney(group_a: np.ndarray, group_b: np.ndarray) -> Tuple[float, float]:
        """Return (U statistic, p-value) from a two-sided Mann-Whitney U test."""
        u_stat, p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")
        return float(u_stat), float(p_value)

    @staticmethod
    def _compute_cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
        """Return Cohen's d effect size using pooled standard deviation."""
        n_a, n_b = len(group_a), len(group_b)
        var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
        pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
        if pooled_std == 0:
            return 0.0
        return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)

    @staticmethod
    def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
        """Apply Benjamini-Hochberg FDR correction and return adjusted p-values."""
        _reject, adjusted, _alphac_sidak, _alphac_bonf = multipletests(
            p_values, method="fdr_bh"
        )
        return adjusted

    # ------------------------------------------------------------------
    # Deviation scores
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_deviation_scores(
        blood_df: pd.DataFrame,
        ref_df: pd.DataFrame,
        clinical_df: pd.DataFrame,
    ) -> List[DeviationCell]:
        """Compute per-patient-analyte deviation scores using sex-appropriate reference ranges.

        Formula: deviation_score = (value - midpoint) / (range_width / 2)
        where midpoint = (ref_min + ref_max) / 2 and range_width = ref_max - ref_min.

        Returns None for the deviation_score when no reference range exists for
        the patient's sex.
        """
        # Build a lookup: patient_id -> sex
        sex_lookup: Dict[str, str] = dict(
            zip(clinical_df["patient_id"].astype(str), clinical_df["sex"].astype(str))
        )

        # Build a lookup: analyte_name -> {male: (min, max), female: (min, max)}
        ref_lookup: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for _, row in ref_df.iterrows():
            analyte = str(row["analyte_name"])
            ref_lookup[analyte] = {}
            if pd.notna(row.get("normal_male_min")) and pd.notna(row.get("normal_male_max")):
                ref_lookup[analyte]["male"] = (float(row["normal_male_min"]), float(row["normal_male_max"]))
            if pd.notna(row.get("normal_female_min")) and pd.notna(row.get("normal_female_max")):
                ref_lookup[analyte]["female"] = (float(row["normal_female_min"]), float(row["normal_female_max"]))

        cells: List[DeviationCell] = []
        for _, row in blood_df.iterrows():
            pid = str(row["patient_id"])
            analyte = str(row["analyte_name"])
            value = row["value"]

            sex = sex_lookup.get(pid)
            ref_ranges = ref_lookup.get(analyte, {})
            sex_range = ref_ranges.get(sex) if sex else None

            if sex_range is None or pd.isna(value):
                cells.append(DeviationCell(
                    patient_id=pid,
                    analyte_name=analyte,
                    deviation_score=None,
                    cohort="",  # cohort label set by caller
                ))
            else:
                ref_min, ref_max = sex_range
                midpoint = (ref_min + ref_max) / 2.0
                range_width = ref_max - ref_min
                half_range = range_width / 2.0
                if half_range == 0:
                    deviation = 0.0
                else:
                    deviation = (float(value) - midpoint) / half_range
                cells.append(DeviationCell(
                    patient_id=pid,
                    analyte_name=analyte,
                    deviation_score=round(deviation, 6),
                    cohort="",  # cohort label set by caller
                ))

        return cells

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_box_plot(self, params: BoxPlotRequest) -> BoxPlotResponse:
        """Generate a Plotly box plot for a selected analyte comparing cohorts.

        Shows Non_Responder vs Responder distributions side by side with
        jitter overlay and horizontal reference lines from sex-appropriate
        reference ranges when available.
        """
        criteria = params.criteria
        analyte_name = params.analyte_name

        if not any([
            criteria.deceased,
            criteria.tumor_caused_death,
            criteria.recurrence,
            criteria.progression,
            criteria.metastasis,
        ]):
            raise ValueError("At least one outcome criterion must be selected")

        with self._lock:
            clinical_df = self._load_clinical_data()
            blood_df = self._load_blood_data()
            ref_df = self._load_reference_ranges()

        # Exclude patients with insufficient data
        excluded_ids = self._filter_insufficient_data(blood_df)
        eligible_clinical = clinical_df[~clinical_df["patient_id"].isin(excluded_ids)]
        eligible_blood = blood_df[~blood_df["patient_id"].isin(excluded_ids)]

        # Partition into cohorts
        nr_mask = self._apply_outcome_criteria(eligible_clinical, criteria)
        nr_ids = set(eligible_clinical.loc[nr_mask, "patient_id"])
        r_ids = set(eligible_clinical.loc[~nr_mask, "patient_id"])

        # Filter blood data for the selected analyte
        analyte_data = eligible_blood[eligible_blood["analyte_name"] == analyte_name]
        nr_values = analyte_data.loc[
            analyte_data["patient_id"].isin(nr_ids), "value"
        ].dropna().values
        r_values = analyte_data.loc[
            analyte_data["patient_id"].isin(r_ids), "value"
        ].dropna().values

        # Build Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Box(
            y=nr_values,
            name="Non_Responder",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.5,
            marker=dict(color="rgba(239, 85, 59, 0.7)"),
            line=dict(color="rgb(239, 85, 59)"),
        ))

        fig.add_trace(go.Box(
            y=r_values,
            name="Responder",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.5,
            marker=dict(color="rgba(99, 110, 250, 0.7)"),
            line=dict(color="rgb(99, 110, 250)"),
        ))

        # Add reference range lines if available
        has_reference_range = False
        ref_row = ref_df[ref_df["analyte_name"] == analyte_name]

        if not ref_row.empty:
            row = ref_row.iloc[0]

            # Male reference range
            male_min = row.get("normal_male_min")
            male_max = row.get("normal_male_max")
            if pd.notna(male_min) and pd.notna(male_max):
                has_reference_range = True
                fig.add_hline(
                    y=float(male_min),
                    line_dash="dash",
                    line_color="steelblue",
                    annotation_text=f"Male min ({male_min})",
                    annotation_position="top left",
                )
                fig.add_hline(
                    y=float(male_max),
                    line_dash="dash",
                    line_color="steelblue",
                    annotation_text=f"Male max ({male_max})",
                    annotation_position="top left",
                )

            # Female reference range
            female_min = row.get("normal_female_min")
            female_max = row.get("normal_female_max")
            if pd.notna(female_min) and pd.notna(female_max):
                has_reference_range = True
                fig.add_hline(
                    y=float(female_min),
                    line_dash="dot",
                    line_color="orchid",
                    annotation_text=f"Female min ({female_min})",
                    annotation_position="top right",
                )
                fig.add_hline(
                    y=float(female_max),
                    line_dash="dot",
                    line_color="orchid",
                    annotation_text=f"Female max ({female_max})",
                    annotation_position="top right",
                )

        # Get unit from blood data if available
        unit = ""
        if not analyte_data.empty and "unit" in analyte_data.columns:
            unit_val = analyte_data["unit"].dropna().iloc[0] if not analyte_data["unit"].dropna().empty else ""
            unit = f" ({unit_val})" if unit_val else ""

        fig.update_layout(
            title=f"{analyte_name} Distribution by Cohort",
            yaxis_title=f"{analyte_name}{unit}",
            xaxis_title="Cohort",
            showlegend=True,
            template="plotly_white",
        )

        plotly_json = pio.to_json(fig)

        return BoxPlotResponse(
            plotly_json=plotly_json,
            has_reference_range=has_reference_range,
        )

    def classify(self, params: ClassifyRequest) -> ClassifyResponse:
        """Assign patients to Non_Responder / Responder cohorts.

        Raises:
            ValueError: If no outcome criteria are selected.
        """
        criteria = params.criteria

        # Validate at least one criterion is selected
        if not any([
            criteria.deceased,
            criteria.tumor_caused_death,
            criteria.recurrence,
            criteria.progression,
            criteria.metastasis,
        ]):
            raise ValueError("At least one outcome criterion must be selected")

        with self._lock:
            clinical_df = self._load_clinical_data()
            blood_df = self._load_blood_data()

        # Exclude patients with insufficient analyte measurements
        excluded_ids = self._filter_insufficient_data(blood_df)
        eligible_df = clinical_df[~clinical_df["patient_id"].isin(excluded_ids)].copy()
        excluded_count = len(excluded_ids)

        # Apply outcome criteria
        nr_mask = self._apply_outcome_criteria(eligible_df, criteria)

        non_responder_df = eligible_df[nr_mask]
        responder_df = eligible_df[~nr_mask]

        return ClassifyResponse(
            non_responder=self._build_cohort_summary(non_responder_df),
            responder=self._build_cohort_summary(responder_df),
            non_responder_ids=non_responder_df["patient_id"].tolist(),
            responder_ids=responder_df["patient_id"].tolist(),
            excluded_count=excluded_count,
        )

    def analyze_biomarkers(self, params: BiomarkerRequest) -> BiomarkerResponse:
        """Run per-analyte statistical comparison between Non_Responder and Responder cohorts."""
        criteria = params.criteria

        if not any([
            criteria.deceased,
            criteria.tumor_caused_death,
            criteria.recurrence,
            criteria.progression,
            criteria.metastasis,
        ]):
            raise ValueError("At least one outcome criterion must be selected")

        with self._lock:
            clinical_df = self._load_clinical_data()
            blood_df = self._load_blood_data()
            ref_df = self._load_reference_ranges()

        # Exclude patients with insufficient analyte measurements
        excluded_ids = self._filter_insufficient_data(blood_df)
        eligible_clinical = clinical_df[~clinical_df["patient_id"].isin(excluded_ids)]
        eligible_blood = blood_df[~blood_df["patient_id"].isin(excluded_ids)]

        # Partition patients into cohorts
        nr_mask = self._apply_outcome_criteria(eligible_clinical, criteria)
        nr_ids = set(eligible_clinical.loc[nr_mask, "patient_id"])
        r_ids = set(eligible_clinical.loc[~nr_mask, "patient_id"])

        # Per-analyte comparison
        comparisons: List[AnalyteComparison] = []
        p_values: List[float] = []
        analyte_results: List[dict] = []

        for analyte_name, analyte_group in eligible_blood.groupby("analyte_name"):
            nr_values = analyte_group.loc[
                analyte_group["patient_id"].isin(nr_ids), "value"
            ].dropna().values
            r_values = analyte_group.loc[
                analyte_group["patient_id"].isin(r_ids), "value"
            ].dropna().values

            # Exclude analytes with <2 values in either cohort
            if len(nr_values) < 2 or len(r_values) < 2:
                continue

            u_stat, p_val = self._compute_mann_whitney(nr_values, r_values)
            effect_size = self._compute_cohens_d(nr_values, r_values)

            # Get LOINC code and group from the first row of this analyte
            first_row = analyte_group.iloc[0]
            loinc_code = str(first_row.get("LOINC_code", ""))
            group = str(first_row.get("group", ""))

            p_values.append(p_val)
            analyte_results.append({
                "analyte_name": str(analyte_name),
                "loinc_code": loinc_code,
                "group": group,
                "non_responder_mean": float(np.mean(nr_values)),
                "non_responder_std": float(np.std(nr_values, ddof=1)),
                "responder_mean": float(np.mean(r_values)),
                "responder_std": float(np.std(r_values, ddof=1)),
                "p_value": p_val,
                "effect_size": effect_size,
            })

        # Apply Benjamini-Hochberg correction
        if p_values:
            adjusted = self._benjamini_hochberg(np.array(p_values))
            for i, result in enumerate(analyte_results):
                adj_p = float(adjusted[i])
                comparisons.append(AnalyteComparison(
                    **result,
                    adjusted_p_value=adj_p,
                    significant=adj_p < 0.05,
                ))

        # Compute deviation scores for all eligible patients
        deviation_cells = self._compute_deviation_scores(eligible_blood, ref_df, eligible_clinical)

        # Assign cohort labels to each deviation cell
        for cell in deviation_cells:
            if cell.patient_id in nr_ids:
                cell.cohort = "non_responder"
            elif cell.patient_id in r_ids:
                cell.cohort = "responder"

        return BiomarkerResponse(
            comparisons=comparisons,
            deviation_scores=deviation_cells,
        )

    # ------------------------------------------------------------------
    # Outcome UMAP
    # ------------------------------------------------------------------

    @staticmethod
    def _build_clinical_features(
        clinical_df: pd.DataFrame,
        blood_df: pd.DataFrame,
        patient_ids: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build a clinical feature matrix for the given patients.

        Features: age (normalized), sex (one-hot), smoking_status (one-hot),
        first_treatment_modality (one-hot), plus normalized blood biomarker
        values (pivoted by analyte, NaN filled with 0).

        Returns (feature_matrix, ordered_patient_ids) where rows correspond
        to ordered_patient_ids.
        """
        cdf = clinical_df[clinical_df["patient_id"].isin(set(patient_ids))].copy()
        cdf = cdf.set_index("patient_id")

        # One-hot encode categorical columns
        cat_cols = []
        for col in ["sex", "smoking_status", "first_treatment_modality"]:
            if col in cdf.columns:
                dummies = pd.get_dummies(cdf[col], prefix=col, dummy_na=False)
                cat_cols.append(dummies)

        # Age (will be normalized later)
        age_series = cdf["age_at_initial_diagnosis"].fillna(
            cdf["age_at_initial_diagnosis"].median()
        )

        # Blood biomarker pivot: rows=patient_id, cols=analyte_name, values=mean value
        bdf = blood_df[blood_df["patient_id"].isin(set(patient_ids))].copy()
        blood_pivot = bdf.pivot_table(
            index="patient_id", columns="analyte_name", values="value", aggfunc="mean"
        ).fillna(0)

        # Align all features to the same patient order
        ordered_pids = sorted(set(cdf.index) & set(patient_ids))
        # Ensure blood_pivot has all patients (some may have no blood data)
        blood_pivot = blood_pivot.reindex(ordered_pids, fill_value=0)

        parts = [pd.DataFrame({"age": age_series}).reindex(ordered_pids)]
        for d in cat_cols:
            parts.append(d.reindex(ordered_pids, fill_value=0))
        parts.append(blood_pivot.reindex(ordered_pids, fill_value=0))

        feature_df = pd.concat(parts, axis=1).fillna(0)

        # Normalize all features
        scaler = StandardScaler()
        matrix = scaler.fit_transform(feature_df.values)

        return matrix, ordered_pids

    @staticmethod
    def _build_imaging_features(
        embeddings: Dict[str, np.ndarray],
        patient_ids: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build patient-level imaging feature matrix by mean-pooling slide embeddings.

        Returns (feature_matrix, ordered_patient_ids).
        """
        pid_set = set(patient_ids)
        # Group slide embeddings by patient
        buckets: Dict[str, List[np.ndarray]] = {}
        for slide_name, vec in embeddings.items():
            m = re.search(r"patient(\d+)", slide_name)
            if m:
                # Strip leading zeros to match clinical data format ("001" -> "1")
                pid = str(int(m.group(1)))
                if pid in pid_set:
                    buckets.setdefault(pid, []).append(vec)

        # Mean-pool per patient
        ordered_pids = sorted(pid_set & set(buckets.keys()), key=lambda x: int(x))
        if not ordered_pids:
            return np.empty((0, 0)), []
        matrix = np.stack([np.mean(np.stack(buckets[pid]), axis=0) for pid in ordered_pids])

        return matrix, ordered_pids

    def run_outcome_umap(
        self, params: OutcomeUMAPRequest, data_service
    ) -> OutcomeUMAPResponse:
        """Generate UMAP projection colored by cohort membership or clinical variable.

        Supports three modalities:
        - "imaging": 1536-dim ABMIL morphological embeddings
        - "clinical": age, sex, smoking_status, treatment_modality, blood biomarkers
        - "multimodal": concatenation of imaging + clinical vectors
        """
        criteria = params.criteria

        if not any([
            criteria.deceased,
            criteria.tumor_caused_death,
            criteria.recurrence,
            criteria.progression,
            criteria.metastasis,
        ]):
            raise ValueError("At least one outcome criterion must be selected")

        with self._lock:
            clinical_df = self._load_clinical_data()
            blood_df = self._load_blood_data()

        # Exclude patients with insufficient data
        excluded_ids = self._filter_insufficient_data(blood_df)
        eligible_clinical = clinical_df[~clinical_df["patient_id"].isin(excluded_ids)]

        # Partition into cohorts
        nr_mask = self._apply_outcome_criteria(eligible_clinical, criteria)
        nr_ids = set(eligible_clinical.loc[nr_mask, "patient_id"])
        r_ids = set(eligible_clinical.loc[~nr_mask, "patient_id"])
        all_eligible_ids = list(nr_ids | r_ids)

        # Build feature matrix based on modality
        if params.modality == "imaging":
            embeddings = data_service.get_embeddings()
            if embeddings is None:
                raise RuntimeError("Dataset not loaded. Call POST /api/data/load first.")
            feature_matrix, ordered_pids = self._build_imaging_features(
                embeddings, all_eligible_ids
            )
        elif params.modality == "clinical":
            eligible_blood = blood_df[~blood_df["patient_id"].isin(excluded_ids)]
            feature_matrix, ordered_pids = self._build_clinical_features(
                eligible_clinical, eligible_blood, all_eligible_ids
            )
        elif params.modality == "multimodal":
            embeddings = data_service.get_embeddings()
            if embeddings is None:
                raise RuntimeError("Dataset not loaded. Call POST /api/data/load first.")
            img_matrix, img_pids = self._build_imaging_features(
                embeddings, all_eligible_ids
            )
            eligible_blood = blood_df[~blood_df["patient_id"].isin(excluded_ids)]
            clin_matrix, clin_pids = self._build_clinical_features(
                eligible_clinical, eligible_blood, all_eligible_ids
            )
            # Intersect patients that have both imaging and clinical data
            common_pids = sorted(set(img_pids) & set(clin_pids))
            img_idx = {pid: i for i, pid in enumerate(img_pids)}
            clin_idx = {pid: i for i, pid in enumerate(clin_pids)}
            img_aligned = np.stack([img_matrix[img_idx[pid]] for pid in common_pids])
            clin_aligned = np.stack([clin_matrix[clin_idx[pid]] for pid in common_pids])
            feature_matrix = np.hstack([img_aligned, clin_aligned])
            ordered_pids = common_pids
        else:
            raise ValueError(f"Unknown modality: {params.modality}")

        if len(ordered_pids) < 2:
            raise ValueError("Cohort has fewer than 2 patients. Adjust criteria.")

        # Run UMAP
        n_neighbors = min(params.n_neighbors, len(ordered_pids) - 1)
        reducer = umap_lib.UMAP(
            n_neighbors=n_neighbors,
            min_dist=params.min_dist,
            random_state=42,
        )
        coords = reducer.fit_transform(feature_matrix)

        # Assign cohort labels
        cohort_labels = [
            "Non_Responder" if pid in nr_ids else "Responder"
            for pid in ordered_pids
        ]

        # Compute silhouette score for cohort separation
        if len(set(cohort_labels)) > 1:
            sil_score = float(silhouette_score(coords, cohort_labels))
        else:
            sil_score = 0.0

        # Build clinical lookup for tooltips
        clin_lookup = eligible_clinical.set_index("patient_id")

        # Determine color values
        color_by = params.color_by
        if color_by == "cohort":
            color_values = cohort_labels
        else:
            color_values = []
            for pid in ordered_pids:
                if pid in clin_lookup.index:
                    val = clin_lookup.loc[pid, color_by] if color_by in clin_lookup.columns else "N/A"
                    if pd.isna(val):
                        val = "N/A"
                    color_values.append(str(val))
                else:
                    color_values.append("N/A")

        # Build hover text with required tooltip fields
        hover_texts = []
        for i, pid in enumerate(ordered_pids):
            parts = [f"Patient: {pid}", f"Cohort: {cohort_labels[i]}"]
            if pid in clin_lookup.index:
                row = clin_lookup.loc[pid]
                sv = row.get("survival_status", "N/A")
                parts.append(f"Survival: {sv if pd.notna(sv) else 'N/A'}")
                rec = row.get("recurrence", "N/A")
                parts.append(f"Recurrence: {rec if pd.notna(rec) else 'N/A'}")
                age = row.get("age_at_initial_diagnosis", "N/A")
                parts.append(f"Age: {age if pd.notna(age) else 'N/A'}")
            hover_texts.append("<br>".join(parts))

        # Build Plotly scatter plot
        palette = px.colors.qualitative.Plotly
        unique_groups = sorted(set(color_values))
        color_map = {}
        # Use red/blue for cohort coloring
        if color_by == "cohort":
            color_map = {
                "Non_Responder": "rgba(239, 85, 59, 0.8)",
                "Responder": "rgba(99, 110, 250, 0.8)",
            }
        else:
            for idx, group in enumerate(unique_groups):
                if group == "N/A":
                    color_map[group] = "lightgray"
                else:
                    color_map[group] = palette[idx % len(palette)]

        fig = go.Figure()
        for group in unique_groups:
            mask = [cv == group for cv in color_values]
            group_x = coords[mask, 0]
            group_y = coords[mask, 1]
            group_hover = [ht for ht, m in zip(hover_texts, mask) if m]
            fig.add_trace(go.Scatter(
                x=group_x,
                y=group_y,
                mode="markers",
                marker=dict(size=7, color=color_map.get(group, "gray")),
                name=str(group),
                text=group_hover,
                hoverinfo="text",
            ))

        fig.update_layout(
            title=f"Outcome UMAP — {params.modality} features, colored by {color_by}"
                  f"<br><sub>Silhouette score: {sil_score:.3f} | {len(ordered_pids)} patients</sub>",
            xaxis_title="UMAP 1",
            yaxis_title="UMAP 2",
            legend_title=color_by,
            template="plotly_white",
            width=900,
            height=650,
        )

        plotly_json = pio.to_json(fig)

        return OutcomeUMAPResponse(
            plotly_json=plotly_json,
            n_points=len(ordered_pids),
            silhouette_score=round(sil_score, 4),
        )

    # ------------------------------------------------------------------
    # AI Interpretation via Bedrock
    # ------------------------------------------------------------------

    @staticmethod
    def _build_interpretation_prompt(context_type: str, context_data: Dict[str, Any]) -> str:
        """Construct a domain-specific prompt for Bedrock interpretation."""
        if context_type == "biomarker_stats":
            analytes = context_data.get("significant_analytes", [])
            total = context_data.get("total_analytes", 0)
            lines = [
                "You are a clinical biomarker analyst reviewing blood analyte comparisons "
                "between Non_Responder (unfavorable outcome) and Responder cohorts in a "
                "head and neck cancer study.",
                "",
                f"Out of {total} analytes tested, the following were statistically significant "
                f"(adjusted p-value < 0.05):",
                "",
            ]
            for a in analytes:
                lines.append(
                    f"- {a.get('analyte_name', 'Unknown')}: "
                    f"p={a.get('adjusted_p_value', 'N/A')}, "
                    f"effect size (Cohen's d)={a.get('effect_size', 'N/A')}, "
                    f"Non_Responder mean={a.get('non_responder_mean', 'N/A')}, "
                    f"Responder mean={a.get('responder_mean', 'N/A')}"
                )
            lines.append("")
            lines.append(
                "Provide a concise clinical interpretation: which biomarkers most "
                "differentiate the cohorts, what the effect sizes suggest about magnitude, "
                "and any potential clinical relevance of these findings for head and neck "
                "cancer prognosis. Keep the response under 300 words."
            )
            return "\n".join(lines)

        elif context_type == "umap_clusters":
            silhouette = context_data.get("silhouette_score", "N/A")
            modality = context_data.get("modality", "unknown")
            n_points = context_data.get("n_points", "N/A")
            nr_count = context_data.get("non_responder_count", "N/A")
            r_count = context_data.get("responder_count", "N/A")
            lines = [
                "You are a clinical data scientist interpreting UMAP dimensionality "
                "reduction results for a head and neck cancer patient cohort.",
                "",
                f"Modality: {modality}",
                f"Number of patients: {n_points}",
                f"Non_Responder count: {nr_count}",
                f"Responder count: {r_count}",
                f"Silhouette score (cohort separation): {silhouette}",
                "",
                "Provide a concise interpretation of the spatial grouping patterns: "
                "how well do the cohorts separate in this feature space, what does the "
                "silhouette score indicate about cluster quality, and what might the "
                "spatial patterns suggest about the relationship between the selected "
                "modality features and patient outcomes. Keep the response under 300 words."
            ]
            return "\n".join(lines)

        # Fallback for unknown context types
        return (
            f"Interpret the following {context_type} data in the context of "
            f"head and neck cancer biomarker discovery:\n\n{context_data}"
        )

    def interpret_results(self, params: InterpretRequest) -> InterpretResponse:
        """Send statistical results or UMAP metrics to Bedrock Claude for clinical interpretation."""
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-west-2"),
        )
        prompt = self._build_interpretation_prompt(params.context_type, params.context_data)
        response = client.converse(
            modelId="us.anthropic.claude-sonnet-4-6",
            messages=[{"role": "user", "content": [{"text": prompt}]}],
        )
        interpretation = response["output"]["message"]["content"][0]["text"]
        return InterpretResponse(
            interpretation=interpretation,
            context_type=params.context_type,
        )

    def export_csv(self, params: BiomarkerRequest) -> str:
        """Return a CSV string of the full biomarker comparison table."""
        response = self.analyze_biomarkers(params)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "analyte_name",
            "non_responder_mean",
            "non_responder_std",
            "responder_mean",
            "responder_std",
            "p_value",
            "adjusted_p_value",
            "effect_size",
        ])
        for c in response.comparisons:
            writer.writerow([
                c.analyte_name,
                c.non_responder_mean,
                c.non_responder_std,
                c.responder_mean,
                c.responder_std,
                c.p_value,
                c.adjusted_p_value,
                c.effect_size,
            ])
        return buf.getvalue()


# Module-level singleton shared by all routers
outcome_service = OutcomeService()
