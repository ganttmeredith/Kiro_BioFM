"""ChatService — Strands agentic chat with tools for retrieval, UMAP, and clustering."""

import json
import os
import queue
import threading
from typing import Generator, List

from strands import Agent, tool
from strands.models import BedrockModel

from backend.models import AppContext, ChatMessage, ClusterParams, RetrievalParams, UMAPParams
from backend.services.context_builder import build_system_prompt

DEFAULT_MODEL_ID = "us.anthropic.claude-sonnet-4-6"


def _resolve_patient_id(raw: str, all_patients: list) -> str | None:
    """Resolve a user-supplied patient ID to the canonical form in the dataset.

    Handles zero-padding differences: "15", "015", "0015" all match "015"
    if "015" is the stored form. Strips any non-digit prefix/suffix too.

    Args:
        raw: The patient ID as supplied by the user or the model.
        all_patients: Sorted list of canonical patient IDs from the dataset.

    Returns:
        The canonical patient ID string, or None if no match found.
    """
    # Extract digits only from the raw input
    digits = ''.join(c for c in str(raw) if c.isdigit())
    if not digits:
        return None

    numeric = int(digits)

    # 1. Exact match first
    if raw in all_patients:
        return raw

    # 2. Match by numeric value (handles leading-zero variants)
    for pid in all_patients:
        try:
            if int(pid) == numeric:
                return pid
        except ValueError:
            continue

    return None


def _make_tools(app_context: AppContext, artifact_queue: queue.Queue):
    """Return tool functions closed over the current app context and service singletons.

    artifact_queue receives SSE artifact lines so the streaming generator can
    emit them inline alongside text tokens.
    """
    from backend.services.cluster_service import cluster_service
    from backend.services.composition_service import composition_service
    from backend.services.data_service import data_service
    from backend.services.retrieval_service import retrieval_service
    from backend.services.umap_service import umap_service

    @tool
    def search_patients(  # noqa: PLR0912
        age_min: int = 0,
        age_max: int = 200,
        primary_tumor_site: str = "",
        histologic_type: str = "",
        hpv_association_p16: str = "",
        pt_stage: str = "",
        pn_stage: str = "",
        smoking_status: str = "",
        sex: str = "",
        survival_status: str = "",
        recurrence: str = "",
        year_min: int = 0,
        year_max: int = 9999,
        limit: int = 2000,
    ) -> str:
        """Search patients in the loaded dataset by clinical criteria.

        Use this tool when the user asks to find, list, or count patients
        matching demographic or clinical criteria — e.g. "patients aged 55-75",
        "female patients with oropharynx cancer", "HPV-positive non-smokers".

        IMPORTANT: When searching for patients similar to a known patient, you MUST
        supply all four of these filters simultaneously:
          - primary_tumor_site (exact site of the reference patient)
          - histologic_type (exact histology of the reference patient)
          - hpv_association_p16 (HPV/p16 status of the reference patient)
          - age_min / age_max (reference patient age ±5 years)
        Never call this tool with no filters or only one filter when you already
        know the patient's clinical profile — that returns the full unfiltered
        dataset which is not useful.

        All parameters are optional and act as AND filters. String parameters
        are case-insensitive substring matches (e.g. "oro" matches "Oropharynx").
        Leave a parameter at its default to skip that filter.

        Args:
            age_min: Minimum age at initial diagnosis (inclusive, default 0).
            age_max: Maximum age at initial diagnosis (inclusive, default 200).
            primary_tumor_site: Substring to match against primary_tumor_site.
            histologic_type: Substring to match against histologic_type.
            hpv_association_p16: Substring to match against hpv_association_p16.
            pt_stage: Substring to match against pT_stage.
            pn_stage: Substring to match against pN_stage.
            smoking_status: Substring to match against smoking_status.
            sex: Substring to match against sex column.
            survival_status: Substring to match against survival_status.
            recurrence: Substring to match against recurrence column.
            year_min: Minimum year_of_initial_diagnosis (inclusive, default 0).
            year_max: Maximum year_of_initial_diagnosis (inclusive, default 9999).
            limit: Maximum number of patient rows to return (default 2000 — effectively all).

        Returns:
            JSON with matched patient count and a list of patient records
            (patient_id, age, year, primary_tumor_site, histologic_type,
            hpv_association_p16, pt_stage, pn_stage, smoking_status, sex,
            survival_status, recurrence).
        """
        import re as _re

        meta = data_service.get_metadata()
        if meta is None:
            return json.dumps({"error": "Dataset not loaded."})

        df = meta.copy()

        # Age range filter
        if "age_at_initial_diagnosis" in df.columns:
            age_col = df["age_at_initial_diagnosis"]
            age_numeric = age_col.apply(
                lambda v: int(v) if str(v).lstrip("-").isdigit() else None
            )
            mask = age_numeric.apply(
                lambda v: v is not None and age_min <= v <= age_max
            )
            df = df[mask]

        # Year range filter
        if "year_of_initial_diagnosis" in df.columns and (year_min > 0 or year_max < 9999):
            year_col = df["year_of_initial_diagnosis"]
            year_numeric = year_col.apply(
                lambda v: int(v) if str(v).lstrip("-").isdigit() else None
            )
            mask = year_numeric.apply(
                lambda v: v is not None and year_min <= v <= year_max
            )
            df = df[mask]

        # Categorical substring filters
        _str_filters = {
            "primary_tumor_site": primary_tumor_site,
            "histologic_type": histologic_type,
            "hpv_association_p16": hpv_association_p16,
            "pT_stage": pt_stage,
            "pN_stage": pn_stage,
            "smoking_status": smoking_status,
            "sex": sex,
            "survival_status": survival_status,
            "recurrence": recurrence,
        }
        for col, val in _str_filters.items():
            if val and col in df.columns:
                df = df[df[col].astype(str).str.contains(val, case=False, na=False)]

        # Deduplicate by patient ID
        def _pid(slide_name: str) -> str:
            m = _re.search(r"patient(\d+)", str(slide_name))
            return m.group(1) if m else slide_name

        df = df.copy()
        df["_patient_id"] = df["slide_name"].apply(_pid)
        df = df.drop_duplicates(subset="_patient_id")

        total = len(df)
        rows = []
        for _, row in df.iterrows():
            rows.append({
                "patient_id": row["_patient_id"],
                "age_at_initial_diagnosis": row.get("age_at_initial_diagnosis", "N/A"),
                "year_of_initial_diagnosis": row.get("year_of_initial_diagnosis", "N/A"),
                "primary_tumor_site": row.get("primary_tumor_site", "N/A"),
                "histologic_type": row.get("histologic_type", "N/A"),
                "hpv_association_p16": row.get("hpv_association_p16", "N/A"),
                "pt_stage": row.get("pT_stage", "N/A"),
                "pn_stage": row.get("pN_stage", "N/A"),
                "smoking_status": row.get("smoking_status", "N/A"),
                "sex": row.get("sex", "N/A"),
                "survival_status": row.get("survival_status", "N/A"),
                "recurrence": row.get("recurrence", "N/A"),
            })

        # Only emit a table artifact when the call is genuinely filtered.
        # Count how many meaningful filters were applied.
        _str_filter_values = [
            primary_tumor_site, histologic_type, hpv_association_p16,
            pt_stage, pn_stage, smoking_status, sex, survival_status, recurrence,
        ]
        _active_str_filters = sum(1 for v in _str_filter_values if v)
        _active_age = 1 if (age_min > 0 or age_max < 200) else 0
        _active_year = 1 if (year_min > 0 or year_max < 9999) else 0
        _n_filters = _active_str_filters + _active_age + _active_year

        # Build human-readable query summary for the frontend widget header
        _query_parts = []
        if primary_tumor_site: _query_parts.append(f"site={primary_tumor_site}")
        if histologic_type: _query_parts.append(f"histology={histologic_type}")
        if hpv_association_p16: _query_parts.append(f"HPV={hpv_association_p16}")
        if pt_stage: _query_parts.append(f"pT={pt_stage}")
        if pn_stage: _query_parts.append(f"pN={pn_stage}")
        if smoking_status: _query_parts.append(f"smoking={smoking_status}")
        if sex: _query_parts.append(f"sex={sex}")
        if survival_status: _query_parts.append(f"survival={survival_status}")
        if recurrence: _query_parts.append(f"recurrence={recurrence}")
        if age_min > 0 or age_max < 200:
            _query_parts.append(f"age={age_min}–{age_max}")
        if year_min > 0 or year_max < 9999:
            _query_parts.append(f"year={year_min}–{year_max}")
        _query_summary = ", ".join(_query_parts) if _query_parts else "no filters"

        # Suppress artifact only for completely unfiltered calls (no filters at all)
        # and empty results — both are noise in the chat UI.
        if _n_filters >= 1 and total > 0:
            artifact_queue.put(
                f"data: {json.dumps({'artifact': {'type': 'search', 'total': total, 'patients': rows, 'query': _query_summary}})}\n\n"
            )

        return json.dumps({"total_matched": total, "patients": rows})

    @tool
    def get_dataset_info() -> str:
        """Get information about the currently loaded dataset: slide count,
        patient count, and available filterable column names.

        Returns:
            JSON string with dataset statistics and column names.
        """
        status = data_service.get_status()
        if not status.loaded:
            return json.dumps({"error": "Dataset not loaded."})
        summary = data_service.get_summary()
        cols = [c.name for c in summary.filterable_columns] if summary else []
        return json.dumps({
            "n_slides": status.n_slides,
            "n_patients": status.n_patients,
            "filterable_columns": cols,
            "active_filters": app_context.active_filters,
        })

    @tool
    def run_retrieval(patient_id: str, k: int = 10, alpha: float = 0.4,
                      beta: float = 0.4, gamma: float = 0.2) -> str:
        """Find the most similar patients to a given query patient using composite
        similarity scoring (slide morphology + tissue composition + clinical metadata).

        Args:
            patient_id: The numeric patient ID to query (e.g. "15", "015", "492").
                        Leading zeros are handled automatically — "15", "015", "0015"
                        all refer to the same patient.
            k: Number of similar patients to return (default 10, max 50).
            alpha: Weight for slide morphology similarity (0-1).
            beta: Weight for tissue composition similarity (0-1).
            gamma: Weight for clinical metadata similarity (0-1).
                   alpha + beta + gamma must equal 1.0.

        Returns:
            JSON string with ranked patient matches including composite scores
            and clinical metadata (tumor site, histologic type, HPV status, staging).
            The UI will render this as a table automatically — do NOT reproduce the
            data as markdown in your text response. Provide only a brief interpretation.
        """
        # Resolve the canonical patient ID by matching on numeric value
        # (handles "15" == "015" == "0015", etc.)
        try:
            all_patients = retrieval_service.list_patients(data_service)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        canonical_id = _resolve_patient_id(patient_id, all_patients)
        if canonical_id is None:
            return json.dumps({
                "error": f"Patient '{patient_id}' not found. "
                         f"Available IDs (sample): {all_patients[:10]}"
            })
        total = alpha + beta + gamma
        if total <= 0:
            alpha, beta, gamma = 0.4, 0.4, 0.2
        else:
            alpha = round(alpha / total, 6)
            beta = round(beta / total, 6)
            gamma = round(1.0 - alpha - beta, 6)

        try:
            params = RetrievalParams(
                patient_id=canonical_id,
                k=min(int(k), 50),
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                filters=app_context.active_filters,
                deduplicate_by_patient=True,
                slide_aggregation="max",
            )
            result = retrieval_service.query(params, data_service, composition_service)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        matches = [
            {
                "rank": m.rank,
                "patientId": m.patient_id,
                "composite": round(m.composite, 4),
                "slideSim": round(m.slide_sim, 4),
                "compSim": round(m.comp_sim, 4),
                "metaSim": round(m.meta_sim, 4),
                "primaryTumorSite": m.primary_tumor_site,
                "histologicType": m.histologic_type,
                "hpvAssociationP16": m.hpv_association_p16,
                "ptStage": m.pt_stage,
                "pnStage": m.pn_stage,
                "smokingStatus": m.smoking_status,
                "ageAtInitialDiagnosis": m.age_at_initial_diagnosis,
            }
            for m in result.matches
        ]
        q_patient = result.query_patient
        query_row = {
            "rank": 0,
            "patientId": q_patient.patient_id,
            "composite": 1.0,
            "slideSim": round(q_patient.slide_sim, 4),
            "compSim": round(q_patient.comp_sim, 4),
            "metaSim": round(q_patient.meta_sim, 4),
            "primaryTumorSite": q_patient.primary_tumor_site,
            "histologicType": q_patient.histologic_type,
            "hpvAssociationP16": q_patient.hpv_association_p16,
            "ptStage": q_patient.pt_stage,
            "pnStage": q_patient.pn_stage,
            "smokingStatus": q_patient.smoking_status,
            "ageAtInitialDiagnosis": q_patient.age_at_initial_diagnosis,
        }

        # Emit artifact so the frontend can render the table inline
        artifact_queue.put(f"data: {json.dumps({'artifact': {'type': 'retrieval', 'queryPatient': query_row, 'matches': matches}})}\n\n")

        return json.dumps({"query_patient": query_row, "matches": matches})

    @tool
    def run_umap(n_neighbors: int = 15, min_dist: float = 0.1,
                 color_by: str = "primary_tumor_site",
                 patient_id: str = "") -> str:
        """Run a UMAP dimensionality reduction projection on the patient embeddings.
        If a patient_id is provided, that patient's slide will be highlighted on the plot.

        Args:
            n_neighbors: UMAP n_neighbors parameter (5-50, default 15).
            min_dist: UMAP min_dist parameter (0.01-0.5, default 0.1).
            color_by: Metadata column to colour points by (default primary_tumor_site).
            patient_id: Optional patient ID to highlight on the UMAP (e.g. "15", "015").
                        Leading zeros handled automatically.

        Returns:
            JSON string with n_points, n_clusters, and the Plotly figure JSON.
            The UI renders the chart automatically — do NOT describe it as text or markdown.
            Provide only a brief interpretation of the cluster structure.
        """
        # Resolve query slide and nearest-neighbour slides for highlighting
        query_slide = None
        neighbor_slides = []
        if patient_id:
            try:
                all_patients = retrieval_service.list_patients(data_service)
                canonical = _resolve_patient_id(patient_id, all_patients)
                if canonical:
                    embeddings = data_service.get_embeddings() or {}
                    slides = [s for s in embeddings if f"patient{canonical}" in s]
                    if slides:
                        query_slide = slides[0]

                    # Run retrieval to get nearest neighbours for highlighting
                    try:
                        from backend.services.composition_service import composition_service
                        ret_params = RetrievalParams(
                            patient_id=canonical,
                            k=10,
                            alpha=app_context.alpha,
                            beta=app_context.beta,
                            gamma=app_context.gamma,
                            filters=app_context.active_filters,
                            deduplicate_by_patient=True,
                            slide_aggregation="max",
                        )
                        ret_result = retrieval_service.query(ret_params, data_service, composition_service)
                        neighbor_slides = [
                            s for pid in [m.patient_id for m in ret_result.matches]
                            for s in [next((sl for sl in embeddings if f"patient{pid}" in sl), None)]
                            if s is not None
                        ]
                    except Exception:
                        pass  # neighbours are optional — UMAP still works without them
            except Exception:
                pass

        try:
            params = UMAPParams(
                n_neighbors=max(5, min(50, int(n_neighbors))),
                min_dist=max(0.01, min(0.5, float(min_dist))),
                color_by=color_by,
                filters=app_context.active_filters,
                query_slide=query_slide,
                neighbor_slides=neighbor_slides,
            )
            result = umap_service.run_umap(params, data_service)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        # Emit artifact with the full Plotly JSON for inline rendering
        artifact_queue.put(f"data: {json.dumps({'artifact': {'type': 'plotly', 'plotlyJson': result.plotly_json, 'title': f'UMAP — {result.n_points} points, {result.n_clusters} clusters'}})}\n\n")

        return json.dumps({
            "n_points": result.n_points,
            "n_clusters": result.n_clusters,
            "color_by": color_by,
        })

    @tool
    def run_clustering(k_min: int = 2, k_max: int = 10) -> str:
        """Run K-sweep clustering to find the optimal number of patient clusters.

        Args:
            k_min: Minimum number of clusters to evaluate (default 2).
            k_max: Maximum number of clusters to evaluate (default 10).

        Returns:
            JSON string with best_k, best_silhouette, and the Plotly figure JSON.
            The UI renders the chart automatically — do NOT reproduce results as markdown.
            Provide only a brief interpretation of the optimal cluster count.
        """
        try:
            params = ClusterParams(
                k_min=max(2, int(k_min)),
                k_max=min(20, int(k_max)),
                filters=app_context.active_filters,
            )
            result = cluster_service.run_clustering(params, data_service)
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        artifact_queue.put(f"data: {json.dumps({'artifact': {'type': 'plotly', 'plotlyJson': result.plotly_json, 'title': f'Clustering — best k={result.best_k}, silhouette={result.best_silhouette:.3f}'}})}\n\n")

        return json.dumps({
            "best_k": result.best_k,
            "best_silhouette": round(result.best_silhouette, 4),
        })

    @tool
    def recompute_slide_embeddings(patient_ids: str) -> str:
        """Download H5 patch files from S3 for a list of patients and recompute
        their slide-level embeddings using the trained GatedAttentionABMIL encoder.
        Also refreshes their tissue composition profiles against the existing codebook.

        Call this AFTER run_retrieval to enrich the top matched patients with
        ABMIL-derived embeddings before running run_umap or run_clustering.
        This ensures the visualisation and clustering use the highest-quality
        morphological representations rather than the pre-computed CSV vectors.

        Args:
            patient_ids: Comma-separated numeric patient ID strings,
                         e.g. "583,306,712,564,121,236,744,745,298,039,682,015,492,411,222".
                         Include the query patient and all top-k retrieval matches.

        Returns:
            JSON with 'downloaded' (H5 files fetched), 'embeddings_updated',
            'profiles_updated', and 'patient_ids' processed.
        """
        from backend.services.h5_service import h5_service

        pid_list = [p.strip() for p in patient_ids.split(",") if p.strip()]
        if not pid_list:
            return json.dumps({"error": "No patient IDs provided."})

        try:
            stats = h5_service.enrich_patients(
                patient_ids=pid_list,
                data_service=data_service,
                composition_service=composition_service,
            )
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps({
            "patient_ids": pid_list,
            "downloaded": stats["downloaded"],
            "embeddings_updated": stats["embeddings_updated"],
            "profiles_updated": stats["profiles_updated"],
            "message": (
                f"Recomputed ABMIL embeddings for {stats['embeddings_updated']} slides "
                f"across {len(pid_list)} patients. "
                "Subsequent run_umap and run_clustering calls will use these updated vectors."
            ),
        })

    @tool
    def validate_composition_profiles(color_by: str = "histologic_type") -> str:
        """Validate composition profiles by projecting patient-level tissue composition
        profiles to 2D via UMAP and rendering an interactive scatter plot.

        This mirrors the 'composition-umap' cell in the research notebook:
        each patient is represented by their morphological cluster proportion
        vector (not their slide embedding), so the plot reveals whether
        composition profiles capture clinically meaningful structure independent
        of the ABMIL slide embeddings.

        Compare this plot with the slide-embedding UMAP (run_umap): any
        additional grouping visible here reflects tissue composition patterns
        not captured by the aggregated embeddings. If the composition UMAP
        shows no structure, the β=0.4 composition weight in composite scoring
        is adding noise rather than signal.

        Args:
            color_by: Metadata column to colour points by.
                      Good choices: 'histologic_type', 'primary_tumor_site',
                      'hpv_association_p16', 'pT_stage', 'smoking_status'.
                      Default: 'histologic_type'.

        Returns:
            JSON with n_patients and a brief interpretation hint.
            The UI renders the UMAP chart automatically.
        """
        profiles = composition_service.get_profiles()
        if profiles is None:
            return json.dumps({"error": "Composition profiles not built. Build profiles first."})

        metadata_df = data_service.get_metadata()
        if metadata_df is None:
            return json.dumps({"error": "Dataset not loaded."})

        try:
            import re as _re
            from umap_retrieval.embedding import assign_clusters_by_metadata, run_umap as _run_umap
            from umap_retrieval.visualization import plot_umap_interactive

            # Build patient-id → representative slide_name map (same as notebook)
            pid_to_slide: dict = {}
            for sn in metadata_df["slide_name"]:
                m = _re.search(r"patient(\d+)", str(sn))
                if m:
                    pid = m.group(1)
                    if pid not in pid_to_slide:
                        pid_to_slide[pid] = sn

            # Build comp_embeddings: slide_name → profile vector (float64 for UMAP)
            comp_embeddings = {
                pid_to_slide[pid]: profiles[pid].astype(float)
                for pid in profiles
                if pid in pid_to_slide
            }

            if len(comp_embeddings) < 5:
                return json.dumps({"error": "Too few patients with profiles to run UMAP."})

            # One-row-per-patient metadata for coloring
            meta_patients = metadata_df.copy()
            meta_patients["_pid"] = meta_patients["slide_name"].apply(
                lambda s: _re.search(r"patient(\d+)", str(s)).group(1)
                if _re.search(r"patient(\d+)", str(s)) else None
            )
            meta_patients = meta_patients.dropna(subset=["_pid"])
            meta_patients = meta_patients.drop_duplicates(subset=["_pid"]).copy()
            # Replace slide_name with the representative slide used in comp_embeddings
            meta_patients["slide_name"] = meta_patients["_pid"].map(pid_to_slide)
            meta_patients = meta_patients.dropna(subset=["slide_name"])

            umap_df = _run_umap(comp_embeddings, n_neighbors=15, min_dist=0.1)
            umap_df = assign_clusters_by_metadata(umap_df, meta_patients, color_by)

            fig = plot_umap_interactive(
                umap_df,
                meta_patients,
                color_by=color_by,
            )
            plotly_json = fig.to_json()
            n_clusters = int(umap_df["cluster"].nunique())

        except Exception as exc:
            return json.dumps({"error": str(exc)})

        _title = (
            f"Composition Profile UMAP — full cohort ({len(comp_embeddings)} patients), "
            f"colored by {color_by}. "
            "Each point = one patient's tissue composition profile "
            "(patch-level morphological cluster proportions, not slide embeddings)."
        )
        artifact_queue.put(
            f"data: {json.dumps({'artifact': {'type': 'plotly', 'plotlyJson': plotly_json, 'title': _title}})}\n\n"
        )

        return json.dumps({
            "n_patients": len(comp_embeddings),
            "n_clusters": n_clusters,
            "color_by": color_by,
            "interpretation": (
                "If composition profiles are working well, patients with similar "
                "morphological cluster distributions should group together. "
                "Compare with the slide-embedding UMAP — additional grouping here "
                "reflects tissue composition patterns not captured by ABMIL aggregation."
            ),
        })

    @tool
    def plot_cohort_analysis(  # noqa: PLR0912
        chart_type: str = "distribution",
        x_column: str = "primary_tumor_site",
        y_column: str = "",
        color_by: str = "",
        patient_ids: str = "",
        title: str = "",
    ) -> str:
        """Generate an exploratory data analysis chart from the patient cohort metadata.

        Use this tool when the user asks to visualise, explore, or analyse the
        distribution of clinical variables, compare groups, or generate charts
        for a report. Supports bar charts, pie charts, box plots, scatter plots,
        heatmaps, and survival-style count plots.

        Args:
            chart_type: One of:
                'distribution'  — bar chart of value counts for x_column
                'pie'           — pie chart of value counts for x_column
                'crosstab'      — grouped bar chart of x_column vs y_column
                'boxplot'       — box plot of a numeric column (x_column) grouped by color_by
                'scatter'       — scatter plot of x_column vs y_column, colored by color_by
                'heatmap'       — correlation heatmap of x_column vs y_column counts
                'age_histogram' — histogram of age at initial diagnosis
            x_column: Primary column for the chart (default: primary_tumor_site).
                Valid columns: primary_tumor_site, histologic_type, hpv_association_p16,
                pT_stage, pN_stage, smoking_status, sex, survival_status, recurrence,
                age_at_initial_diagnosis, year_of_initial_diagnosis.
            y_column: Secondary column for crosstab, scatter, or heatmap charts.
            color_by: Column to use for colour grouping in scatter/boxplot.
            patient_ids: Optional comma-separated patient IDs to restrict the chart
                to a specific subset (e.g. retrieval matches). Leave empty for full cohort.
            title: Optional custom chart title.

        Returns:
            JSON with chart metadata. The UI renders the chart automatically.
        """
        import re as _re
        import plotly.express as px
        import plotly.graph_objects as go

        # ── AWS Cloudscape colour palette ─────────────────────────────
        _AWS_COLORS = [
            "#0972d3",  # AWS blue
            "#037f0c",  # green
            "#8c4fff",  # purple
            "#e07941",  # orange
            "#c7162b",  # red
            "#00a4a6",  # teal
            "#f89256",  # light orange
            "#6b40c4",  # deep purple
            "#1a9c3e",  # mid green
            "#d91515",  # bright red
        ]
        _FONT = dict(family="'Amazon Ember', 'Helvetica Neue', Arial, sans-serif", size=12, color="#16191f")
        _AXIS = dict(
            showgrid=True, gridcolor="#e9ebed", gridwidth=1,
            linecolor="#d1d5db", linewidth=1,
            tickfont=dict(size=11, color="#414d5c"),
            title_font=dict(size=12, color="#414d5c"),
            zeroline=False,
        )
        _BASE_LAYOUT = dict(
            height=400,
            margin=dict(t=56, r=24, b=72, l=64),
            plot_bgcolor="#ffffff",
            paper_bgcolor="#ffffff",
            font=_FONT,
            title_font=dict(size=14, color="#16191f", family="'Amazon Ember', 'Helvetica Neue', Arial, sans-serif"),
            title_x=0,
            title_pad=dict(l=4, t=4),
            legend=dict(
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#d1d5db",
                borderwidth=1,
                font=dict(size=11),
            ),
            colorway=_AWS_COLORS,
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor="#d1d5db",
                font=dict(size=12, color="#16191f"),
            ),
        )

        meta = data_service.get_metadata()
        if meta is None:
            return json.dumps({"error": "Dataset not loaded."})

        df = meta.copy()

        def _pid(s: str) -> str:
            m = _re.search(r"patient(\d+)", str(s))
            return m.group(1) if m else s

        df["_pid"] = df["slide_name"].apply(_pid)
        df = df.drop_duplicates(subset="_pid")

        if patient_ids.strip():
            pid_set = {p.strip() for p in patient_ids.split(",") if p.strip()}
            df = df[df["_pid"].isin(pid_set)]
            if df.empty:
                return json.dumps({"error": "No patients found for the given IDs."})

        n = len(df)

        def _label(col: str) -> str:
            return col.replace("_", " ").replace("association", "").replace("  ", " ").strip().title()

        chart_title = title or f"{_label(x_column)} — {n} patients"

        try:
            if chart_type == "distribution":
                counts = df[x_column].value_counts().reset_index()
                counts.columns = [x_column, "count"]
                fig = px.bar(
                    counts, x=x_column, y="count",
                    title=chart_title, text="count",
                    color=x_column, color_discrete_sequence=_AWS_COLORS,
                    labels={x_column: _label(x_column), "count": "Patients"},
                )
                fig.update_traces(
                    textposition="outside",
                    textfont=dict(size=11, color="#414d5c"),
                    marker_line_width=0,
                )
                fig.update_layout(
                    **_BASE_LAYOUT,
                    showlegend=False,
                    xaxis=dict(**_AXIS, tickangle=-30, title=_label(x_column)),
                    yaxis=dict(**_AXIS, title="Patients"),
                    bargap=0.35,
                )

            elif chart_type == "pie":
                counts = df[x_column].value_counts().reset_index()
                counts.columns = [x_column, "count"]
                fig = px.pie(
                    counts, names=x_column, values="count",
                    title=chart_title,
                    color_discrete_sequence=_AWS_COLORS,
                    hole=0.35,
                )
                fig.update_traces(
                    textposition="outside",
                    textinfo="percent+label",
                    textfont=dict(size=11),
                    marker=dict(line=dict(color="#ffffff", width=2)),
                    pull=[0.03] * len(counts),
                )
                fig.update_layout(**_BASE_LAYOUT)

            elif chart_type == "crosstab":
                if not y_column:
                    return json.dumps({"error": "y_column is required for crosstab charts."})
                ct = df.groupby([x_column, y_column]).size().reset_index(name="count")
                fig = px.bar(
                    ct, x=x_column, y="count", color=y_column, barmode="group",
                    title=chart_title,
                    color_discrete_sequence=_AWS_COLORS,
                    labels={"count": "Patients", x_column: _label(x_column), y_column: _label(y_column)},
                    text="count",
                )
                fig.update_traces(textposition="outside", textfont=dict(size=10), marker_line_width=0)
                fig.update_layout(
                    **_BASE_LAYOUT,
                    xaxis=dict(**_AXIS, tickangle=-30, title=_label(x_column)),
                    yaxis=dict(**_AXIS, title="Patients"),
                    bargap=0.2, bargroupgap=0.05,
                )

            elif chart_type == "heatmap":
                if not y_column:
                    return json.dumps({"error": "y_column is required for heatmap charts."})
                ct = df.groupby([x_column, y_column]).size().unstack(fill_value=0)
                fig = go.Figure(data=go.Heatmap(
                    z=ct.values,
                    x=[_label(str(c)) for c in ct.columns],
                    y=[_label(str(r)) for r in ct.index],
                    colorscale=[[0, "#f0f4ff"], [0.5, "#4a90d9"], [1, "#0972d3"]],
                    text=ct.values,
                    texttemplate="%{text}",
                    textfont=dict(size=11),
                    hoverongaps=False,
                    showscale=True,
                    colorbar=dict(thickness=12, tickfont=dict(size=10)),
                ))
                fig.update_layout(
                    **_BASE_LAYOUT,
                    xaxis=dict(**_AXIS, title=_label(y_column), tickangle=-30),
                    yaxis=dict(**_AXIS, title=_label(x_column)),
                )

            elif chart_type == "boxplot":
                num_col = x_column
                if num_col not in df.columns:
                    return json.dumps({"error": f"Column '{num_col}' not found."})
                df[num_col] = df[num_col].apply(
                    lambda v: float(v) if str(v).lstrip("-").replace(".", "").isdigit() else None
                )
                df = df.dropna(subset=[num_col])
                grp = color_by if color_by and color_by in df.columns else None
                fig = px.box(
                    df, x=grp, y=num_col, color=grp,
                    title=chart_title,
                    color_discrete_sequence=_AWS_COLORS,
                    labels={num_col: _label(num_col), grp: _label(grp) if grp else ""},
                    points="outliers",
                )
                fig.update_traces(
                    marker=dict(size=4, opacity=0.6),
                    line=dict(width=1.5),
                )
                fig.update_layout(
                    **_BASE_LAYOUT,
                    xaxis=dict(**_AXIS, tickangle=-20),
                    yaxis=dict(**_AXIS, title=_label(num_col)),
                )

            elif chart_type == "scatter":
                if not y_column:
                    return json.dumps({"error": "y_column is required for scatter charts."})
                for col in [x_column, y_column]:
                    df[col] = df[col].apply(
                        lambda v: float(v) if str(v).lstrip("-").replace(".", "").isdigit() else None
                    )
                df = df.dropna(subset=[x_column, y_column])
                clr = color_by if color_by and color_by in df.columns else None
                fig = px.scatter(
                    df, x=x_column, y=y_column, color=clr,
                    hover_data=["_pid"],
                    title=chart_title,
                    color_discrete_sequence=_AWS_COLORS,
                    labels={x_column: _label(x_column), y_column: _label(y_column)},
                    opacity=0.75,
                )
                fig.update_traces(marker=dict(size=7, line=dict(width=0.5, color="#ffffff")))
                fig.update_layout(
                    **_BASE_LAYOUT,
                    xaxis=dict(**_AXIS, title=_label(x_column)),
                    yaxis=dict(**_AXIS, title=_label(y_column)),
                )

            elif chart_type == "age_histogram":
                df["age_at_initial_diagnosis"] = df["age_at_initial_diagnosis"].apply(
                    lambda v: float(v) if str(v).lstrip("-").replace(".", "").isdigit() else None
                )
                df = df.dropna(subset=["age_at_initial_diagnosis"])
                clr = color_by if color_by and color_by in df.columns else None
                fig = px.histogram(
                    df, x="age_at_initial_diagnosis", color=clr,
                    nbins=20,
                    title=chart_title or f"Age at Diagnosis — {n} patients",
                    color_discrete_sequence=_AWS_COLORS,
                    labels={"age_at_initial_diagnosis": "Age at Diagnosis", "count": "Patients"},
                    barmode="overlay", opacity=0.8,
                )
                fig.update_traces(marker_line_width=0)
                fig.update_layout(
                    **_BASE_LAYOUT,
                    bargap=0.05,
                    xaxis=dict(**_AXIS, title="Age at Diagnosis"),
                    yaxis=dict(**_AXIS, title="Patients"),
                )

            else:
                return json.dumps({"error": f"Unknown chart_type '{chart_type}'. "
                                   "Valid: distribution, pie, crosstab, boxplot, scatter, heatmap, age_histogram."})

        except Exception as exc:
            return json.dumps({"error": str(exc)})

        artifact_queue.put(
            f"data: {json.dumps({'artifact': {'type': 'plotly', 'plotlyJson': fig.to_json(), 'title': chart_title}})}\n\n"
        )

        return json.dumps({"chart_type": chart_type, "n_patients": n, "title": chart_title})

    # ------------------------------------------------------------------
    # Biomarker discovery tools
    # ------------------------------------------------------------------

    @tool
    def classify_cohorts(
        deceased: bool = False,
        tumor_caused_death: bool = False,
        recurrence: bool = False,
        progression: bool = False,
        metastasis: bool = False,
    ) -> str:
        """Classify patients into Non_Responder / Responder cohorts based on
        clinical outcome criteria.

        Use this tool when the user asks about cohort classification, non-responders,
        or patient outcomes — e.g. "show me non-responders who had recurrence or
        metastasis", "how many patients died from their tumor?".

        At least one criterion must be True. A patient is Non_Responder if at least
        one selected criterion matches.

        Args:
            deceased: Include patients with survival_status = "deceased".
            tumor_caused_death: Include patients with survival_status_with_cause = "deceased_due_to_tumor".
            recurrence: Include patients with recurrence = "yes".
            progression: Include patients with progress_1 or progress_2 = "yes".
            metastasis: Include patients with metastasis_1_locations not null.

        Returns:
            JSON with Non_Responder count, Responder count, excluded count,
            mean ages, and sex distributions for each cohort.
        """
        from backend.models import ClassifyRequest, OutcomeCriteria
        from backend.services.outcome_service import outcome_service

        criteria = OutcomeCriteria(
            deceased=deceased,
            tumor_caused_death=tumor_caused_death,
            recurrence=recurrence,
            progression=progression,
            metastasis=metastasis,
        )
        try:
            result = outcome_service.classify(ClassifyRequest(criteria=criteria))
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        return json.dumps(result.model_dump(by_alias=True))

    @tool
    def query_biomarkers(
        deceased: bool = False,
        tumor_caused_death: bool = False,
        recurrence: bool = False,
        progression: bool = False,
        metastasis: bool = False,
    ) -> str:
        """Query biomarker analysis results between Non_Responder and Responder cohorts.

        Use this tool when the user asks about biomarker differences, significant
        analytes, or blood test comparisons — e.g. "which biomarkers are significantly
        different?", "compare blood values between responders and non-responders".

        At least one criterion must be True.

        Args:
            deceased: Include patients with survival_status = "deceased".
            tumor_caused_death: Include patients with survival_status_with_cause = "deceased_due_to_tumor".
            recurrence: Include patients with recurrence = "yes".
            progression: Include patients with progress_1 or progress_2 = "yes".
            metastasis: Include patients with metastasis_1_locations not null.

        Returns:
            JSON with significant analyte count, total analytes tested, and a list
            of significant analytes sorted by adjusted p-value with effect sizes.
        """
        from backend.models import BiomarkerRequest, OutcomeCriteria
        from backend.services.outcome_service import outcome_service

        criteria = OutcomeCriteria(
            deceased=deceased,
            tumor_caused_death=tumor_caused_death,
            recurrence=recurrence,
            progression=progression,
            metastasis=metastasis,
        )
        try:
            result = outcome_service.analyze_biomarkers(BiomarkerRequest(criteria=criteria))
        except Exception as exc:
            return json.dumps({"error": str(exc)})

        significant = [c for c in result.comparisons if c.significant]
        summary = [
            c.model_dump(by_alias=True)
            for c in sorted(significant, key=lambda x: x.adjusted_p_value)
        ]
        return json.dumps({
            "significant_count": len(significant),
            "total_analytes": len(result.comparisons),
            "significant_analytes": summary,
        })

    @tool
    def generate_biomarker_chart(
        chart_type: str = "boxplot",
        analyte_name: str = "",
        deceased: bool = False,
        tumor_caused_death: bool = False,
        recurrence: bool = False,
        progression: bool = False,
        metastasis: bool = False,
    ) -> str:
        """Generate a biomarker visualization and emit it as an inline chart artifact.

        Use this tool when the user asks for a biomarker chart, box plot, heatmap,
        or deviation plot — e.g. "show me a box plot of Hemoglobin", "generate a
        heatmap of deviation scores".

        At least one criterion must be True.

        Args:
            chart_type: One of "boxplot", "heatmap", or "deviation".
                - boxplot: Box plot of a single analyte comparing cohorts (requires analyte_name).
                - heatmap: Heatmap of deviation scores (rows=analytes, columns=patients grouped by cohort).
                - deviation: Same as heatmap (alias).
            analyte_name: Required for boxplot chart_type. The analyte to visualize.
            deceased: Include patients with survival_status = "deceased".
            tumor_caused_death: Include patients with survival_status_with_cause = "deceased_due_to_tumor".
            recurrence: Include patients with recurrence = "yes".
            progression: Include patients with progress_1 or progress_2 = "yes".
            metastasis: Include patients with metastasis_1_locations not null.

        Returns:
            JSON with chart_type and analyte name. The Plotly figure is emitted as
            an inline artifact — do NOT reproduce the chart as text or markdown.
        """
        from backend.models import BiomarkerRequest, BoxPlotRequest, OutcomeCriteria
        from backend.services.outcome_service import outcome_service

        criteria = OutcomeCriteria(
            deceased=deceased,
            tumor_caused_death=tumor_caused_death,
            recurrence=recurrence,
            progression=progression,
            metastasis=metastasis,
        )

        try:
            if chart_type == "boxplot":
                if not analyte_name:
                    return json.dumps({"error": "analyte_name is required for boxplot chart_type."})
                result = outcome_service.generate_box_plot(
                    BoxPlotRequest(criteria=criteria, analyte_name=analyte_name)
                )
                title = f"{analyte_name} — Non_Responder vs Responder"
                artifact_queue.put(
                    f"data: {json.dumps({'artifact': {'type': 'plotly', 'plotlyJson': result.plotly_json, 'title': title}})}\n\n"
                )
                return json.dumps({
                    "chart_type": "boxplot",
                    "analyte": analyte_name,
                    "has_reference_range": result.has_reference_range,
                })

            elif chart_type in ("heatmap", "deviation"):
                import pandas as pd
                import plotly.graph_objects as go
                import plotly.io as pio

                bio_result = outcome_service.analyze_biomarkers(
                    BiomarkerRequest(criteria=criteria)
                )
                cells = bio_result.deviation_scores
                if not cells:
                    return json.dumps({"error": "No deviation scores available."})

                # Build a DataFrame from deviation cells
                rows = [
                    {"patient_id": c.patient_id, "analyte_name": c.analyte_name,
                     "deviation_score": c.deviation_score, "cohort": c.cohort}
                    for c in cells if c.deviation_score is not None
                ]
                if not rows:
                    return json.dumps({"error": "No valid deviation scores (all missing reference ranges)."})

                df = pd.DataFrame(rows)
                pivot = df.pivot_table(
                    index="analyte_name", columns="patient_id",
                    values="deviation_score", aggfunc="mean",
                )

                # Sort columns: non_responder patients first, then responder
                cohort_map = {c.patient_id: c.cohort for c in cells}
                nr_cols = sorted([c for c in pivot.columns if cohort_map.get(c) == "non_responder"])
                r_cols = sorted([c for c in pivot.columns if cohort_map.get(c) == "responder"])
                ordered_cols = nr_cols + r_cols
                pivot = pivot.reindex(columns=ordered_cols)

                fig = go.Figure(data=go.Heatmap(
                    z=pivot.values,
                    x=[f"P{c}" for c in pivot.columns],
                    y=list(pivot.index),
                    colorscale="RdBu_r",
                    zmid=0,
                    text=pivot.values.round(2),
                    texttemplate="%{text}",
                    textfont=dict(size=8),
                    hoverongaps=False,
                    colorbar=dict(title="Deviation", thickness=12),
                ))
                # Add a vertical line separating cohorts
                fig.add_vline(
                    x=len(nr_cols) - 0.5,
                    line_dash="dash", line_color="black", line_width=1,
                )
                fig.update_layout(
                    title="Deviation Score Heatmap — Non_Responder | Responder",
                    xaxis_title="Patients",
                    yaxis_title="Analytes",
                    height=max(400, len(pivot.index) * 25),
                    template="plotly_white",
                )
                plotly_json = pio.to_json(fig)
                title = "Deviation Score Heatmap"
                artifact_queue.put(
                    f"data: {json.dumps({'artifact': {'type': 'plotly', 'plotlyJson': plotly_json, 'title': title}})}\n\n"
                )
                return json.dumps({
                    "chart_type": chart_type,
                    "n_analytes": len(pivot.index),
                    "n_patients": len(pivot.columns),
                })

            else:
                return json.dumps({
                    "error": f"Unknown chart_type '{chart_type}'. Valid: boxplot, heatmap, deviation."
                })

        except Exception as exc:
            return json.dumps({"error": str(exc)})

    return [
        search_patients, get_dataset_info, run_retrieval,
        recompute_slide_embeddings, run_umap, run_clustering,
        validate_composition_profiles, plot_cohort_analysis,
        classify_cohorts, query_biomarkers, generate_biomarker_chart,
    ]


class ChatService:
    def stream_response(
        self,
        messages: List[ChatMessage],
        app_context: AppContext,
    ) -> Generator[str, None, None]:
        """Run a Strands agent and yield SSE lines for text tokens, tool use, and artifacts."""
        model_id = os.environ.get("BEDROCK_MODEL_ID", DEFAULT_MODEL_ID)
        region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

        system_prompt = build_system_prompt(app_context)

        # Shared queue for artifact events emitted from tool closures
        artifact_q: queue.Queue = queue.Queue()
        tools = _make_tools(app_context, artifact_q)
        model = BedrockModel(model_id=model_id, region_name=region)

        # Main SSE queue — receives text tokens, tool_use notifications, and artifacts
        q: queue.Queue = queue.Queue()
        _SENTINEL = object()

        def callback_handler(**kwargs):
            # Drain any artifact events emitted by tools since last callback
            while not artifact_q.empty():
                try:
                    q.put(artifact_q.get_nowait())
                except queue.Empty:
                    break

            data = kwargs.get("data", "")
            tool_use = (
                kwargs.get("event", {})
                .get("contentBlockStart", {})
                .get("start", {})
                .get("toolUse")
            )
            if tool_use:
                q.put(f"data: {json.dumps({'tool_use': tool_use['name']})}\n\n")
            if data:
                q.put(f"data: {json.dumps({'delta': data})}\n\n")

        def run_agent():
            try:
                agent = Agent(
                    model=model,
                    tools=tools,
                    system_prompt=system_prompt,
                    callback_handler=callback_handler,
                )
                prior = [
                    {"role": msg.role, "content": [{"text": msg.content}]}
                    for msg in messages[:-1]
                    if msg.content.strip()  # skip empty-content turns (artifact-only responses)
                ]
                if prior:
                    agent.messages = prior

                last_message = messages[-1].content if messages else ""
                agent(last_message)
            except Exception as exc:
                q.put(f"data: {json.dumps({'error': str(exc)})}\n\n")
            finally:
                # Drain any remaining artifacts before sentinel
                while not artifact_q.empty():
                    try:
                        q.put(artifact_q.get_nowait())
                    except queue.Empty:
                        break
                q.put(_SENTINEL)

        thread = threading.Thread(target=run_agent, daemon=True)
        thread.start()

        while True:
            item = q.get()
            if item is _SENTINEL:
                yield f"data: {json.dumps({'done': True})}\n\n"
                break
            yield item
