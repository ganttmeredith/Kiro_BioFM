from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, model_validator
from pydantic.alias_generators import to_camel


class _CamelModel(BaseModel):
    """Base model that serialises to camelCase JSON for the frontend."""

    model_config = {"alias_generator": to_camel, "populate_by_name": True}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class LoadRequest(_CamelModel):
    csv_path: str


class ColumnMeta(_CamelModel):
    name: str
    label: str
    values: List[str]


class NullDistStats(_CamelModel):
    mean: float
    std: float
    percentile_5: float
    percentile_95: float


class SilhouetteRow(_CamelModel):
    column: str
    score: float


class DataSummary(_CamelModel):
    n_slides: int
    n_patients: int
    filterable_columns: List[ColumnMeta]
    null_distribution: NullDistStats
    silhouette_scores: List[SilhouetteRow]


# ---------------------------------------------------------------------------
# App status
# ---------------------------------------------------------------------------

class AppStatus(_CamelModel):
    loaded: bool
    n_slides: Optional[int] = None
    n_patients: Optional[int] = None
    profiles_ready: bool = False
    csv_path: Optional[str] = None


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

class UMAPParams(_CamelModel):
    n_neighbors: int = 15
    min_dist: float = 0.1
    color_by: str = "primary_tumor_site"
    filters: Dict[str, List[str]] = {}
    query_slide: Optional[str] = None
    neighbor_slides: List[str] = []


class UMAPResponse(_CamelModel):
    plotly_json: str
    n_points: int
    n_clusters: int


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class ClusterParams(_CamelModel):
    k_min: int = 2
    k_max: int = 10
    cross_tabulate_cols: List[str] = [
        "primary_tumor_site",
        "histologic_type",
        "hpv_association_p16",
    ]
    filters: Dict[str, List[str]] = {}


class ClusterResponse(_CamelModel):
    plotly_json: str
    best_k: int
    best_silhouette: float


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class RetrievalParams(_CamelModel):
    patient_id: str
    k: int = 10
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    filters: Dict[str, List[str]] = {}
    deduplicate_by_patient: bool = True
    slide_aggregation: str = "max"

    @model_validator(mode="after")
    def weights_must_sum_to_one(self) -> "RetrievalParams":
        total = round(self.alpha + self.beta + self.gamma, 10)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"alpha + beta + gamma must equal 1.0, got {total:.6f}"
            )
        return self


class PatientMatch(_CamelModel):
    rank: int
    patient_id: str
    composite: float
    slide_sim: float
    comp_sim: float
    meta_sim: float
    primary_tumor_site: Optional[str] = None
    histologic_type: Optional[str] = None
    hpv_association_p16: Optional[str] = None
    pt_stage: Optional[str] = None
    pn_stage: Optional[str] = None
    smoking_status: Optional[str] = None
    age_at_initial_diagnosis: Optional[int] = None
    year_of_initial_diagnosis: Optional[int] = None


class RetrievalResponse(_CamelModel):
    matches: List[PatientMatch]
    query_patient: PatientMatch
    query_slide_name: Optional[str] = None
    neighbor_slide_names: List[str] = []


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AppContext(BaseModel):
    n_slides: Optional[int] = None
    n_patients: Optional[int] = None
    active_filters: Dict[str, List[str]] = {}
    query_patient_id: Optional[str] = None
    retrieval_results: Optional[List[PatientMatch]] = None
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    umap_n_points: Optional[int] = None
    umap_n_clusters: Optional[int] = None
    best_k: Optional[int] = None


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    app_context: AppContext


# ---------------------------------------------------------------------------
# Outcome – cohort classification & biomarker discovery
# ---------------------------------------------------------------------------


class OutcomeCriteria(_CamelModel):
    deceased: bool = False
    tumor_caused_death: bool = False
    recurrence: bool = False
    progression: bool = False
    metastasis: bool = False


class ClassifyRequest(_CamelModel):
    criteria: OutcomeCriteria


class CohortSummary(_CamelModel):
    count: int
    mean_age: Optional[float] = None
    sex_distribution: Dict[str, int] = {}


class ClassifyResponse(_CamelModel):
    non_responder: CohortSummary
    responder: CohortSummary
    non_responder_ids: List[str]
    responder_ids: List[str]
    excluded_count: int


class BiomarkerRequest(_CamelModel):
    criteria: OutcomeCriteria


class AnalyteComparison(_CamelModel):
    analyte_name: str
    loinc_code: str
    group: str
    non_responder_mean: float
    non_responder_std: float
    responder_mean: float
    responder_std: float
    p_value: float
    adjusted_p_value: float
    effect_size: float
    significant: bool


class DeviationCell(_CamelModel):
    patient_id: str
    analyte_name: str
    deviation_score: Optional[float] = None
    cohort: str


class BiomarkerResponse(_CamelModel):
    comparisons: List[AnalyteComparison]
    deviation_scores: List[DeviationCell]
    box_plot_json: Optional[str] = None


class BoxPlotRequest(_CamelModel):
    criteria: OutcomeCriteria
    analyte_name: str


class BoxPlotResponse(_CamelModel):
    plotly_json: str
    has_reference_range: bool


class OutcomeUMAPRequest(_CamelModel):
    criteria: OutcomeCriteria
    modality: Literal["imaging", "clinical", "multimodal"] = "imaging"
    n_neighbors: int = 15
    min_dist: float = 0.1
    color_by: str = "cohort"


class OutcomeUMAPResponse(_CamelModel):
    plotly_json: str
    n_points: int
    silhouette_score: float


# ---------------------------------------------------------------------------
# Interpretation (Bedrock AI)
# ---------------------------------------------------------------------------


class InterpretRequest(_CamelModel):
    context_type: Literal["biomarker_stats", "umap_clusters"]
    context_data: Dict[str, Any]


class InterpretResponse(_CamelModel):
    interpretation: str
    context_type: str
