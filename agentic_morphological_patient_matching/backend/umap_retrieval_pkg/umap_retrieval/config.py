"""Constants and configuration for the UMAP retrieval pipeline."""

SEED = 42

FILTERABLE_COLUMNS = [
    "primary_tumor_site",
    "pT_stage",
    "pN_stage",
    "grading_hpv",
    "hpv_association_p16",
    "histologic_type",
    "year_of_initial_diagnosis",
    "survival_status",
    "recurrence",
    "smoking_status",
    "sex",
]

NONFILTERABLE_COLUMNS = [
    "perineural_invasion_Pn",
    "lymphovascular_invasion",
    "perinodal_invasion",
    "age_at_initial_diagnosis",
    "primarily_metastasis",
]

REQUIRED_METADATA_COLUMNS = FILTERABLE_COLUMNS + NONFILTERABLE_COLUMNS

METADATA_COLUMNS = ["slide_name"] + FILTERABLE_COLUMNS + NONFILTERABLE_COLUMNS

# Optional columns passed through when present in the CSV
OPTIONAL_METADATA_COLUMNS = ["h5_path", "h5file"]
