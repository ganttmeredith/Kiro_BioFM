"""Outcome router: cohort classification, biomarker analysis, UMAP, and export."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response

from backend.models import (
    BiomarkerRequest,
    BiomarkerResponse,
    BoxPlotRequest,
    BoxPlotResponse,
    ClassifyRequest,
    ClassifyResponse,
    InterpretRequest,
    InterpretResponse,
    OutcomeUMAPRequest,
    OutcomeUMAPResponse,
)
from backend.services.data_service import data_service
from backend.services.outcome_service import outcome_service

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_cohorts(params: ClassifyRequest):
    """POST /api/outcome/classify — partition patients into Non_Responder / Responder.

    - HTTP 422: no criteria selected
    - HTTP 500: data loading failure
    """
    try:
        return outcome_service.classify(params)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except RuntimeError as exc:
        msg = str(exc)
        if "not loaded" in msg:
            return JSONResponse({"detail": msg}, status_code=409)
        return JSONResponse({"detail": msg}, status_code=500)


@router.post("/biomarkers", response_model=BiomarkerResponse)
async def analyze_biomarkers(params: BiomarkerRequest):
    """POST /api/outcome/biomarkers — per-analyte statistical comparison.

    - HTTP 422: no criteria selected
    - HTTP 500: data loading failure
    """
    try:
        return outcome_service.analyze_biomarkers(params)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except RuntimeError as exc:
        msg = str(exc)
        if "not loaded" in msg:
            return JSONResponse({"detail": msg}, status_code=409)
        return JSONResponse({"detail": msg}, status_code=500)


@router.post("/boxplot", response_model=BoxPlotResponse)
async def generate_box_plot(params: BoxPlotRequest):
    """POST /api/outcome/boxplot — box plot for a selected analyte.

    - HTTP 422: no criteria selected
    - HTTP 500: data loading failure
    """
    try:
        return outcome_service.generate_box_plot(params)
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except RuntimeError as exc:
        msg = str(exc)
        if "not loaded" in msg:
            return JSONResponse({"detail": msg}, status_code=409)
        return JSONResponse({"detail": msg}, status_code=500)


@router.post("/umap", response_model=OutcomeUMAPResponse)
async def run_outcome_umap(params: OutcomeUMAPRequest):
    """POST /api/outcome/umap — UMAP projection colored by cohort or clinical variable.

    - HTTP 409: dataset not loaded (imaging/multimodal modality)
    - HTTP 422: no criteria selected
    - HTTP 400: fewer than 2 patients in cohort
    - HTTP 500: other runtime errors
    """
    try:
        return outcome_service.run_outcome_umap(params, data_service)
    except ValueError as exc:
        msg = str(exc)
        if "fewer than 2" in msg:
            return JSONResponse({"detail": msg}, status_code=400)
        return JSONResponse({"detail": msg}, status_code=422)
    except RuntimeError as exc:
        msg = str(exc)
        if "not loaded" in msg:
            return JSONResponse({"detail": msg}, status_code=409)
        return JSONResponse({"detail": msg}, status_code=500)


@router.post("/export")
async def export_biomarker_csv(params: BiomarkerRequest):
    """POST /api/outcome/export — download biomarker comparison as CSV.

    - HTTP 422: no criteria selected
    - HTTP 500: data loading failure
    """
    try:
        csv_content = outcome_service.export_csv(params)
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=biomarker_comparison.csv"},
        )
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except RuntimeError as exc:
        msg = str(exc)
        if "not loaded" in msg:
            return JSONResponse({"detail": msg}, status_code=409)
        return JSONResponse({"detail": msg}, status_code=500)


@router.post("/interpret", response_model=InterpretResponse)
async def interpret_results(params: InterpretRequest):
    """POST /api/outcome/interpret — AI-powered interpretation of biomarker or UMAP results.

    - HTTP 500: Bedrock interpretation failure
    """
    try:
        return outcome_service.interpret_results(params)
    except Exception as exc:
        return JSONResponse({"detail": f"AI interpretation unavailable: {exc}"}, status_code=500)
