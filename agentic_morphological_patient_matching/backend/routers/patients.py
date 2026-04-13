"""Patients router: list all patient IDs for autocomplete."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.services.data_service import data_service
from backend.services.retrieval_service import retrieval_service

router = APIRouter()

_NOT_LOADED = "Dataset not loaded. Call POST /api/data/load first."


@router.get("/patients")
async def list_patients():
    """GET /api/patients — return sorted list of all patient IDs (Requirement 9.1)."""
    if data_service.get_status().loaded is False:
        return JSONResponse({"detail": _NOT_LOADED}, status_code=409)

    try:
        patients = retrieval_service.list_patients(data_service)
        return JSONResponse(content=patients)
    except RuntimeError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
