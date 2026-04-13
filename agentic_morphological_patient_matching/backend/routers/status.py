"""Status router: GET /api/status — current data-load and profiles state."""

from fastapi import APIRouter

from backend.services.composition_service import composition_service
from backend.services.data_service import data_service

router = APIRouter()


@router.get("/status")
async def get_status():
    """GET /api/status — return current data-load state (Requirement 1.1)."""
    status = data_service.get_status()
    profiles = composition_service.get_profiles()
    return {
        "loaded": status.loaded,
        "n_slides": status.n_slides,
        "n_patients": status.n_patients,
        "profiles_ready": profiles is not None,
    }
