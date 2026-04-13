"""Retrieval router: composite patient similarity queries."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models import RetrievalParams, RetrievalResponse
from backend.services.composition_service import composition_service
from backend.services.data_service import data_service
from backend.services.retrieval_service import retrieval_service

router = APIRouter()

_NOT_LOADED = "Dataset not loaded. Call POST /api/data/load first."
_NO_PROFILES = "Composition profiles not built. Call POST /api/profiles/build first."


@router.post("/query", response_model=RetrievalResponse)
async def query_retrieval(params: RetrievalParams):
    """POST /api/retrieval/query — ranked composite similarity retrieval.

    - HTTP 409: data not loaded
    - HTTP 409: profiles not built
    - HTTP 400: patient not found in filtered candidate pool
    """
    if data_service.get_status().loaded is False:
        return JSONResponse({"detail": _NOT_LOADED}, status_code=409)

    if composition_service.get_profiles() is None:
        return JSONResponse({"detail": _NO_PROFILES}, status_code=409)

    try:
        result = retrieval_service.query(params, data_service, composition_service)
        return result
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=400)
    except RuntimeError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=409)
