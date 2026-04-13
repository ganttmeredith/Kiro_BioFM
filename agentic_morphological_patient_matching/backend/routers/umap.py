"""UMAP router: run UMAP projections on loaded embeddings."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models import UMAPParams, UMAPResponse
from backend.services.data_service import data_service
from backend.services.umap_service import umap_service

router = APIRouter()

_NOT_LOADED = "Dataset not loaded. Call POST /api/data/load first."
_TOO_RESTRICTIVE = "Filters match fewer than 2 slides — too restrictive."


@router.post("/run", response_model=UMAPResponse)
async def run_umap(params: UMAPParams):
    """POST /api/umap/run — project embeddings to 2D and return a Plotly figure.

    - HTTP 409: data not loaded
    - HTTP 400: filters too restrictive (< 2 slides)
    - HTTP 422: out-of-range n_neighbors / min_dist (handled by Pydantic)
    """
    if data_service.get_status().loaded is False:
        return JSONResponse({"detail": _NOT_LOADED}, status_code=409)

    # Validate param ranges (Requirement 2.6)
    if not (5 <= params.n_neighbors <= 50):
        return JSONResponse(
            {"detail": f"n_neighbors must be in [5, 50], got {params.n_neighbors}"},
            status_code=422,
        )
    if not (0.01 <= params.min_dist <= 0.5):
        return JSONResponse(
            {"detail": f"min_dist must be in [0.01, 0.5], got {params.min_dist}"},
            status_code=422,
        )

    try:
        result = umap_service.run_umap(params, data_service)
        return result
    except ValueError as exc:
        msg = str(exc)
        if "fewer than 2" in msg or "too restrictive" in msg.lower():
            return JSONResponse({"detail": _TOO_RESTRICTIVE}, status_code=400)
        return JSONResponse({"detail": msg}, status_code=422)
