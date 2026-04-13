"""Cluster router: K-sweep clustering analysis."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.models import ClusterParams, ClusterResponse
from backend.services.cluster_service import cluster_service
from backend.services.data_service import data_service

router = APIRouter()

_NOT_LOADED = "Dataset not loaded. Call POST /api/data/load first."


@router.post("/run", response_model=ClusterResponse)
async def run_clustering(params: ClusterParams):
    """POST /api/cluster/run — sweep k and return silhouette chart + heatmaps.

    - HTTP 409: data not loaded
    """
    if data_service.get_status().loaded is False:
        return JSONResponse({"detail": _NOT_LOADED}, status_code=409)

    try:
        result = cluster_service.run_clustering(params, data_service)
        return result
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
