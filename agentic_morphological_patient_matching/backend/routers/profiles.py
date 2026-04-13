"""Profiles router: build or check composition profiles."""

import json

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from backend.services.composition_service import (
    BuildInProgressError,
    S3AccessError,
    composition_service,
)
from backend.services.data_service import data_service

router = APIRouter()

from pathlib import Path

_PROFILES_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "02_Morphological_Patient_Similarity_Retrieval" / "profiles")


@router.post("/build")
async def build_profiles():
    """POST /api/profiles/build — build composition profiles with SSE progress streaming.

    - HTTP 409: build already in progress (Requirement 6.3)
    - HTTP 502: S3 access failure (Requirement 6.4)
    - Streams ``text/event-stream`` while the build runs (Requirement 6.1)
    """
    if composition_service.is_building():
        return JSONResponse(
            {"detail": "A profile build is already in progress."},
            status_code=409,
        )

    # Eagerly check data is loaded before starting the stream
    if data_service.get_status().loaded is False:
        return JSONResponse(
            {"detail": "Dataset not loaded. Call POST /api/data/load first."},
            status_code=409,
        )

    async def _stream():
        try:
            async for chunk in composition_service.build_profiles(
                profiles_dir=_PROFILES_DIR,
                data_service=data_service,
            ):
                yield chunk
        except BuildInProgressError as exc:
            yield _sse_event("error", {"message": str(exc)})
        except S3AccessError as exc:
            # Emit the S3 error as an SSE event; HTTP status is already 200
            # because headers were sent when streaming started.
            yield _sse_event("s3_error", {"message": str(exc)})

    return StreamingResponse(_stream(), media_type="text/event-stream")


@router.get("/status")
async def profiles_status():
    """GET /api/profiles/status — whether composition profiles are loaded."""
    profiles = composition_service.get_profiles()
    return {
        "profiles_ready": profiles is not None,
        "n_patients": len(profiles) if profiles is not None else 0,
        "building": composition_service.is_building(),
    }


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
