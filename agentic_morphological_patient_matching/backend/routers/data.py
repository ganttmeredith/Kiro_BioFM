"""Data router: load CSV dataset and expose app status."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse

from backend.models import DataSummary, LoadRequest
from backend.services.data_service import data_service

router = APIRouter()


@router.post("/upload", response_model=DataSummary)
async def upload_data(file: UploadFile = File(...)):
    """POST /api/data/upload — upload a CSV file directly and load it.

    Saves the uploaded file to a temp location, then loads it via DataService.
    - HTTP 422: CSV validation failure
    - HTTP 500: unexpected error
    """
    try:
        contents = await file.read()
        # Write to a named temp file so DataService can read it by path
        suffix = Path(file.filename or "upload.csv").suffix or ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        summary = data_service.load_from_temp(tmp_path, display_path=file.filename or "uploaded.csv")
        return JSONResponse(content=summary.model_dump(by_alias=True))
    except ValueError as exc:
        return JSONResponse({"detail": str(exc)}, status_code=422)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)


@router.get("/summary", response_model=DataSummary)
async def get_summary():
    """GET /api/data/summary — return cached DataSummary if data is loaded.

    - HTTP 409: dataset not loaded yet
    """
    summary = data_service.get_summary()
    if summary is None:
        return JSONResponse(
            {"detail": "Dataset not loaded. Call POST /api/data/load first."},
            status_code=409,
        )
    return JSONResponse(content=summary.model_dump(by_alias=True))


@router.post("/load", response_model=DataSummary)
async def load_data(body: LoadRequest):
    """POST /api/data/load — load and validate the CSV dataset.

    Returns DataSummary on success.
    - HTTP 400: path traversal detected
    - HTTP 409: dataset is currently loading (concurrent guard)
    - HTTP 422: CSV validation failure (missing columns, duplicates, etc.)
    """
    try:
        summary = data_service.load(body.csv_path)
        return JSONResponse(content=summary.model_dump(by_alias=True))
    except ValueError as exc:
        msg = str(exc)
        if "traversal" in msg.lower() or "outside project root" in msg.lower():
            return JSONResponse({"detail": msg}, status_code=400)
        # CSV validation failure → 422
        return JSONResponse({"detail": msg}, status_code=422)
    except Exception as exc:
        return JSONResponse({"detail": str(exc)}, status_code=500)
