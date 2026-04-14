import os
import sys
import configparser
from pathlib import Path

# ── AWS credentials — must happen before ANY other imports ────────────────────
# Read the pathologyworkshop profile directly from ~/.aws/credentials and
# inject as env vars so boto3, strands-agents, s3fs all pick them up
# regardless of import order or default session state.
# In containers (ECS/AgentCore) the keys won't be present — env vars are
# already set by the task/execution role, so this block is a no-op there.
_AWS_PROFILE = os.environ.get("AWS_PROFILE", "pathologyworkshop")
_AWS_REGION   = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")

if "AWS_ACCESS_KEY_ID" not in os.environ:
    _creds_file = Path.home() / ".aws" / "credentials"
    if _creds_file.exists():
        _cfg = configparser.ConfigParser()
        _cfg.read(_creds_file)
        if _AWS_PROFILE in _cfg:
            _sec = _cfg[_AWS_PROFILE]
            if "aws_access_key_id" in _sec:
                os.environ["AWS_ACCESS_KEY_ID"]     = _sec["aws_access_key_id"]
                os.environ["AWS_SECRET_ACCESS_KEY"]  = _sec["aws_secret_access_key"]
                if "aws_session_token" in _sec:
                    os.environ["AWS_SESSION_TOKEN"]  = _sec["aws_session_token"]
                os.environ["AWS_DEFAULT_REGION"]     = _AWS_REGION

# Make the umap_retrieval package importable regardless of working directory.
# Local dev: common/ folder at the project root.
_UMAP_PKG_DIR = Path(__file__).resolve().parent.parent.parent / "common"
if str(_UMAP_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_UMAP_PKG_DIR))

# Container path: umap_retrieval_pkg/ copied into the Docker build context.
_UMAP_PKG_CONTAINER = Path(__file__).resolve().parent.parent / "umap_retrieval_pkg"
if _UMAP_PKG_CONTAINER.exists() and str(_UMAP_PKG_CONTAINER) not in sys.path:
    sys.path.insert(0, str(_UMAP_PKG_CONTAINER))

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from backend.routers import chat, cluster, data, outcome, patients, profiles, retrieval, status, umap
from backend.services.composition_service import composition_service
from backend.services.data_service import data_service

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Support both local dev layout and containerised layout
_CONTAINER_DATA_DIR = Path(__file__).resolve().parent.parent

def _resolve_path(local: Path, container: Path) -> str:
    """Return container path if it exists, otherwise fall back to local dev path."""
    return str(container) if container.exists() else str(local)

_PROFILES_DIR = _resolve_path(
    _PROJECT_ROOT / "data" / "profiles",
    _CONTAINER_DATA_DIR / "profiles",
)
_DEFAULT_CSV = _resolve_path(
    _PROJECT_ROOT / "data" / "enriched_slide_embeddings.csv",
    _CONTAINER_DATA_DIR / "enriched_slide_embeddings.csv",
)
# Allow env-var override for both paths
_PROFILES_DIR = os.environ.get("PROFILES_DIR", _PROFILES_DIR)
_DEFAULT_CSV = os.environ.get("DEFAULT_CSV", _DEFAULT_CSV)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Auto-load the default dataset on startup if it exists
    import asyncio
    if Path(_DEFAULT_CSV).exists():
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, data_service.load, _DEFAULT_CSV)
    # Load pre-built profiles from disk on startup (Requirement 6.2)
    composition_service.ensure_profiles(_PROFILES_DIR)
    yield


app = FastAPI(title="Patient Similarity App", version="0.1.0", lifespan=lifespan)

# Middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)
# Allow localhost for dev + any additional origins injected via CORS_ORIGINS env var
# (e.g. the ECS ALB URL: CORS_ORIGINS=http://my-alb-dns.elb.amazonaws.com)
_default_origins = ["http://localhost:5173", "http://localhost:3000"]
_extra_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]
_allowed_origins = _default_origins + _extra_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers — /api/status lives on the status router at /api prefix
app.include_router(status.router, prefix="/api", tags=["status"])
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(umap.router, prefix="/api/umap", tags=["umap"])
app.include_router(cluster.router, prefix="/api/cluster", tags=["cluster"])
app.include_router(retrieval.router, prefix="/api/retrieval", tags=["retrieval"])
app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"])
app.include_router(patients.router, prefix="/api", tags=["patients"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(outcome.router, prefix="/api/outcome", tags=["outcome"])
