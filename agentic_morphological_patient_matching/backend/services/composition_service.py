"""CompositionService: loads or builds composition profiles from H5 patch files."""

import json
import threading
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

import numpy as np

from umap_retrieval.composition import (
    build_composition_profiles_from_metadata,
    build_patient_profiles,
    load_profiles,
    save_profiles,
)

_PROFILES_NPZ = "patient_profiles.npz"
_CODEBOOK_JL = "codebook.joblib"


class CompositionService:
    """Loads or builds composition profiles; caches them in memory.

    Profiles are persisted to ``profiles_dir/patient_profiles.npz`` and
    ``profiles_dir/codebook.joblib``.  On startup, ``ensure_profiles``
    loads them from disk if they already exist (Requirement 6.2).

    Only one concurrent build is allowed at a time (Requirement 6.3).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._profiles: Optional[Dict[str, np.ndarray]] = None
        self._codebook = None
        self._building: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_profiles(self, profiles_dir: str) -> bool:
        """Load profiles from disk if they exist; no-op otherwise.

        Args:
            profiles_dir: Directory containing ``patient_profiles.npz``
                and ``codebook.joblib``.

        Returns:
            ``True`` if profiles were loaded from disk, ``False`` if the
            files were not present.
        """
        npz_path = Path(profiles_dir) / _PROFILES_NPZ
        cb_path = Path(profiles_dir) / _CODEBOOK_JL

        if not npz_path.exists() or not cb_path.exists():
            return False

        with self._lock:
            patient_profiles, _codebook = load_profiles(str(profiles_dir))
            self._profiles = patient_profiles
            self._codebook = _codebook

        return True

    async def build_profiles(
        self,
        profiles_dir: str,
        data_service,
    ) -> AsyncGenerator[str, None]:
        """Build composition profiles, streaming SSE progress events.

        Enforces a single-concurrent-build lock (Requirement 6.3).
        Streams ``data:`` lines compatible with the SSE protocol.

        On S3 access failure the generator yields an ``error`` event and
        raises an ``S3AccessError`` so the router can return HTTP 502
        (Requirement 6.4).

        Args:
            profiles_dir: Directory to persist the built profiles.
            data_service: DataService instance providing ``metadata_df``.

        Yields:
            SSE-formatted strings (``data: <json>\\n\\n``).

        Raises:
            BuildInProgressError: If a build is already running.
            S3AccessError: If S3 access fails during H5 streaming.
        """
        # Reject concurrent builds (Requirement 6.3)
        with self._build_lock:
            if self._building:
                raise BuildInProgressError(
                    "A profile build is already in progress. "
                    "Please wait for it to complete."
                )
            self._building = True

        try:
            yield _sse("progress", {"message": "Starting profile build…", "pct": 0})

            metadata_df = data_service.get_metadata()
            if metadata_df is None:
                raise RuntimeError(
                    "Dataset not loaded. Call POST /api/data/load first."
                )

            yield _sse("progress", {"message": "Streaming H5 files from S3…", "pct": 10})

            # Build slide-level profiles (may raise on S3 failure)
            try:
                slide_profiles, codebook = build_composition_profiles_from_metadata(
                    metadata_df=metadata_df,
                    n_tissue_types=10,
                    seed=42,
                    max_workers=8,
                )
            except Exception as exc:
                err_str = str(exc)
                # Detect S3 / credential errors (Requirement 6.4)
                if _is_s3_error(err_str):
                    raise S3AccessError(
                        f"S3 read failed: {err_str}. "
                        "Check AWS_PROFILE=pathologyworkshop."
                    ) from exc
                raise

            yield _sse("progress", {"message": "Building patient-level profiles…", "pct": 80})

            patient_profiles = build_patient_profiles(slide_profiles)

            yield _sse("progress", {"message": "Persisting profiles to disk…", "pct": 90})

            save_profiles(patient_profiles, codebook, profiles_dir)

            with self._lock:
                self._profiles = patient_profiles
                self._codebook = codebook

            yield _sse(
                "done",
                {
                    "message": "Profile build complete.",
                    "n_patients": len(patient_profiles),
                    "pct": 100,
                },
            )

        except (BuildInProgressError, S3AccessError):
            raise
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})
            raise
        finally:
            with self._build_lock:
                self._building = False

    def get_profiles(self) -> Optional[Dict[str, np.ndarray]]:
        """Return the cached patient profiles, or ``None`` if not yet built."""
        return self._profiles

    def update_profiles(self, profiles: Dict[str, np.ndarray]) -> None:
        """Merge profiles into the cached profiles dict in-place."""
        with self._lock:
            if self._profiles is not None:
                self._profiles.update(profiles)
            else:
                self._profiles = profiles

    def is_building(self) -> bool:
        """Return ``True`` if a profile build is currently in progress."""
        return self._building


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class BuildInProgressError(RuntimeError):
    """Raised when a second build is requested while one is already running."""


class S3AccessError(RuntimeError):
    """Raised when S3 access fails during H5 file streaming."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event string."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _is_s3_error(message: str) -> bool:
    """Heuristic: detect S3 / credential-related error messages."""
    lower = message.lower()
    s3_keywords = (
        "s3",
        "credential",
        "access denied",
        "nosuchbucket",
        "nosuchkey",
        "botocore",
        "boto3",
        "aws",
        "permission",
        "forbidden",
        "unauthorized",
        "endpoint",
        "connection",
        "timeout",
    )
    return any(kw in lower for kw in s3_keywords)


# Module-level singleton shared by all routers
composition_service = CompositionService()
