"""H5Service: download H5 patch files and recompute ABMIL slide embeddings.

For a given list of patient IDs this service:
  1. Downloads the corresponding H5 files from S3 (or uses local cache).
  2. Runs the trained GatedAttentionABMIL encoder to produce fresh 1536-d
     slide embeddings.
  3. Merges those embeddings back into the DataService cache so that
     subsequent retrieval / UMAP calls use the ABMIL-derived vectors.
  4. Re-profiles the slides against the existing composition codebook and
     merges the updated profiles into CompositionService.
"""

import concurrent.futures
import os
import re
import subprocess
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_H5_CACHE_DIR = str(Path(__file__).resolve().parent.parent.parent.parent / "data" / "h5_cache")
_ENCODER_PATH = str(Path(__file__).resolve().parent.parent.parent.parent / "abmil_pipeline" / "abmil_encoder.pth")


def _download_h5_for_patients(
    metadata_df,
    patient_ids: List[str],
    local_dir: str = _H5_CACHE_DIR,
    max_workers: int = 8,
) -> List[Tuple[str, str]]:
    """Download H5 files for the given patient IDs, skipping cached files.

    Args:
        metadata_df: DataFrame with 'slide_name' and 'h5file' columns.
        patient_ids: Numeric patient ID strings to download.
        local_dir: Local cache directory.
        max_workers: Parallel download threads.

    Returns:
        List of (slide_name, local_path) for successfully fetched files.
    """
    os.makedirs(local_dir, exist_ok=True)
    pid_set = set(patient_ids)

    rows_to_fetch: List[Tuple[str, str]] = []
    for _, row in metadata_df.iterrows():
        s3_path = row.get("h5file", "")
        if not s3_path or (isinstance(s3_path, float) and np.isnan(s3_path)):
            continue
        s3_path = str(s3_path)
        m = re.search(r"patient(\d+)", str(row.get("slide_name", "")))
        if not m or m.group(1) not in pid_set:
            continue
        rows_to_fetch.append((str(row["slide_name"]), s3_path))

    def _fetch(slide_name: str, s3_path: str) -> Optional[Tuple[str, str]]:
        local_path = os.path.join(local_dir, os.path.basename(s3_path))
        if os.path.exists(local_path):
            return (slide_name, local_path)
        result = subprocess.run(
            ["aws", "s3", "cp", s3_path, local_path],
            capture_output=True, text=True, check=False,
        )
        return (slide_name, local_path) if result.returncode == 0 else None

    results: List[Tuple[str, str]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, sn, sp): sn for sn, sp in rows_to_fetch}
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    return results


def _run_abmil_inference(
    h5_files: List[Tuple[str, str]],
    encoder_path: str = _ENCODER_PATH,
) -> Dict[str, np.ndarray]:
    """Run GatedAttentionABMIL on downloaded H5 files to get slide embeddings.

    Requires torch — raises RuntimeError if not installed (torch is excluded
    from the AgentCore container image to stay within size limits).
    """
    try:
        import torch
        import h5py
    except ImportError as e:
        raise RuntimeError(
            "ABMIL re-embedding requires 'torch' and 'h5py' which are not "
            "installed in this environment. This feature is unavailable in the "
            f"AgentCore runtime. ({e})"
        ) from e

    # Make the ABMIL model importable
    abmil_dir = str(Path(encoder_path).resolve().parent.parent)
    if abmil_dir not in sys.path:
        sys.path.insert(0, abmil_dir)

    from abmil_hancock.model import GatedAttentionABMIL  # type: ignore

    device = torch.device("cpu")
    encoder = GatedAttentionABMIL(embed_dim=1536, attention_dim=128)

    if Path(encoder_path).exists():
        state = torch.load(encoder_path, map_location=device, weights_only=True)
        # The checkpoint may wrap the encoder inside 'encoder.' prefix
        if any(k.startswith("encoder.") for k in state):
            state = {k[len("encoder."):]: v for k, v in state.items()
                     if k.startswith("encoder.")}
        encoder.load_state_dict(state, strict=False)

    encoder.eval()

    embeddings: Dict[str, np.ndarray] = {}
    with torch.no_grad():
        for slide_name, h5_path in h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    feat_key = "features" if "features" in f else "embeddings"
                    patches = torch.tensor(
                        f[feat_key][:].astype(np.float32), device=device
                    )
                m = encoder(patches)          # (1, 1536)
                vec = m.squeeze(0).numpy().astype(np.float64)
                embeddings[slide_name] = vec
            except Exception:  # noqa: BLE001
                pass  # skip corrupt / missing files

    return embeddings


def _recompute_composition_profiles(
    h5_files: List[Tuple[str, str]],
    composition_service,
) -> Dict[str, np.ndarray]:
    """Re-profile downloaded slides against the existing codebook.

    Args:
        h5_files: List of (slide_name, local_h5_path).
        composition_service: CompositionService with loaded codebook.

    Returns:
        Updated patient-level profiles dict (existing + new patients).
    """
    from umap_retrieval.composition import build_patient_profiles, profile_slide

    codebook = getattr(composition_service, "_codebook", None)
    existing_profiles = composition_service.get_profiles() or {}

    if codebook is None:
        return existing_profiles

    new_slide_profiles: Dict[str, np.ndarray] = {}
    for slide_name, local_path in h5_files:
        try:
            new_slide_profiles[slide_name] = profile_slide(local_path, codebook)
        except Exception:  # noqa: BLE001
            pass

    if not new_slide_profiles:
        return existing_profiles

    new_patient_profiles = build_patient_profiles(new_slide_profiles)
    return {**existing_profiles, **new_patient_profiles}


class H5Service:
    """Downloads H5 files and recomputes ABMIL embeddings for a patient cohort."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def enrich_patients(
        self,
        patient_ids: List[str],
        data_service,
        composition_service,
        encoder_path: str = _ENCODER_PATH,
        h5_cache_dir: str = _H5_CACHE_DIR,
        max_workers: int = 8,
    ) -> Dict[str, int]:
        """Download H5 files and recompute embeddings + profiles for patient_ids.

        Merges the fresh ABMIL embeddings back into data_service's embedding
        cache and updates composition_service's profiles in-place.

        Args:
            patient_ids: Numeric patient ID strings (e.g. ['583', '306']).
            data_service: DataService instance (provides metadata + embedding cache).
            composition_service: CompositionService instance (provides profiles).
            encoder_path: Path to abmil_encoder.pth.
            h5_cache_dir: Local directory for H5 file cache.
            max_workers: Parallel download threads.

        Returns:
            Dict with keys 'downloaded', 'embeddings_updated', 'profiles_updated'.
        """
        metadata_df = data_service.get_metadata()
        if metadata_df is None:
            raise RuntimeError("Dataset not loaded.")

        # 1. Download H5 files
        h5_files = _download_h5_for_patients(
            metadata_df, patient_ids, local_dir=h5_cache_dir, max_workers=max_workers
        )
        if not h5_files:
            return {"downloaded": 0, "embeddings_updated": 0, "profiles_updated": 0}

        # 2. Run ABMIL inference
        new_embeddings = _run_abmil_inference(h5_files, encoder_path=encoder_path)

        # 3. Merge embeddings into DataService cache
        data_service.update_embeddings(new_embeddings)

        # 4. Recompute and merge composition profiles
        updated_profiles = _recompute_composition_profiles(h5_files, composition_service)
        if updated_profiles:
            composition_service.update_profiles(updated_profiles)

        return {
            "downloaded": len(h5_files),
            "embeddings_updated": len(new_embeddings),
            "profiles_updated": len(updated_profiles),
        }


# Module-level singleton
h5_service = H5Service()
