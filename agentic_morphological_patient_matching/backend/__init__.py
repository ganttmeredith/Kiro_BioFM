# Backend package — ensure umap_retrieval is importable
import sys
from pathlib import Path

_UMAP_PKG_DIR = Path(__file__).resolve().parent.parent.parent / "02_Morphological_Patient_Similarity_Retrieval"
if str(_UMAP_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_UMAP_PKG_DIR))
