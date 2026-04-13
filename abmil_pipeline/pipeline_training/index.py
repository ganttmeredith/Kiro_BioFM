"""
FAISS index builder and nearest-neighbor retrieval for patient matching.

Demo scale (< 10k slides): IndexFlatIP (exact search, inner product on L2-normalised vectors = cosine).
Production scale (100k+ slides): swap to IndexIVFPQ — see build_index() comments.

Usage:
    # Build once after training
    python -m abmil_pipeline.abmil_hancock.index --encoder abmil_encoder.pth --csv enriched_slide_embeddings.csv

    # Query at inference
    from abmil_pipeline.abmil_hancock.index import PatientIndex
    idx = PatientIndex.load("hancock_index")
    results = idx.query("TumorCenter_HE_block10_x1_y10_patient492", k=10)
"""

import argparse
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch

from .data_loaders import load_hancock_manifest, HancockS3PatchDataset
from .model import GatedAttentionABMIL


def _embed_all(encoder: GatedAttentionABMIL, csv_path: str, device: torch.device) -> tuple[np.ndarray, list[str]]:
    """Run encoder over all slides in the manifest, return (N, 1536) float32 matrix and slide names."""
    df = load_hancock_manifest(csv_path)
    ds = HancockS3PatchDataset(df, return_metadata=False)

    encoder.eval()
    embeddings, slide_names = [], []

    with torch.no_grad():
        for i in range(len(ds)):
            features, _, slide_name = ds[i]
            patches = features.to(device)
            if patches.dim() == 3:
                patches = patches.squeeze(0)       # (N_patches, 1536)
            elif patches.dim() == 1:
                patches = patches.unsqueeze(0)     # (1, 1536)
            m = encoder(patches)                        # (1, 1536)
            embeddings.append(m.squeeze(0).cpu().numpy())
            slide_names.append(slide_name)

    return np.stack(embeddings).astype(np.float32), slide_names


def build_index(
    encoder_path: str,
    csv_path: str = "enriched_slide_embeddings.csv",
    output_dir: str = "hancock_index",
    use_ivfpq: bool = False,
    nlist: int = 64,
    m_pq: int = 64,
):
    """
    Embed all slides and build a FAISS index.

    Demo (default): IndexFlatIP — exact cosine search on L2-normalised vectors.
    Production: pass use_ivfpq=True for IndexIVFPQ — sub-linear search, ~16x memory reduction.

    Saves to output_dir/:
        index.faiss   — FAISS index
        names.txt     — slide names, one per line (positional, matches index rows)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    encoder = GatedAttentionABMIL(embed_dim=1536, attention_dim=128)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device)
    print(f"Loaded encoder from {encoder_path}")

    print("Embedding all slides...")
    matrix, slide_names = _embed_all(encoder, csv_path, device)
    print(f"Embedded {len(slide_names)} slides -> shape {matrix.shape}")

    # L2-normalise so inner product == cosine similarity
    faiss.normalize_L2(matrix)

    dim = matrix.shape[1]
    if use_ivfpq:
        # Production: IVF + Product Quantization
        # nlist: number of Voronoi cells (rule of thumb: sqrt(N))
        # m_pq: number of sub-quantizers (must divide dim=1536; 64 works)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m_pq, 8)
        index.train(matrix)
        print(f"Built IndexIVFPQ (nlist={nlist}, m={m_pq})")
    else:
        index = faiss.IndexFlatIP(dim)
        print("Built IndexFlatIP (exact cosine)")

    index.add(matrix)
    print(f"Index contains {index.ntotal} vectors")

    out = Path(output_dir)
    out.mkdir(exist_ok=True)
    faiss.write_index(index, str(out / "index.faiss"))
    (out / "names.txt").write_text("\n".join(slide_names))
    print(f"Saved index to {output_dir}/")


class PatientIndex:
    """
    Thin wrapper around a saved FAISS index for patient similarity queries.

    Supports:
    - query by slide name
    - query by raw embedding vector (for new patients not yet in the index)
    - incremental add (new patient arrives, no retraining needed)
    """

    def __init__(self, index: faiss.Index, slide_names: list[str], metadata_df: pd.DataFrame):
        self.index = index
        self.slide_names = slide_names
        self._name_to_idx = {name: i for i, name in enumerate(slide_names)}
        self.metadata = metadata_df.set_index("slide_name") if metadata_df is not None else None

    @classmethod
    def load(cls, index_dir: str, metadata_csv: str = "enriched_slide_embeddings.csv") -> "PatientIndex":
        """Load a saved index from build_index() output."""
        d = Path(index_dir)
        index = faiss.read_index(str(d / "index.faiss"))
        slide_names = (d / "names.txt").read_text().splitlines()
        df = pd.read_csv(metadata_csv) if Path(metadata_csv).exists() else None
        print(f"Loaded index: {index.ntotal} vectors, {len(slide_names)} slides")
        return cls(index, slide_names, df)

    def query(
        self,
        query_slide: str,
        k: int = 10,
        exclude_same_patient: bool = True,
    ) -> pd.DataFrame:
        """
        Find k most similar slides to query_slide.

        Returns a DataFrame with columns: rank, slide_name, patient_id, similarity,
        plus available clinical metadata columns.
        """
        if query_slide not in self._name_to_idx:
            raise KeyError(f"Slide '{query_slide}' not in index")

        idx = self._name_to_idx[query_slide]
        vec = self.index.reconstruct(idx).reshape(1, -1).astype(np.float32)
        return self._search(vec, query_slide, k, exclude_same_patient)

    def query_new(
        self,
        embedding: np.ndarray,
        k: int = 10,
    ) -> pd.DataFrame:
        """
        Query with a raw (1536,) embedding for a new patient not yet in the index.
        L2-normalises the vector before search.
        """
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        return self._search(vec, query_slide=None, k=k, exclude_same_patient=False)

    def add_new_slide(self, slide_name: str, embedding: np.ndarray):
        """
        Add a new slide to the index without retraining.
        embedding: (1536,) float32 array (raw encoder output M).
        """
        vec = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(vec)
        self.index.add(vec)
        self.slide_names.append(slide_name)
        self._name_to_idx[slide_name] = len(self.slide_names) - 1
        print(f"Added '{slide_name}' — index now has {self.index.ntotal} vectors")

    def _search(self, vec: np.ndarray, query_slide, k: int, exclude_same_patient: bool) -> pd.DataFrame:
        import re

        def patient_id(name):
            m = re.search(r'patient(\d+)', name)
            return m.group(1) if m else None

        query_patient = patient_id(query_slide) if query_slide else None

        # Over-fetch to allow filtering same-patient slides
        fetch_k = min(k * 5, self.index.ntotal)
        sims, idxs = self.index.search(vec, fetch_k)
        sims, idxs = sims[0], idxs[0]

        rows = []
        for sim, i in zip(sims, idxs):
            if i < 0:
                continue
            name = self.slide_names[i]
            if name == query_slide:
                continue
            if exclude_same_patient and query_patient and patient_id(name) == query_patient:
                continue
            rows.append({"slide_name": name, "similarity": float(sim)})
            if len(rows) == k:
                break

        df = pd.DataFrame(rows)
        df.insert(0, "rank", range(1, len(df) + 1))
        df["patient_id"] = df["slide_name"].apply(patient_id)

        if self.metadata is not None:
            meta_cols = ["primary_tumor_site", "histologic_type", "hpv_association_p16", "pT_stage", "pN_stage"]
            for col in meta_cols:
                if col in self.metadata.columns:
                    df[col] = df["slide_name"].map(self.metadata[col])

        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", default="abmil_encoder.pth")
    parser.add_argument("--csv", default="enriched_slide_embeddings.csv")
    parser.add_argument("--output", default="hancock_index")
    parser.add_argument("--ivfpq", action="store_true", help="Use IndexIVFPQ for production scale")
    args = parser.parse_args()
    build_index(args.encoder, args.csv, args.output, use_ivfpq=args.ivfpq)
