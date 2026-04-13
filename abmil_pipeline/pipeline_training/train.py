"""SupCon training loop for GatedAttentionABMIL patient matching encoder."""

import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from .data_loaders import HancockS3PatchDataset, load_hancock_manifest, identify_corrupt_slides
from .losses import SupConLoss, build_pair_mask, LABEL_COLUMNS
from .model import ABMILWithProjection, GatedAttentionABMIL


def _collate_variable_bags(batch):
    """Custom collate: keep variable-length patch bags as a list, stack labels."""
    features, labels, slides, meta_rows = zip(*batch)
    return list(features), torch.stack(labels), list(slides), list(meta_rows)


def _encode_batch(model, feature_list, device):
    """Run encoder on a list of variable-length patch bags, return stacked M and z."""
    ms, zs = [], []
    for patches in feature_list:
        patches = patches.to(device)
        if patches.dim() == 1:
            patches = patches.unsqueeze(0)
        elif patches.dim() == 3:
            patches = patches.squeeze(0)
        m, z = model(patches)
        ms.append(m)
        zs.append(z)
    return torch.cat(ms, dim=0), torch.cat(zs, dim=0)


def _silhouette_on_set(model, loader, device, label_col="primary_tumor_site"):
    """
    Compute silhouette score on a set of slides — same metric as the notebook.
    Uses cosine distance on L2-normalised embeddings, patient-deduplicated.
    Only slides with a known (non-unknown, non-not_tested) label contribute.
    """
    model.eval()
    pid_embeddings: dict = {}   # patient_id -> list of M vectors
    pid_label: dict = {}        # patient_id -> label string

    with torch.no_grad():
        for feature_list, _, _, meta_rows in loader:
            ms, _ = _encode_batch(model, feature_list, device)
            for i, row in enumerate(meta_rows):
                val = str(row.get(label_col, "unknown")).strip().lower()
                if val in ("unknown", "not_tested", "nan", ""):
                    continue
                pid = str(row.get("patient_id", row.get("slide_name", i)))
                pid_embeddings.setdefault(pid, []).append(ms[i].cpu().numpy())
                pid_label[pid] = val

    if len(pid_label) < 4 or len(set(pid_label.values())) < 2:
        return float("nan")

    # Mean-pool per patient — matches notebook's embedding_silhouette_scores
    pids = list(pid_label.keys())
    matrix = normalize(
        np.stack([np.mean(pid_embeddings[p], axis=0) for p in pids]), norm="l2"
    )
    labels = [pid_label[p] for p in pids]
    return float(silhouette_score(matrix, labels, metric="cosine"))
    """Run encoder on a list of variable-length patch bags, return stacked (N, embed_dim) M and z."""
    ms, zs = [], []
    for patches in feature_list:
        patches = patches.to(device)
        if patches.dim() == 1:
            patches = patches.unsqueeze(0)         # (1, 1536)
        elif patches.dim() == 3:
            patches = patches.squeeze(0)           # (N_patches, 1536)
        m, z = model(patches)                      # (1, 1536), (1, proj_dim)
        ms.append(m)
        zs.append(z)
    return torch.cat(ms, dim=0), torch.cat(zs, dim=0)  # (batch, 1536), (batch, proj_dim)


def train_abmil(
    csv_path: str = "enriched_slide_embeddings.csv",
    output_path: str = "abmil_encoder.pth",
    num_epochs: int = 50,
    patience: int = 10,
    batch_size: int = 32,
    lr: float = 3e-4,
    temperature: float = 0.1,
):
    """
    Train GatedAttentionABMIL with SupCon loss for patient similarity retrieval.

    Uses all 1534 slides — not_tested HPV slides contribute to pairs defined
    by tumor site, stage, and histology. No slides are dropped.

    Stops early when val silhouette hasn't improved for `patience` epochs.
    Best encoder weights are saved automatically on each improvement.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_hancock_manifest(csv_path)
    print(f"Total slides: {len(df)}")
    print(f"Label distribution (hpv_association_p16):\n{df['hpv_association_p16'].value_counts()}\n")

    # Pre-filter corrupt H5 files — avoids per-slide warning spam during training
    corrupt_slides = identify_corrupt_slides(df)
    if corrupt_slides:
        df = df[~df["slide_name"].isin(corrupt_slides)].reset_index(drop=True)
        print(f"Slides after filtering corrupt H5s: {len(df)}\n")

    # Hold out ~10% of patients for silhouette monitoring only — no gradient updates
    # GroupShuffleSplit ensures no patient appears in both sets
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    train_idx, monitor_idx = next(gss.split(df, groups=df["patient_id"].values))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    monitor_df = df.iloc[monitor_idx].reset_index(drop=True)
    print(f"Train: {len(train_df)} slides | Monitor (silhouette only): {len(monitor_df)} slides")

    train_loader = DataLoader(
        HancockS3PatchDataset(train_df, return_metadata=True),
        batch_size=batch_size, shuffle=True,
        collate_fn=_collate_variable_bags,
        num_workers=4,
        multiprocessing_context='spawn',
        persistent_workers=True,
    )
    monitor_loader = DataLoader(
        HancockS3PatchDataset(monitor_df, return_metadata=True),
        batch_size=batch_size, shuffle=False,
        collate_fn=_collate_variable_bags,
        num_workers=2,
        multiprocessing_context='spawn',
        persistent_workers=True,
    )

    model = ABMILWithProjection(embed_dim=1536, attention_dim=128, proj_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # Linear warmup for 5 epochs, then hold flat — better for SupCon than cosine decay
    def lr_lambda(epoch):
        if epoch < 5:
            return (epoch + 1) / 5
        return 1.0

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = SupConLoss(temperature=temperature)

    best_sil = float("-inf")
    epochs_without_improvement = 0
    warmup_epochs = 15  # don't early-stop until embeddings have had time to settle

    for epoch in range(num_epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        t0 = time.time()

        for feature_list, _, _, meta_rows in train_loader:
            _, z = _encode_batch(model, feature_list, device)
            mask = build_pair_mask(meta_rows, LABEL_COLUMNS)
            loss = criterion(z, mask)

            if torch.isnan(loss):
                continue  # skip batch if loss is nan (e.g. all not_tested batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Same silhouette metric as the notebook: cosine, L2-normalised, patient-deduplicated
        sil = _silhouette_on_set(model, monitor_loader, device)
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Silhouette (tumor_site): {sil:.4f} | "
            f"Time: {time.time() - t0:.1f}s"
        )

        if sil > best_sil + 1e-4:
            best_sil = sil
            epochs_without_improvement = 0
            torch.save(model.encoder.state_dict(), output_path)
            print(f"  -> Saved encoder (silhouette={sil:.4f})")
        elif epoch >= warmup_epochs:
            epochs_without_improvement += 1
            print(f"  -> No improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping at epoch {epoch + 1} — silhouette plateau for {patience} epochs.")
                break
        else:
            print(f"  -> Warmup phase ({epoch + 1}/{warmup_epochs})")

    print(f"\nTraining complete. Best silhouette: {best_sil:.4f}")
    print(f"Encoder saved to {output_path}")
    return model.encoder


def export_embeddings(
    encoder_path: str = "abmil_encoder.pth",
    csv_path: str = "enriched_slide_embeddings.csv",
    output_csv: str = "enriched_slide_embeddings_abmil.csv",
):
    """
    Re-embed all slides through the trained encoder and write a new CSV.

    The output CSV is a drop-in replacement for enriched_slide_embeddings.csv —
    same structure, same metadata columns, but e_0..e_1535 are now the
    attention-aggregated ABMIL embeddings instead of mean-pooled H-optimus-0 vectors.

    The notebook and umap_retrieval pipeline require no changes — just point
    them at the new CSV.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    encoder = GatedAttentionABMIL(embed_dim=1536, attention_dim=128)
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    encoder.to(device).eval()
    print(f"Loaded encoder from {encoder_path}")

    df = load_hancock_manifest(csv_path)
    ds = HancockS3PatchDataset(df, return_metadata=False)

    # Identify metadata columns (everything before e_0)
    embedding_cols = [f'e_{i}' for i in range(1536)]
    meta_cols = [c for c in df.columns if c not in embedding_cols]

    new_embeddings = []
    with torch.no_grad():
        for i in range(len(ds)):
            features, _, _ = ds[i]
            patches = features.to(device)
            if patches.dim() == 3:
                patches = patches.squeeze(0)
            elif patches.dim() == 1:
                patches = patches.unsqueeze(0)
            m = encoder(patches).squeeze(0).cpu().numpy()   # (1536,)
            new_embeddings.append(m)
            if (i + 1) % 100 == 0:
                print(f"  Embedded {i + 1}/{len(ds)} slides")

    emb_matrix = np.stack(new_embeddings)                   # (N, 1536)
    emb_df = pd.DataFrame(emb_matrix, columns=embedding_cols)
    out = pd.concat([df[meta_cols].reset_index(drop=True), emb_df], axis=1)
    out.to_csv(output_csv, index=False)
    print(f"\nSaved {len(out)} slides -> {output_csv}")
    print("Point the notebook at this CSV to use ABMIL embeddings.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-only", action="store_true", help="Skip training, just export embeddings")
    parser.add_argument("--encoder", default="abmil_encoder.pth")
    parser.add_argument("--csv", default="enriched_slide_embeddings.csv")
    parser.add_argument("--output-csv", default="enriched_slide_embeddings_abmil.csv")
    args = parser.parse_args()

    if args.export_only:
        export_embeddings(args.encoder, args.csv, args.output_csv)
    else:
        encoder = train_abmil(csv_path=args.csv)
        export_embeddings(args.encoder, args.csv, args.output_csv)
