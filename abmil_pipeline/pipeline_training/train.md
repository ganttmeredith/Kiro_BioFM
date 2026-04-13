# ABMIL Patient Matching Pipeline: HANCOCK Dataset

## Objective

Train an ABMIL encoder that produces **slide-level embeddings optimized for patient similarity retrieval** — not binary classification. Once trained, new patients are embedded via a single forward pass and added to a FAISS index. No retraining is required when new data arrives.

---

## Why Not Classification?

The original plan supervised ABMIL with `hpv_association_p16` (binary BCE loss). This has two problems:

1. **860 of 1534 slides are `not_tested`** — over half the dataset has no HPV label, wasting available morphological signal.
2. **Retraining on every new patient is not feasible** — a classifier head doesn't generalize to an open-ended retrieval problem.

The correct framing is **metric learning**: train the encoder so that clinically similar patients are close in embedding space, then use nearest-neighbor search at inference time.

---

## Architecture

### Encoder: GatedAttentionABMIL

Unchanged from the original design. Takes a bag of N patch embeddings and produces a single slide-level vector via learned attention aggregation.

- Input: `N × 1536` (patch embeddings from H-optimus-0, streamed from S3)
- Attention: gated attention (V, U, w) → softmax weights → weighted sum
- Output: `1 × 1536` slide embedding `M`
- Projection head (training only): `Linear(1536 → 256) → L2-normalize` — maps `M` into a compact metric space

The projection head is **discarded after training**. The raw `M` vector is used for indexing and retrieval.

### Loss: Supervised Contrastive (SupCon)

Use all available clinical labels as supervision signal — not just HPV. Define a **similarity label** between two slides as positive if they share the same value on any of:

- `primary_tumor_site`
- `histologic_type`
- `hpv_association_p16` (where available)
- `pT_stage`, `pN_stage`

Slides with `not_tested` HPV simply contribute to pairs defined by the other labels. All 1534 slides participate in training.

Loss: `SupConLoss` (Khosla et al. 2020) over projected embeddings within each batch.

---

## Dataset Strategy

### Using All 1534 Slides

| Slides | Count | Role |
|---|---|---|
| HPV labeled (pos/neg) | 674 | Contribute to HPV-based pairs + all other label pairs |
| `not_tested` | 860 | Contribute to pairs defined by tumor site, stage, histology |

No slides are dropped. The multi-label SupCon loss handles partial label availability naturally.

### Patient-Level Grouping

Patient ID is extracted from `slide_name` (e.g. `patient492`). GroupKFold on `patient_id` ensures no patient appears in both train and validation splits.

### Data Source

- Primary: `.h5` patch embeddings streamed from S3 via `boto3` + `h5py` (N patches × 1536 per slide)
- Fallback: pre-computed 1536-d embeddings from `enriched_slide_embeddings.csv` (single vector per slide, shape `1 × 1536`) — enables training without S3 access

---

## Inference: Adding New Patients

```
new_slide.h5  →  HancockS3PatchDataset  →  GatedAttentionABMIL (frozen)  →  M (1536-d)
                                                                                  ↓
                                                                         faiss_index.add(M)
```

No retraining. The FAISS index is updated incrementally. Retrieval is `faiss_index.search(query_embedding, k=10)`.

---

## File Structure

```
abmil_pipeline/abmil_hancock/
├── data_loaders.py   — HancockS3PatchDataset, load_hancock_manifest, pair sampler
├── model.py          — GatedAttentionABMIL encoder + projection head
├── losses.py         — SupConLoss, multi-label pair mask builder
├── train.py          — Training loop (SupCon), saves encoder weights
└── index.py          — FAISS index builder and nearest-neighbor retrieval
```

---

## Training Plan

1. **Batch construction**: sample slides, stream patch embeddings, run through encoder → get `M` per slide
2. **Project**: `M → z` (256-d, L2-normalized) via projection head
3. **Build pair mask**: for each pair in batch, mark positive if they share ≥1 clinical label
4. **SupCon loss**: maximize agreement between positive pairs, push negatives apart
5. **Validate**: silhouette score on held-out patients by `primary_tumor_site` and `histologic_type`

### Key Hyperparameters

| Parameter | Value |
|---|---|
| Encoder output dim | 1536 |
| Projection dim | 256 |
| SupCon temperature | 0.07 |
| Batch size | 32 slides |
| Optimizer | AdamW, lr=1e-4, wd=1e-5 |
| Epochs | 30 |
| GroupKFold splits | 5 |

---

## Verification Plan

1. After training, build FAISS index on all 1534 slides
2. For 20 held-out patients, retrieve top-10 nearest neighbors
3. Check that retrieved patients share the same `primary_tumor_site` and `histologic_type` at a rate significantly above random baseline
4. Compare silhouette score (by `histologic_type`) against the mean-pooled CSV embeddings baseline


### Adding new patients

When new H&E slides are acquired, no retraining is required. The ABMIL encoder is frozen after training — new slides are embedded via a single forward pass and added to the index.

**Step-by-step for a new patient:**

| Step | What to do | Tool |
|---|---|---|
| 1. Extract patch embeddings | Run H-optimus-0 on the new WSI to produce an `.h5` file (N × 1536 patches) and upload to S3 | `abmil_pipeline/utils.py` |
| 2. Add metadata row | Append a row to `enriched_slide_embeddings_abmil.csv` with clinical metadata and the S3 `h5file` URI. Leave `e_0`..`e_1535` blank for now. | Manual / ETL |
| 3. Embed the new slide | Run the frozen encoder on the new `.h5` to produce a 1536-d vector `M` | `abmil_pipeline.abmil_hancock.index` |
| 4. Add to FAISS index | `PatientIndex.add_new_slide(slide_name, M)` — no rebuild needed | `abmil_pipeline.abmil_hancock.index` |
| 5. Update CSV embeddings | Optionally write `M` back into the CSV row (`e_0`..`e_1535`) so the notebook stays in sync | `export_embeddings()` |

```python
from abmil_pipeline.abmil_hancock.index import PatientIndex
import torch

# Load the existing index
idx = PatientIndex.load("hancock_index")

# Embed the new slide (patches already loaded as tensor)
encoder.eval()
with torch.no_grad():
    m = encoder(new_patches).squeeze(0).cpu().numpy()  # (1536,)

# Add to index — immediately queryable, no retraining
idx.add_new_slide("TumorCenter_HE_block10_x1_y1_patient999", m)
```

> **When to retrain:** Retraining is only needed if the cohort grows substantially (e.g. hundreds of new patients) or if new clinical label types become available. The encoder generalises well to new slides from the same scanner and staining protocol. For slides from a different institution or staining protocol, fine-tuning on a small bridging cohort is recommended.