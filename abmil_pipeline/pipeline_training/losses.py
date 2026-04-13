"""Supervised Contrastive loss with multi-label pair mask for HANCOCK training."""

import torch
import torch.nn as nn


LABEL_COLUMNS = [
    "primary_tumor_site",
    "histologic_type",
    "hpv_association_p16",
    "pT_stage",
    "pN_stage",
]

# Values that should not be used to define positive pairs
_UNKNOWN_VALUES = {"unknown", "not_tested", "nan", "", "none"}


def build_pair_mask(metadata_rows: list[dict], label_columns: list[str] = LABEL_COLUMNS) -> torch.Tensor:
    """
    Build a boolean positive-pair mask for a batch of slides.

    Two slides are a positive pair if they share the same non-unknown value
    on ANY of the label_columns. This lets not_tested HPV slides still
    contribute to pairs defined by tumor site, stage, or histology.

    Args:
        metadata_rows: list of dicts, one per slide in the batch.
        label_columns: clinical fields to use for pair definition.

    Returns:
        mask: (N, N) bool tensor, True where (i, j) is a positive pair.
              Diagonal is False (a slide is not its own positive).
    """
    n = len(metadata_rows)
    mask = torch.zeros(n, n, dtype=torch.bool)

    for i in range(n):
        for j in range(i + 1, n):
            for col in label_columns:
                val_i = str(metadata_rows[i].get(col, "")).strip().lower()
                val_j = str(metadata_rows[j].get(col, "")).strip().lower()
                if val_i in _UNKNOWN_VALUES or val_j in _UNKNOWN_VALUES:
                    continue
                if val_i == val_j:
                    mask[i, j] = True
                    mask[j, i] = True
                    break  # one matching label is enough

    return mask


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).

    Operates on L2-normalised projections z of shape (N, proj_dim).
    Requires a positive-pair mask from build_pair_mask.

    Args:
        temperature: Scaling factor for the dot-product similarities.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n = z.shape[0]
        device = z.device
        mask = mask.to(device)

        # Re-normalise on device — guards against any drift from MPS
        z = torch.nn.functional.normalize(z, dim=1)

        # Similarity matrix scaled by temperature
        sim = torch.mm(z, z.t()) / self.temperature  # (N, N)

        # Mask out self-similarities with a large negative (not -inf — avoids MPS nan)
        self_mask = torch.eye(n, dtype=torch.bool, device=device)
        sim = sim.masked_fill(self_mask, -1e9)

        # Numerically stable log-softmax
        sim_max = sim.max(dim=1, keepdim=True).values.detach()
        exp_sim = torch.exp(sim - sim_max)
        exp_sim = exp_sim.masked_fill(self_mask, 0.0)  # zero out diagonal
        log_prob = (sim - sim_max) - torch.log(exp_sim.sum(dim=1, keepdim=True).clamp(min=1e-9))

        # For each anchor, average log-prob over its positive pairs
        n_positives = mask.float().sum(dim=1)
        has_positive = n_positives > 0

        if not has_positive.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -(log_prob * mask.float()).sum(dim=1)
        loss = loss[has_positive] / n_positives[has_positive]
        return loss.mean()
