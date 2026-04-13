"""GatedAttentionABMIL encoder with optional projection head for metric learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedAttentionABMIL(nn.Module):
    """
    Gated Attention ABMIL encoder.

    Takes a bag of N patch embeddings and produces a single slide-level
    embedding M via learned attention aggregation. The classification head
    from the original design is removed — this module is the encoder only.

    For metric learning, attach a ProjectionHead on top of M during training,
    then discard it. Use M directly for FAISS indexing and retrieval.
    """

    def __init__(self, embed_dim: int = 1536, attention_dim: int = 128):
        super().__init__()
        self.attention_v = nn.Sequential(nn.Linear(embed_dim, attention_dim), nn.Tanh())
        self.attention_u = nn.Sequential(nn.Linear(embed_dim, attention_dim), nn.Sigmoid())
        self.attention_w = nn.Linear(attention_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        """
        Args:
            x: (N, embed_dim) patch embeddings for a single slide.
            return_attention: if True, also return attention weights (N,).
        Returns:
            M: (1, embed_dim) slide embedding.
            A: (N,) attention weights — only if return_attention=True.
        """
        a_v = self.attention_v(x)                        # (N, L)
        a_u = self.attention_u(x)                        # (N, L)
        a = self.attention_w(a_v * a_u)                  # (N, 1)
        a = F.softmax(a, dim=0).transpose(0, 1)          # (1, N)
        m = torch.mm(a, x)                               # (1, embed_dim)
        if return_attention:
            return m, a.squeeze(0)
        return m


class ProjectionHead(nn.Module):
    """
    MLP projection head used during SupCon training only.
    Discarded after training — M from the encoder goes into FAISS.
    """

    def __init__(self, embed_dim: int = 1536, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalised projection of shape (1, proj_dim)."""
        z = self.net(m)
        return F.normalize(z, dim=1)


class ABMILWithProjection(nn.Module):
    """Encoder + projection head combined for training convenience."""

    def __init__(self, embed_dim: int = 1536, attention_dim: int = 128, proj_dim: int = 256):
        super().__init__()
        self.encoder = GatedAttentionABMIL(embed_dim, attention_dim)
        self.projector = ProjectionHead(embed_dim, proj_dim)

    def forward(self, x: torch.Tensor):
        """Returns (M, z): slide embedding and L2-normalised projection."""
        m = self.encoder(x)
        z = self.projector(m)
        return m, z
