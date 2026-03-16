"""
Stub VectorQuantize module compatible with seed-vc's length_regulator.
Provides the same interface as descript-audio-codec's VectorQuantize
without pulling in the entire dac dependency tree.
"""

import torch
import torch.nn as nn


class VectorQuantize(nn.Module):
    """Lightweight vector quantization layer."""

    def __init__(
        self,
        input_dim: int = 512,
        codebook_size: int = 1024,
        codebook_dim: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.in_proj = nn.Linear(input_dim, codebook_dim)
        self.out_proj = nn.Linear(codebook_dim, input_dim)
        self.codebook = nn.Embedding(codebook_size, codebook_dim)

    def forward(self, z):
        z_e = self.in_proj(z)
        d = z_e.unsqueeze(-2) - self.codebook.weight
        d = d.pow(2).sum(-1)
        indices = d.argmin(-1)
        z_q = self.codebook(indices)
        z_q_st = z_e + (z_q - z_e).detach()
        z_q_out = self.out_proj(z_q_st)
        commitment_loss = (z_e.detach() - z_q).pow(2).mean()
        codebook_loss = (z_e - z_q.detach()).pow(2).mean()
        return z_q_out, indices, commitment_loss, codebook_loss
