import torch
import torch.nn as nn

"""
DTM.py
Dual TransH Module (self-diversity driven)

Provides a compact DualTransH module that:
 - takes relation encoder outputs and splits into subject/object parts
 - projects them to separate hyperplanes (W_h, W_t)
 - computes a relation vector that emphasizes subject/object differences

API:
  dtm = DualTransH(input_dim, proj_dim)
  c_ij = dtm(rel_enc_output)

The rel_enc_output is expected shape [M, 2*r] and will be split to [M, r] x2
"""


class DualTransH(nn.Module):
    def __init__(self, in_dim, split_dim=None):
        super().__init__()
        # in_dim is 2*r if we follow paper; split_dim is r
        if split_dim is None:
            assert in_dim % 2 == 0, "in_dim must be even when split_dim not provided"
            split_dim = in_dim // 2
        self.r = split_dim
        # learnable transforms to produce normal vectors (w_h, w_t) used for TransH projection
        self.W_h = nn.Linear(self.r, self.r, bias=False)
        self.W_t = nn.Linear(self.r, self.r, bias=False)
        # small FFN to get final relation representation
        self.ffn = nn.Sequential(
            nn.Linear(self.r, self.r),
            nn.ReLU(),
            nn.Linear(self.r, self.r)
        )

    def forward(self, rel_enc):
        """rel_enc: [M, 2*r] -> returns c_ij: [M, r]
        """
        h, t = rel_enc[:, :self.r], rel_enc[:, self.r: self.r*2]
        eps = 1e-6
        # compute normal vectors and normalize
        w_h = self.W_h(h)
        w_t = self.W_t(t)
        w_h = w_h / (w_h.norm(p=2, dim=1, keepdim=True) + eps)
        w_t = w_t / (w_t.norm(p=2, dim=1, keepdim=True) + eps)

        # TransH-like projection: e_proj = e - w * (w^T e)
        proj_h = h - w_h * (h * w_h).sum(dim=1, keepdim=True)
        proj_t = t - w_t * (t * w_t).sum(dim=1, keepdim=True)

        diff = proj_h - proj_t
        return self.ffn(diff)
