import torch
import torch.nn as nn

"""
SOL.py
Semantic Oriented Learning module (SOL)

Provides:
- predicate_semantic_embeddings: build embeddings from tokenized predicate labels + weights
- MultiFusion (lightweight fusion block)
- SemanticOrientedLearning class that computes teacher (semantic-guided) and student
  pair-level contexts and produces predicate logits for both (to be used in KL teacher loss)

This is a compact, well-documented reimplementation matching the ART paper
"""


class MultiFusion(nn.Module):
    """A small multi-fusion block: project inputs independently, concat, fuse."""
    def __init__(self, in_dims, hidden_dim=256, out_dim=256):
        super().__init__()
        # in_dims: list of input dims
        self.projs = nn.ModuleList([nn.Linear(d, hidden_dim) for d in in_dims])
        self.fuse = nn.Sequential(
            nn.Linear(len(in_dims) * hidden_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim)
        )

    def forward(self, *inputs):
        # inputs are tensors of shape [N, D_i]
        feats = [p(x) for p, x in zip(self.projs, inputs)]
        cat = torch.cat(feats, dim=-1)
        return self.fuse(cat)


class SemanticOrientedLearning(nn.Module):
    """SOL module.

    Usage:
      sol = SemanticOrientedLearning(glove_dim=200, pair_feat_dim=256, global_feat_dim=512)
      u_teacher, u_student, p_teacher_logits, p_student_logits = sol(pair_ctx, global_feat, pair_pos, predicate_ids)

    - pair_ctx: [M, D] pair-level contexts coming from ART (or ART-like)
    - global_feat: [K, G] global per-proposal features (pooled), can be used to build additional context
    - pair_pos: [M, P] positional encodings for pairs
    - predicate_ids: optional tensor of predicate class indices for building embeddings

    The module returns fused contexts and logits for teacher (semantic-guided) and student (no predicate semantic)
    """

    def __init__(self, glove_dim=200, pair_ctx_dim=256, pair_pos_dim=128, global_feat_dim=512, num_predicates=51, out_dim=256):
        super().__init__()
        # MF for pair-level contexts (pair message, pair positional, pair positional/aux)
        self.pair_mf = MultiFusion([pair_ctx_dim, pair_pos_dim, pair_pos_dim], hidden_dim=256, out_dim=out_dim)
        # MF for global features and pair positional
        self.glob_mf = MultiFusion([global_feat_dim, pair_pos_dim], hidden_dim=256, out_dim=out_dim)
        # predicate semantic projection
        self.pred_proj = nn.Linear(glove_dim, out_dim)

        # final classifier head for predicate logits from fused context
        self.cls = nn.Sequential(
            nn.Linear(out_dim * 3, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, num_predicates)
        )

    def forward(self, pair_ctx, pair_ctx_other, pair_pos, global_feat=None, pred_semantic=None):
        """Compute u (teacher) and u' (student) contexts and logits.

        pair_ctx: [M, D] - enhanced pair context (e.g., from ART stage2)
        pair_ctx_other: [M, D] - auxiliary pair context (e.g., global fusion)
        pair_pos: [M, P] - pair positional features
        global_feat: [K, G] - optional global features per proposal (unused shape-wise here)
        pred_semantic: [num_predicates, glove_dim] or None
        """
        # fuse contexts
        f1 = self.pair_mf(pair_ctx, pair_ctx_other, pair_pos)   # [M, out_dim]
        # global_feat can be [1, G] - repeat per pair
        g = global_feat.repeat(f1.size(0), 1) if (global_feat is not None and global_feat.size(0) == 1) else (global_feat if global_feat is not None else pair_ctx)
        f2 = self.glob_mf(g, pair_pos)
        # semantic embeddings
        if pred_semantic is not None:
            # teacher uses predicate semantic guidance: we compute a residual semantic vector for each pair
            # pred_semantic may be (num_predicates, glove_dim) or (M, glove_dim)
            s = self.pred_proj(pred_semantic) if pred_semantic.ndim == 2 else self.pred_proj(pred_semantic)
            # if pred_semantic is class-level, expand: here we expect pred_semantic per pair; if not, broadcasting
            if s.size(0) != f1.size(0):
                # try broadcast if possible; otherwise we'll expand first row
                s = s.repeat(f1.size(0), 1)
            f3 = s
        else:
            # student: no external semantics
            f3 = torch.zeros_like(f1)

        # final fused pair context
        fused = torch.cat([f1, f2, f3], dim=-1)
        logits = self.cls(fused)
        return fused, logits


def build_predicate_semantics(label_list, glove_dict, emb_dim=200):
    """Given a list of predicate strings, build semantic embeddings using glove dict.
    Rules:
      - split tokens, weight tokens (simple heuristic)
      - return tensor [num_predicates, emb_dim]
    """
    import numpy as np
    vecs = []
    for lab in label_list:
        toks = lab.lower().split()
        if len(toks) == 1:
            v = glove_dict.get(toks[0], np.random.normal(size=(emb_dim,))).astype('float32')
        elif len(toks) == 2:
            # bias first token
            a = 0.7
            v0 = glove_dict.get(toks[0], np.random.normal(size=(emb_dim,))).astype('float32')
            v1 = glove_dict.get(toks[1], np.random.normal(size=(emb_dim,))).astype('float32')
            v = a * v0 + (1 - a) * v1
        else:
            # three or more: center token weight
            mid = len(toks) // 2
            a = 0.7
            vmid = glove_dict.get(toks[mid], np.random.normal(size=(emb_dim,))).astype('float32')
            others = [glove_dict.get(t, np.zeros(emb_dim)).astype('float32') for i, t in enumerate(toks) if i!=mid]
            v = a * vmid + 0.5 * sum(others)
        vecs.append(v)
    return torch.tensor(vecs, dtype=torch.float32)
