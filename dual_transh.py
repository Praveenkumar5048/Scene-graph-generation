import torch
import torch.nn as nn
import torch.nn.functional as F


class DualTransHScorer(nn.Module):
    """
    Dual TransH-style triple scorer.
    For each relation r, learn:
      - n_s[r]: subject-side normal vector (defines hyperplane)
      - n_o[r]: object-side normal vector
      - d[r]: relation translation vector
    Score(s, r, o) = -|| proj_s(e_s, n_s[r]) + d[r] - proj_o(e_o, n_o[r]) ||_2

    Where proj_x(e, n) = e - <e, n_hat> * n_hat and n_hat = n / ||n||.
    """

    def __init__(self, num_relations: int, embed_dim: int):
        super().__init__()
        self.num_rel = num_relations
        self.dim = embed_dim
        # Parameters
        self.rel_norm_subj = nn.Parameter(torch.randn(num_relations, embed_dim) * 0.1)
        self.rel_norm_obj = nn.Parameter(torch.randn(num_relations, embed_dim) * 0.1)
        self.rel_trans = nn.Parameter(torch.randn(num_relations, embed_dim) * 0.1)

    @staticmethod
    def _project(e: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
        # e: [B, D] or [D]; n: [B, D] or [D]
        n_hat = F.normalize(n, p=2, dim=-1)
        # projection onto hyperplane orthogonal to n_hat
        coef = torch.sum(e * n_hat, dim=-1, keepdim=True)  # [B, 1]
        return e - coef * n_hat

    def forward(self, obj_embeds: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """
        obj_embeds: [K, D]
        pairs: [P, 2] over local indices
        returns: scores [P, R] (higher is better)
        """
        if pairs.numel() == 0:
            return torch.zeros(0, self.num_rel, device=obj_embeds.device)

        subj = obj_embeds[pairs[:, 0]]  # [P, D]
        obj = obj_embeds[pairs[:, 1]]   # [P, D]

        # Expand relations across P pairs
        n_s = self.rel_norm_subj.unsqueeze(0).expand(subj.size(0), -1, -1)  # [P, R, D]
        n_o = self.rel_norm_obj.unsqueeze(0).expand(subj.size(0), -1, -1)  # [P, R, D]
        d = self.rel_trans.unsqueeze(0).expand(subj.size(0), -1, -1)       # [P, R, D]

        # Expand entities to [P, R, D]
        subj_e = subj.unsqueeze(1).expand(-1, self.num_rel, -1)
        obj_e = obj.unsqueeze(1).expand(-1, self.num_rel, -1)

        # Project
        proj_s = self._project(subj_e, n_s)
        proj_o = self._project(obj_e, n_o)

        # TransH translation and distance
        diff = proj_s + d - proj_o  # [P, R, D]
        dist = torch.norm(diff, p=2, dim=-1)  # [P, R]
        # Convert to scores (higher is better)
        scores = -dist
        return scores
