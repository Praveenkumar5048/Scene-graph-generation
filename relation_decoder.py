import torch
import torch.nn as nn


class PairGeometry(nn.Module):
    """
    Compute a simple 13-dim geometric feature for a subject-object pair.
    Inputs are [x, y, w, h] for each box in absolute image coordinates.
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(13, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _pair_feats(b1: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
        # b1, b2: [4] = [x, y, w, h]
        xi, yi, wi, hi = b1
        xj, yj, wj, hj = b2

        dx = (xj - xi) / (wi + 1e-6)
        dy = (yj - yi) / (hi + 1e-6)
        dw = torch.log(wj / (wi + 1e-6) + 1e-6)
        dh = torch.log(hj / (hi + 1e-6) + 1e-6)

        x1i, y1i, x2i, y2i = xi, yi, xi + wi, yi + hi
        x1j, y1j, x2j, y2j = xj, yj, xj + wj, yj + hj
        inter_w = torch.clamp(torch.min(x2i, x2j) - torch.max(x1i, x1j), min=0)
        inter_h = torch.clamp(torch.min(y2i, y2j) - torch.max(y1i, y1j), min=0)
        inter_area = inter_w * inter_h
        union_area = wi * hi + wj * hj - inter_area + 1e-6
        iou = inter_area / union_area

        return torch.stack([dx, dy, dw, dh, wi, hi, wj, hj, iou, xi, yi, xj, yj], dim=0)

    def forward(self, boxes_xywh: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """
        boxes_xywh: [K, 4]
        pairs: [P, 2] indices (subject, object)
        returns: [P, 128]
        """
        if pairs.numel() == 0:
            return torch.zeros(0, 128, device=boxes_xywh.device)

        feats = []
        for (si, oi) in pairs:
            b1 = boxes_xywh[si]
            b2 = boxes_xywh[oi]
            feats.append(self._pair_feats(b1, b2))
        feats = torch.stack(feats, dim=0).to(boxes_xywh.device)
        return self.mlp(feats)


class RelationDecoder(nn.Module):
    """
    Classifies predicates for subject-object pairs using contextualized object embeddings
    and pair geometry features.
    """

    def __init__(self, obj_dim: int, pair_geom_dim: int = 128, num_predicates: int = 50, hidden_dim: int = 512):
        super().__init__()
        self.geom = PairGeometry(out_dim=pair_geom_dim)
        in_dim = obj_dim * 2 + pair_geom_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_predicates)
        )

    def forward(self, obj_embeds: torch.Tensor, boxes_xywh: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
        """
        obj_embeds: [K, D]
        boxes_xywh: [K, 4]
        pairs: [P, 2]
        returns: logits [P, num_predicates]
        """
        if obj_embeds.size(0) == 0 or pairs.numel() == 0:
            return torch.zeros(0, self.mlp[-1].out_features, device=obj_embeds.device)

        geom = self.geom(boxes_xywh, pairs)
        subj = obj_embeds[pairs[:, 0]]
        obj = obj_embeds[pairs[:, 1]]
        x = torch.cat([subj, obj, geom], dim=-1)
        return self.mlp(x)
