import torch
import torch.nn as nn
import math

# --- Pairwise Spatial Encoder ---
class PairSpatialEncoder(nn.Module):
    def __init__(self, in_dim=13, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, boxes):
        # Vectorized pairwise feature computation.
        # boxes: [K, 4] expected as [x_center, y_center, w, h] (normalized or absolute)
        K = boxes.size(0)
        if K <= 1:
            return self.mlp(torch.zeros((0, self.mlp[0].in_features), device=boxes.device))

        eps = 1e-6
        x = boxes[:, 0].unsqueeze(1)  # [K,1]
        y = boxes[:, 1].unsqueeze(1)
        w = boxes[:, 2].unsqueeze(1)
        h = boxes[:, 3].unsqueeze(1)

        # Broadcast to pairwise matrices [K, K]
        dx = (x.transpose(0, 1) - x) / (w + eps)  # xj - xi over wi
        dy = (y.transpose(0, 1) - y) / (h + eps)
        dw = torch.log((w.transpose(0, 1) + eps) / (w + eps))
        dh = torch.log((h.transpose(0, 1) + eps) / (h + eps))

        # areas
        area_i = (w * h)
        area_j = (w.transpose(0, 1) * h.transpose(0, 1))

        # coordinates for IoU: convert center to x1,y1,x2,y2
        x1 = (x - 0.5 * w)
        y1 = (y - 0.5 * h)
        x2 = (x + 0.5 * w)
        y2 = (y + 0.5 * h)

        x1_t = x1.transpose(0, 1)
        y1_t = y1.transpose(0, 1)
        x2_t = x2.transpose(0, 1)
        y2_t = y2.transpose(0, 1)

        inter_x1 = torch.max(x1, x1_t)
        inter_y1 = torch.max(y1, y1_t)
        inter_x2 = torch.min(x2, x2_t)
        inter_y2 = torch.min(y2, y2_t)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter_area = inter_w * inter_h
        union_area = (area_i + area_j - inter_area).clamp(min=eps)
        iou = inter_area / union_area

        # flatten pairs excluding diagonal
        idx_i, idx_j = torch.where(~torch.eye(K, dtype=torch.bool, device=boxes.device))
        feats = torch.stack([
            dx[idx_i, idx_j], dy[idx_i, idx_j], dw[idx_i, idx_j], dh[idx_i, idx_j],
            w[idx_i, 0], h[idx_i, 0], w[idx_j, 0], h[idx_j, 0], iou[idx_i, idx_j],
            x[idx_i, 0], y[idx_i, 0], x[idx_j, 0], y[idx_j, 0]
        ], dim=1)

        return self.mlp(feats)


def np_log_safe(x):
    # small helper in module scope to avoid import issues
    import math
    try:
        return math.log(x + 1e-6)
    except Exception:
        return 0.0

# --- ART Layer ---
class ARTLayer(nn.Module):
    def __init__(self, input_dim=1328, hidden_dim=512, pair_dim=128):
        super().__init__()
        self.att_mlp = nn.Linear(input_dim * 2 + pair_dim, 1)
        self.obj_proj = nn.Linear(input_dim, hidden_dim)
        self.pair_proj = nn.Linear(pair_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x, pair_feats, mask=None):
        K = x.size(0)
        messages = []
        pair_messages = []
        pair_idx = 0
        for i in range(K):
            msgs = []
            for j in range(K):
                if i == j: continue
                # Apply mask provided (Stage 2 filtering)
                if mask is not None and not mask[i, j]:
                    pair_idx += 1
                    continue

                pair_feat = pair_feats[pair_idx]
                pair_idx += 1
                # attention input: x[i], x[j], pair_feat
                att_input = torch.cat([x[i], x[j], pair_feat], dim=-1)
                alpha = torch.sigmoid(self.att_mlp(att_input))
                m = alpha * (self.obj_proj(x[j]) + self.pair_proj(pair_feat))
                msgs.append(m)
                pair_messages.append(m)
            if len(msgs) > 0:
                msgs = torch.stack(msgs).mean(dim=0)
            else:
                msgs = torch.zeros_like(self.obj_proj(x[i]))  # Fix: match output dim
            messages.append(msgs)
        messages = torch.stack(messages)
        # project input objects to hidden dim before adding pairwise messages
        x_proj = self.obj_proj(x)
        out = self.norm(x_proj + messages)
        out = self.norm(out + self.ffn(out))
        if len(pair_messages) > 0:
            pair_messages = torch.stack(pair_messages)
        else:
            pair_messages = torch.zeros((0, out.size(-1)), device=out.device)
        return out, pair_messages

# --- ART Encoder (2 stages) ---
class ARTEncoder(nn.Module):
    def __init__(self, input_dim=1328, hidden_dim=512, pair_dim=128, max_pair_objects: int = None):
        super().__init__()
        self.pair_enc = PairSpatialEncoder()
        self.stage1 = ARTLayer(input_dim, hidden_dim, pair_dim)
        self.stage2 = ARTLayer(hidden_dim, hidden_dim, pair_dim)
        # max_pair_objects: if set, sample top-k objects (by area) and only build pairs among them
        self.max_pair_objects = max_pair_objects

    def forward(self, x, boxes, mask=None):
        # optionally sample top objects to reduce pairwise cost
        if self.max_pair_objects is not None and boxes.size(0) > self.max_pair_objects:
            # sample by area
            areas = boxes[:, 2] * boxes[:, 3]
            _, topk_idx = torch.topk(areas, self.max_pair_objects)
            boxes_sampled = boxes[topk_idx]
            # compute pair_feats for sampled boxes and then map back (this is a simple strategy)
            pair_feats = self.pair_enc(boxes_sampled)
            # NOTE: downstream pair indices will correspond to sampled pairs only
        else:
            pair_feats = self.pair_enc(boxes)  # [M, pair_dim]

        h1, pair_msgs1 = self.stage1(x, pair_feats)
        h2, pair_msgs2 = self.stage2(h1, pair_feats, mask=mask)
        # return final per-proposal features, spatial pair features and pairwise messages
        return h2, pair_feats, pair_msgs2
