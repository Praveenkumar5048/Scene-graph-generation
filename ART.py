import torch
import torch.nn as nn

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
        K = boxes.size(0)
        feats = []
        # boxes expected in [x, y, w, h] or [x1,y1,x2,y2] depending on caller; here we assume [x,y,w,h]
        for i in range(K):
            xi = float(boxes[i, 0].item())
            yi = float(boxes[i, 1].item())
            wi = float(boxes[i, 2].item())
            hi = float(boxes[i, 3].item())
            for j in range(K):
                if i == j:
                    continue
                xj = float(boxes[j, 0].item())
                yj = float(boxes[j, 1].item())
                wj = float(boxes[j, 2].item())
                hj = float(boxes[j, 3].item())

                # relative deltas
                dx = (xj - xi) / (wi + 1e-6)
                dy = (yj - yi) / (hi + 1e-6)
                dw = float(np_log_safe(wj / (wi + 1e-6)))
                dh = float(np_log_safe(hj / (hi + 1e-6)))

                # IoU computed with clamped overlap
                x1i, y1i, x2i, y2i = xi, yi, xi + wi, yi + hi
                x1j, y1j, x2j, y2j = xj, yj, xj + wj, yj + hj
                inter_w = max(0.0, min(x2i, x2j) - max(x1i, x1j))
                inter_h = max(0.0, min(y2i, y2j) - max(y1i, y1j))
                inter_area = inter_w * inter_h
                union_area = wi * hi + wj * hj - inter_area
                iou = inter_area / (union_area + 1e-6)

                feats.append([dx, dy, dw, dh, wi, hi, wj, hj, iou, xi, yi, xj, yj])
        feats = torch.tensor(feats, dtype=torch.float32, device=boxes.device)
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
    def __init__(self, input_dim=1329, hidden_dim=512, pair_dim=128):
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
        out = self.norm(x + messages)
        out = self.norm(out + self.ffn(out))
        if len(pair_messages) > 0:
            pair_messages = torch.stack(pair_messages)
        else:
            pair_messages = torch.zeros((0, out.size(-1)), device=out.device)
        return out, pair_messages

# --- ART Encoder (2 stages) ---
class ARTEncoder(nn.Module):
    def __init__(self, input_dim=1329, hidden_dim=512, pair_dim=128):
        super().__init__()
        self.pair_enc = PairSpatialEncoder()
        self.stage1 = ARTLayer(input_dim, hidden_dim, pair_dim)
        self.stage2 = ARTLayer(hidden_dim, hidden_dim, pair_dim)

    def forward(self, x, boxes, mask=None):
        pair_feats = self.pair_enc(boxes)  # [M, pair_dim]
        h1, pair_msgs1 = self.stage1(x, pair_feats)
        h2, pair_msgs2 = self.stage2(h1, pair_feats, mask=mask)
        # return final per-proposal features, spatial pair features and pairwise messages
        return h2, pair_feats, pair_msgs2
