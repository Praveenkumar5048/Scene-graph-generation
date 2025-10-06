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
        
        # Handle case with single object (no pairs possible)
        if K <= 1:
            # Return empty tensor with correct shape
            device = boxes.device
            return torch.empty(0, 128, device=device, dtype=torch.float32)
        
        feats = []
        for i in range(K):
            for j in range(K):
                if i == j: continue
                xi, yi, wi, hi = boxes[i]
                xj, yj, wj, hj = boxes[j]
                dx, dy = (xj - xi) / (wi + 1e-6), (yj - yi) / (hi + 1e-6)
                dw, dh = torch.log(wj / wi + 1e-6), torch.log(hj / hi + 1e-6)
                
                # Proper IoU calculation (tensor-safe)
                x1i, y1i, x2i, y2i = xi, yi, xi + wi, yi + hi
                x1j, y1j, x2j, y2j = xj, yj, xj + wj, yj + hj
                inter_w = torch.clamp(torch.minimum(x2i, x2j) - torch.maximum(x1i, x1j), min=0)
                inter_h = torch.clamp(torch.minimum(y2i, y2j) - torch.maximum(y1i, y1j), min=0)
                inter_area = inter_w * inter_h
                union_area = wi * hi + wj * hj - inter_area
                iou = inter_area / (union_area + 1e-6)
                
                feats.append([dx, dy, dw, dh, wi, hi, wj, hj, iou, xi, yi, xj, yj])
        
        if len(feats) == 0:
            # Return empty tensor if no features
            device = boxes.device
            return torch.empty(0, 128, device=device, dtype=torch.float32)
            
        feats = torch.tensor(feats, dtype=torch.float32, device=boxes.device)
        return self.mlp(feats)

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
        
        # Handle empty pair features
        if pair_feats.size(0) == 0:
            return x
            
        messages = []
        pair_idx = 0
        for i in range(K):
            msgs = []
            for j in range(K):
                if i == j: continue
                
                # Apply mask if provided (Stage 2 filtering)
                if mask is not None and not mask[i, j]:
                    pair_idx += 1
                    continue
                
                # Check if we have enough pair features
                if pair_idx >= pair_feats.size(0):
                    break
                    
                pair_feat = pair_feats[pair_idx]
                pair_idx += 1
                att_input = torch.cat([x[i], x[j], pair_feat], dim=-1)
                alpha = torch.sigmoid(self.att_mlp(att_input))
                m = alpha * (self.obj_proj(x[j]) + self.pair_proj(pair_feat))
                msgs.append(m)
            if len(msgs) > 0:
                msgs = torch.stack(msgs).mean(dim=0)
            else:
                msgs = torch.zeros_like(self.obj_proj(x[i]))  # Fix: match output dim
            messages.append(msgs)
        messages = torch.stack(messages)
        # Project x to hidden dim for residual addition
        x_proj = self.obj_proj(x)
        out = self.norm(x_proj + messages)
        out = self.norm(out + self.ffn(out))
        return out

# --- ART Encoder (2 stages) ---
class ARTEncoder(nn.Module):
    def __init__(self, input_dim=1329, hidden_dim=512, pair_dim=128):
        super().__init__()
        self.pair_enc = PairSpatialEncoder()
        self.stage1 = ARTLayer(input_dim, hidden_dim, pair_dim)
        self.stage2 = ARTLayer(hidden_dim, hidden_dim, pair_dim)

    def forward(self, x, boxes, mask=None):
        K = x.size(0)
        
        # Handle case with single object
        if K <= 1:
            # No pairs to process, return input unchanged
            return x
            
        pair_feats = self.pair_enc(boxes)
        
        # Handle empty pair features
        if pair_feats.size(0) == 0:
            return x
            
        h1 = self.stage1(x, pair_feats)
        h2 = self.stage2(h1, pair_feats, mask=mask)
        return h2   # [K, hidden_dim]
