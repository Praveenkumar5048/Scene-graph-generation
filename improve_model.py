import torch
import torch.nn as nn


def initialize_sgg_model_weights(model: nn.Module):
    """Apply Xavier/kaiming initialization to Linear layers to stabilize first runs."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def enhance_scene_graph_with_rules(scene_graph: dict, class_ids: torch.Tensor, boxes_xywh: torch.Tensor) -> dict:
    """
    Optional, tiny post-processing: boost relation scores for close-by pairs.
    """
    rels = scene_graph.get("triplets", [])
    if not rels:
        return scene_graph

    # Example: if IoU > 0.3, slightly boost the predicate score
    def iou_xywh(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        X1, Y1, X2, Y2 = x1, y1, x1 + w1, y1 + h1
        X3, Y3, X4, Y4 = x2, y2, x2 + w2, y2 + h2
        iw = max(0.0, min(X2, X4) - max(X1, X3))
        ih = max(0.0, min(Y2, Y4) - max(Y1, Y3))
        inter = iw * ih
        union = w1 * h1 + w2 * h2 - inter + 1e-6
        return inter / union

    for t in rels:
        s, o = t["subject"], t["object"]
        iou = iou_xywh(boxes_xywh[s].tolist(), boxes_xywh[o].tolist())
        if iou > 0.3:
            t["score"] = float(min(1.0, t["score"] + 0.05))

    return scene_graph
