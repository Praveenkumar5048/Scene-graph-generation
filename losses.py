import torch
import torch.nn as nn


class SGGLoss(nn.Module):
    """Combined loss for object classification and relation predicate classification.
    This is a placeholder; for training, wire real targets from dataset.
    """

    def __init__(self, obj_weight: float = 1.0, rel_weight: float = 1.0):
        super().__init__()
        self.obj_weight = obj_weight
        self.rel_weight = rel_weight
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, obj_logits: torch.Tensor, obj_targets: torch.Tensor,
                rel_logits: torch.Tensor, rel_targets: torch.Tensor):
        losses = {}
        if obj_logits.numel() and obj_targets.numel():
            losses["loss_obj"] = self.ce(obj_logits, obj_targets) * self.obj_weight
        else:
            losses["loss_obj"] = torch.tensor(0.0, device=obj_logits.device)

        if rel_logits.numel() and rel_targets.numel():
            losses["loss_rel"] = self.ce(rel_logits, rel_targets) * self.rel_weight
        else:
            losses["loss_rel"] = torch.tensor(0.0, device=obj_logits.device)

        losses["loss_total"] = losses["loss_obj"] + losses["loss_rel"]
        return losses
