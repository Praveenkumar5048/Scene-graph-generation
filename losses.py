import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Multi-class Focal Loss (for logits input).

    Args:
      gamma: focusing parameter
      alpha: balance factor (float or list/ndarray of shape [C])
      reduction: 'mean' | 'sum' | 'none'

    Usage:
      loss = FocalLoss(gamma=2.0, alpha=0.25)
      l = loss(logits, target)  # target: [N] class indices
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', eps=1e-9):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        if alpha is None:
            self.alpha = None
        else:
            if isinstance(alpha, (float, int)):
                self.alpha = float(alpha)
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, logits, targets):
        # logits: [N, C], targets: [N]
        logpt = F.log_softmax(logits, dim=1)
        pt = torch.exp(logpt)
        # gather the log probability for true class
        targets = targets.long()
        logp = logpt.gather(1, targets.unsqueeze(1)).squeeze(1)
        p = pt.gather(1, targets.unsqueeze(1)).squeeze(1)

        # focal term
        loss = -((1 - p) ** self.gamma) * logp

        # alpha balancing
        if self.alpha is not None:
            if isinstance(self.alpha, float):
                at = self.alpha
                loss = at * loss
            else:
                # class-wise alpha
                if self.alpha.device != logits.device:
                    self.alpha = self.alpha.to(logits.device)
                at = self.alpha.gather(0, targets)
                loss = at * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
