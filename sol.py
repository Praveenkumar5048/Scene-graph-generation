import torch


def subject_object_localization(scores: torch.Tensor, boxes: torch.Tensor, topk: int = 50):
    """
    Simple SOL: selects top-k objects by score.
    scores: [K] detection scores
    boxes: [K, 4]
    returns indices [M] (M<=K)
    """
    if scores.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    k = min(topk, scores.numel())
    idx = torch.topk(scores, k=k).indices
    return idx
