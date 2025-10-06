import torch


def dynamic_triplet_mining(num_objects: int, max_pairs: int | None = 2000) -> torch.Tensor:
    """
    Generate candidate (subject, object) index pairs excluding self-pairs.
    Optionally cap to max_pairs.
    returns [P, 2]
    """
    if num_objects <= 1:
        return torch.zeros(0, 2, dtype=torch.long)
    grid_s, grid_o = torch.meshgrid(torch.arange(num_objects), torch.arange(num_objects), indexing="ij")
    mask = grid_s != grid_o
    pairs = torch.stack([grid_s[mask], grid_o[mask]], dim=-1)
    if max_pairs is not None and pairs.size(0) > max_pairs:
        pairs = pairs[:max_pairs]
    return pairs
