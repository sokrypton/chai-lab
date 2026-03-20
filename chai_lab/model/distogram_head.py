from pathlib import Path

import torch
from torch import nn, Tensor


class DistogramHead(nn.Module):
    def __init__(self, pair_dim: int, n_dist_bins: int):
        super().__init__()
        self.m = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, 2 * pair_dim),
            nn.GELU(),
            nn.Linear(2 * pair_dim, n_dist_bins),
        )

    def compute_disto_logits(self, pair_emb: Tensor) -> Tensor:
        return self.m(pair_emb)

    @classmethod
    def load(cls, path: Path, device: torch.device) -> "DistogramHead":
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        pair_dim = state_dict["m.0.weight"].shape[0]
        n_dist_bins = state_dict["m.3.weight"].shape[0]
        model = cls(pair_dim=pair_dim, n_dist_bins=n_dist_bins)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
