"""Temporal and stochastic output modules for the B0.1 architecture crosses."""
import torch
from torch import nn
import torch.nn.functional as F


class MultiscaleEncoder(nn.Module):
    """Two width-three convolutions at each dilation; recent and broad pooling."""
    def __init__(self, inputs, width):
        super().__init__()
        sizes = [width // 3, width // 3, width - 2 * (width // 3)]
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Conv1d(inputs, size, 3, padding=d, dilation=d), nn.SiLU(),
                          nn.Conv1d(size, size, 3, padding=d, dilation=d), nn.SiLU())
            for size, d in zip(sizes, (1, 2, 4))])
        self.project = nn.Linear(2 * width, width)

    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        return self.project(torch.cat([v for h in features for v in (h.mean(-1), h[..., -1])], -1))


class ForecastHead(nn.Module):
    """Identical per-group designs; each group learns its own readout and modulation."""
    def __init__(self, width, latent, local, kind):
        super().__init__()
        from .b0 import ModulatedDecoder, modulation_layer, softplus_inverse
        self.kind = kind
        if kind == 'residual2':
            self.output = ModulatedDecoder(width, latent, local)
        elif kind == 'legacy':
            self.modulate = modulation_layer(latent, width)
            self.output = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
            nn.init.normal_(self.output[-1].weight, std=.01)
            nn.init.zeros_(self.output[-1].bias)
            if local:
                self.local_modulate = modulation_layer(local, width)
                self.local_scale = nn.Parameter(softplus_inverse(1))
        else:
            raise ValueError(f'Unknown forecast head: {kind}')

    def forward(self, h, z, local_z):
        if self.kind == 'residual2':
            return self.output(h, z, local_z)
        if self.kind == 'legacy':
            from .b0 import affine
            local = getattr(self, 'local_modulate', None)
            scale = F.softplus(self.local_scale) if local is not None else None
            gamma, beta = affine(self.modulate, z, local, local_z, scale)
            return self.output(h[None] * (1 + gamma) + beta)
        raise ValueError(f'Unknown forecast head: {self.kind}')
