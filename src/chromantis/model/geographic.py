"""Census 2024 state internal points; geographic proxies, not observed travel flows."""
import json
from pathlib import Path
import torch
from torch import nn

COORDINATES = json.loads(Path(__file__).with_name('state_coordinates.json').read_text())

class GeographicMessage(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        self.project = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(2 * width, width) if kind == 'gravity' else None

    def forward(self, context, observed, locations, populations):
        points = context.new_tensor([COORDINATES.get(loc, [0., 0.]) for loc in locations]) * (torch.pi / 180)
        lat, lon = points[:, 0], points[:, 1]
        a = ((lat[:, None] - lat[None]) / 2).sin().square() + lat[:, None].cos() * lat[None].cos() * ((lon[:, None] - lon[None]) / 2).sin().square()
        distance = 6371 * 2 * a.clamp(0, 1).sqrt().asin()
        allowed = context.new_tensor([loc != 'US' for loc in locations], dtype=torch.bool)
        eligible = allowed[:, None] & allowed[None] & ~torch.eye(len(locations), device=context.device, dtype=torch.bool)
        if self.kind == 'neighbors':
            nearest = distance.masked_fill(~eligible, float('inf')).topk(min(4, len(locations)), largest=False).indices
            weights = torch.zeros_like(distance).scatter_(1, nearest, 1) * eligible
        else:
            weights = torch.exp(-distance / 750) * eligible
            if self.kind == 'gravity':
                population = context.new_tensor([populations[loc] for loc in locations])
                weights = weights * population[None].sqrt()
        weights = weights[None] * observed[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        message = self.project(torch.bmm(weights, context))
        if self.gate is not None:
            message = message * torch.sigmoid(self.gate(torch.cat((context, message), -1)))
        return context + message
