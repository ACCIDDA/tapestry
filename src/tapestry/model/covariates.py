"""Small covariate representations after fitting-fold standardization.

Inputs have [episode, week, covariate, location] axes. Missing values never
contribute. Non-raw representations use signed log1p of standardized values.
"""
import torch
from torch import nn
import torch.nn.functional as F


def signed_log(values):
    return values.sign() * values.abs().log1p()


def trailing_mean(values, available, window=3):
    """Causal mean over available reports in the trailing window, including today."""
    x = torch.where(available, values, 0).permute(0, 2, 3, 1)
    mask = available.to(values.dtype).permute(0, 2, 3, 1)
    shape = x.shape
    total = F.avg_pool1d(F.pad(x.reshape(-1, 1, shape[-1]), (window - 1, 0)), window, stride=1) * window
    count = F.avg_pool1d(F.pad(mask.reshape(-1, 1, shape[-1]), (window - 1, 0)), window, stride=1) * window
    return (total / count.clamp_min(1)).reshape(shape).permute(0, 3, 1, 2), (count > 0).reshape(shape).permute(0, 3, 1, 2)


class CovariateEncoder(nn.Module):
    """Summary or shared four-dimensional learned encoding, retaining source identity."""
    def __init__(self, lookback, kind):
        super().__init__()
        self.kind = kind
        if kind == 'shared':
            self.encoder = nn.Sequential(nn.Linear(2 * lookback, 8), nn.SiLU(), nn.Linear(8, 4))
        elif kind != 'summary':
            raise ValueError(f'Unknown compact covariate representation: {kind}')

    def forward(self, values, available):
        # Each source/location is encoded independently with shared weights.
        x = torch.where(available, values, 0).permute(0, 3, 2, 1)
        mask = available.permute(0, 3, 2, 1)
        p = x.shape[-1]
        count = mask.sum(-1)
        t = torch.arange(p, device=x.device, dtype=x.dtype)
        last_index = torch.where(mask, t, -1).amax(-1)
        age = (p - 1 - last_index) / p  # 1 when no report exists
        coverage = count / p
        if self.kind == 'shared':
            features = self.encoder(torch.cat((x, mask.to(x.dtype)), -1))
            features = features * (count > 0)[..., None]
        else:
            last = x.gather(-1, last_index.clamp_min(0).long()[..., None]).squeeze(-1)
            mean = x.sum(-1) / count.clamp_min(1)
            sd = (torch.where(mask, (x - mean[..., None]).square(), 0).sum(-1) / count.clamp_min(1)).sqrt()
            time_mean = (mask * t).sum(-1) / count.clamp_min(1)
            centered = t - time_mean[..., None]
            slope = (centered * x).sum(-1) / (centered.square() * mask).sum(-1).clamp_min(1)
            recent = x[..., -3:].sum(-1) / mask[..., -3:].sum(-1).clamp_min(1)
            features = torch.stack((last, recent, slope, sd), -1)
        return torch.cat((features, coverage[..., None], age[..., None]), -1).flatten(-2)
