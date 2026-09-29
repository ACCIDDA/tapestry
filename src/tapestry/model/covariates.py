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
            slope = (centered * x).sum(-1) / (centered.square() * mask).sum(-1).clamp_min(1e-8)
            recent = x[..., -3:].sum(-1) / mask[..., -3:].sum(-1).clamp_min(1)
            features = torch.stack((last, recent, slope, sd), -1)
        return torch.cat((features, coverage[..., None], age[..., None]), -1).flatten(-2)


def multiscale_features(values, available, smooth=False):
    """Causal levels, linear trends and quadratic curvature over 3/6/12 weeks.

    Polynomial coefficients use actual calendar positions, never compressed gaps.
    Fewer than 2/3 reports suppress slope/curvature. Coverage and validity accompany
    every window. Features supplement the original history rather than replacing it.
    """
    if smooth:
        values, available = trailing_mean(values, available, 3)
    x = torch.where(available, values, 0).permute(0, 3, 2, 1)
    mask = available.permute(0, 3, 2, 1)
    features = []
    for window in (3, 6, 12):
        width = min(window, x.shape[-1])
        y, keep = x[..., -width:], mask[..., -width:]
        t = torch.arange(1 - width, 1, device=x.device, dtype=x.dtype)
        count = keep.sum(-1)
        mean = y.sum(-1) / count.clamp_min(1)
        time_mean = (keep * t).sum(-1) / count.clamp_min(1)
        centered = t - time_mean[..., None]
        slope = (centered * y).sum(-1) / (centered.square() * keep).sum(-1).clamp_min(1e-8)
        # Residualize squared time against intercept/time: the quadratic least-
        # squares coefficient without a per-window matrix decomposition.
        u = t / max(width - 1, 1)
        u_mean = (keep * u).sum(-1) / count.clamp_min(1)
        centered_u = u - u_mean[..., None]
        square = u.square()
        square_mean = (keep * square).sum(-1) / count.clamp_min(1)
        projection = (keep * centered_u * square).sum(-1) / (keep * centered_u.square()).sum(-1).clamp_min(1e-8)
        residual = (square - square_mean[..., None] - projection[..., None] * centered_u) * keep
        coefficient = (residual * y).sum(-1) / residual.square().sum(-1).clamp_min(1e-10)
        curvature = 2 * coefficient / max(width - 1, 1) ** 2
        features.extend((mean, torch.where(count >= 2, slope, 0),
                         torch.where(count >= 3, curvature, 0), count / width,
                         (count >= 2).to(x.dtype), (count >= 3).to(x.dtype)))
    return torch.stack(features, -1).flatten(-2)
