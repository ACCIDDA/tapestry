"""Build B0: shared focal/context MLP and globally modulated sample decoder."""
import torch
from torch import nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """Two small convolutions shared over weeks; pool mean and latest features.

    All timesteps are observed context. Symmetric padding never sees forecast labels.
    Value/mask pairs let the filters distinguish missing weeks from observed zeros.
    """
    def __init__(self, inputs, width):
        super().__init__()
        self.filters = nn.Sequential(nn.Conv1d(inputs, width, 3, padding=1), nn.SiLU(),
                                     nn.Conv1d(width, width, 3, padding=1), nn.SiLU())
        self.project = nn.Linear(2 * width, width)

    def forward(self, x):
        h = self.filters(x)
        return self.project(torch.cat((h.mean(-1), h[..., -1]), -1))


class ModulatedDecoder(nn.Module):
    """Two residual blocks, each modulated by the same episode-level latent."""
    def __init__(self, width, latent):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.modulations = nn.ModuleList([nn.Linear(latent, 2 * width) for _ in range(2)])
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                                  nn.Linear(width, width)) for _ in range(2)])
        self.output = nn.Linear(width, 1)
        for layer in self.modulations:
            nn.init.normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, h, z):
        hidden = h[None]
        for norm, modulation, block in zip(self.norms, self.modulations, self.blocks):
            gamma, beta = modulation(z).chunk(2, -1)
            perturbed = norm(hidden) * (1 + gamma[:, :, None, None, None]) + beta[:, :, None, None, None]
            hidden = hidden + block(perturbed)
        return self.output(hidden)


def fair_crps(samples, truth, mask):
    """Return one masked score per channel; inputs [M,N,H,C,L], [N,H,C,L].

    Mask BEFORE arithmetic, including pairwise terms. Missing labels contribute
    neither score nor gradient. Members must be independent draws from one model.
    """
    m = samples.shape[0]
    if m < 2:
        raise ValueError('Fair CRPS needs at least two independent members')
    samples = torch.where(mask.bool().unsqueeze(0), samples, 0)
    truth = torch.where(mask.bool(), truth, 0)
    # Sorting avoids an M x M pairwise tensor; this equals the off-diagonal sum.
    # Flatten task axes: Apple's MPS sort cannot handle this rank-5 array.
    ordered = samples.reshape(m, -1).sort(dim=0).values.reshape_as(samples)
    weights = (2 * torch.arange(m, device=samples.device) - m + 1).to(samples.dtype)
    spread = (ordered * weights.reshape(m, 1, 1, 1, 1)).sum(0) / (m * (m - 1))
    score = (samples - truth).abs().mean(0) - spread
    return (score * mask).sum((0, 1, 3)) / mask.sum((0, 1, 3)).clamp_min(1)


class B0(nn.Module):
    def __init__(self, lookback=8, horizons=(1, 2, 3, 4), width=64, latent=16, scale=None,
                 count_transform="raw", populations=None, geography=False, dynamics=False, input_scale=None,
                 encoder='mlp', heads='shared', decoder='legacy'):
        super().__init__()
        self.config = dict(lookback=lookback, horizons=list(horizons), width=width, latent=latent)
        if encoder not in ('mlp', 'conv') or heads not in ('shared', 'state_us') or decoder not in ('legacy', 'residual2'):
            raise ValueError('Unknown B0 architecture option')
        if min(lookback, width, latent) < 1:
            raise ValueError('B0 dimensions must be positive')
        if count_transform not in ('raw', 'sqrt', 'fourth_root'):
            raise ValueError('Unknown count transform')
        if (count_transform != 'raw' or geography) and not populations:
            raise ValueError('Population metadata required')
        if populations and any(not torch.isfinite(torch.tensor(float(v))) or float(v) <= 0 for v in populations.values()):
            raise ValueError('Populations must be finite and positive')
        self.config.update(count_transform=count_transform, populations=populations,
                           geography=geography, dynamics=dynamics, encoder=encoder, heads=heads, decoder=decoder)
        # Separate transformed input scales from native-unit loss normalization.
        # No extra buffer in legacy mode, so original checkpoints still load.
        if count_transform != 'raw':
            self.register_buffer('input_scale', torch.ones(6) if input_scale is None else torch.as_tensor(input_scale).float())
        self.register_buffer('scale', torch.ones(6) if scale is None else torch.as_tensor(scale).float())
        self.context = nn.Sequential(nn.Linear((lookback * 6 * 2 if encoder == 'mlp' else width) + 2 + 2 * geography + 31 * dynamics, width), nn.SiLU(), nn.Linear(width, width))
        self.focal = (nn.Sequential(nn.Linear(lookback * 2, width), nn.SiLU(), nn.Linear(width, width))
                      if encoder == 'mlp' else TemporalEncoder(2, width))
        if encoder == 'conv':
            self.temporal_context = TemporalEncoder(12, width)
        self.source = nn.Embedding(6, width)
        self.horizon = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)
        if decoder == 'legacy':
            # Preserve module names and initialization order for old checkpoints/seeds.
            self.modulate = nn.Linear(latent, 2 * width)
            self.decoder = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
            nn.init.normal_(self.modulate.weight, std=.02)
            nn.init.zeros_(self.modulate.bias)
            nn.init.normal_(self.decoder[-1].weight, std=.01)
            nn.init.zeros_(self.decoder[-1].bias)
        else:
            self.decoder = ModulatedDecoder(width, latent)
        if heads == 'state_us':
            # Identical initial heads isolate specialization during fitting. Modulation
            # parameters then learn separately, while random draws remain shared.
            import copy
            self.us_decoder = copy.deepcopy(self.decoder)
            if decoder == 'legacy':
                self.us_modulate = copy.deepcopy(self.modulate)

    def forward(self, x, calendar, members=8, z=None, locations=None):
        """x [N,P,C,2,L], calendar [N,2]; samples [M,N,H,C,L].

        Each member uses one latent per episode shared across ALL locations,
        horizons and channels. No independent observation noise is appended.
        """
        n, p, c, _, l = x.shape
        if self.config['heads'] == 'state_us' and (locations is None or len(locations) != l):
            raise ValueError('Separate heads require ordered location IDs')
        mask = x[:, :, :, 1, :]
        raw = torch.where(mask.bool(), x[:, :, :, 0, :], 0)
        population = None
        if self.config['count_transform'] != 'raw' or self.config['geography']:
            if locations is None or len(locations) != l:
                raise ValueError('Provide ordered location IDs for population metadata')
            population = x.new_tensor([self.config['populations'][loc] for loc in locations])
        power = {'raw': 1., 'sqrt': .5, 'fourth_root': .25}[self.config['count_transform']]
        if power != 1:
            transformed = torch.cat(((raw[:, :, :3].clamp_min(0) * (100000 / population)).pow(power), raw[:, :, 3:]), 2)
            input_scale = self.input_scale
        else:
            transformed, input_scale = raw, self.scale
        values = transformed / input_scale[None, None, :, None]
        fields = torch.stack((values, mask), dim=-1)  # N,P,C,L,2
        context = fields.permute(0, 3, 1, 2, 4).reshape(n, l, -1)
        if self.config['encoder'] == 'conv':
            context = self.temporal_context(fields.permute(0, 3, 2, 4, 1).reshape(n * l, c * 2, p)).reshape(n, l, -1)
        extras = [calendar[:, None, :2].expand(-1, l, -1)]
        if self.config['geography']:
            # Log population centered at 100,000; native US remains a separate task.
            geo = torch.stack(((population / 100000).log(), x.new_tensor([loc == 'US' for loc in locations])), -1)
            extras.append(geo[None].expand(n, -1, -1))
        if self.config['dynamics']:
            if calendar.shape[-1] != 3:
                raise ValueError('Dynamics requires Christmas calendar feature')
            extras.extend((recent_dynamics(values, mask).permute(0, 2, 1),
                           calendar[:, None, 2:].expand(-1, l, -1)))
        context = self.context(torch.cat([context, *extras], -1))
        if self.config['encoder'] == 'mlp':
            focal = self.focal(fields.permute(0, 3, 2, 1, 4).reshape(n, l, c, p * 2))
        else:
            focal = self.focal(fields.permute(0, 3, 2, 4, 1).reshape(n * l * c, 2, p)).reshape(n, l, c, -1)
        h = context[:, :, None, :] + focal + self.source.weight[None, None, :, :]
        offsets = x.new_tensor(self.config['horizons']).reshape(-1, 1) / 4
        h = self.norm(h[:, None, :, :, :] + self.horizon(offsets)[None, :, None, None, :])
        if z is None:
            z = torch.randn(members, n, self.config['latent'], device=x.device)
        def decode(head, modulation=None):
            if self.config['decoder'] == 'residual2':
                return head(h, z)
            gamma, beta = modulation(z).chunk(2, dim=-1)
            hidden = h[None] * (1 + gamma[:, :, None, None, None, :]) + beta[:, :, None, None, None, :]
            return head(hidden)
        delta = decode(self.decoder, getattr(self, 'modulate', None))
        if self.config['heads'] == 'state_us':
            us = decode(self.us_decoder, getattr(self, 'us_modulate', None))
            is_us = torch.tensor([loc == 'US' for loc in locations], device=x.device)
            delta = torch.where(is_us[None, None, None, :, None, None], us, delta)
        delta = delta.squeeze(-1).permute(0, 1, 2, 4, 3)
        # Last valid focal value anchors the residual; zero-history uses a small prior.
        idx = (mask * torch.arange(1, p + 1, device=x.device)[None, :, None, None]).argmax(1)
        anchor = values.gather(1, idx[:, None]).squeeze(1)
        anchor = torch.where(mask.any(1), anchor, anchor.new_full((), .01))
        positive = anchor.clamp_min(.001)
        count_base = positive + torch.log(-torch.expm1(-positive))  # inverse softplus
        counts = F.softplus(count_base[None, :, None, :3] + delta[:, :, :, :3]) * input_scale[None, None, None, :3, None]
        if power != 1:
            counts = counts.pow(1 / power) * (population / 100000)
        proportion = (anchor[:, 3:] * input_scale[None, 3:, None]).clamp(.0001, .9999)
        ed = torch.sigmoid(torch.logit(proportion)[None, :, None] + delta[:, :, :, 3:])
        return torch.cat((counts, ed), dim=3)


def recent_dynamics(values, mask):
    """Causal last two weekly slopes, acceleration and age, with validity flags.

    Slopes use the last three calendar weeks only (no interpolation over gaps).
    Age is weeks since last observed input / lookback; no-history age is one.
    """
    n, p, c, l = values.shape
    zero = values.new_zeros(n, c, l)
    valid = mask[:, -1] * mask[:, -2] if p >= 2 else zero
    prior_valid = mask[:, -2] * mask[:, -3] if p >= 3 else zero
    slope = (values[:, -1] - values[:, -2]) * valid if p >= 2 else zero
    prior = (values[:, -2] - values[:, -3]) * prior_valid if p >= 3 else zero
    acceleration_valid = valid * prior_valid
    acceleration = (slope - prior) * acceleration_valid
    last = (mask * torch.arange(1, p + 1, device=values.device)[None, :, None, None]).amax(1)
    age = (p - last) / p
    return torch.stack((slope, acceleration, age, valid, acceleration_valid), 2).reshape(n, c * 5, l)
