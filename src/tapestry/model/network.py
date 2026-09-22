"""The one network: today's B0 forward pass, plus B1-direct's thin input wrapping.

B2 was already `B1(direct=True)` was already `B0` wrapped to accept a known-final
flag channel instead of always-finalized input; B1's other pipelines (`two_stage`,
`parallel_recent`, `direct_finalflag`-as-a-separate-thing, `joint_aux025`, and their
`SampleHead`/two-stage encode/decode machinery) are dropped, since `direct` is the
only one this project still uses (see docs/design/restructure-2026-unified.md §1).

`Model.forward` accepts either an already-packed `x` tensor (B0's original
interface) or separate `values`/`available`/`known_final` tensors plus a
`vintaged` bool: `vintaged=False` treats every visible cell as known-final
(finalized-input training); `vintaged=True` uses the supplied flags as-is
(vintaged-input training). This is the whole of what B1 added over B0 — no
subclass, no separate pipeline dispatch.
"""
import torch
from torch import nn
import torch.nn.functional as F

COUNT_TRANSFORMS = ('raw', 'rate', 'sqrt', 'fourth_root', 'log1p')
ED_TRANSFORMS = ('linear', 'logit', 'fourth_root')
ED_BOUNDS = (.0001, .9999)
LOCAL_LATENT = 4
MASK_SCENARIOS = ('natural', 'recent', 'gap', 'outage')


def transform_counts(counts, population, transform):
    """Admissions into model space: raw counts, or a transform of the rate per 100,000."""
    if transform == 'raw':
        return counts
    rate = counts.clamp_min(0) * (100000 / population)
    if transform == 'rate':
        return rate
    if transform == 'sqrt':
        return rate.sqrt()
    if transform == 'fourth_root':
        return rate.pow(.25)
    if transform == 'log1p':
        return rate.log1p()
    raise ValueError(f'Unknown count transform: {transform}')


def invert_counts(values, population, transform):
    """Nonnegative model-space values back to admission counts."""
    if transform == 'raw':
        return values
    if transform == 'rate':
        rate = values
    elif transform == 'sqrt':
        rate = values.square()
    elif transform == 'fourth_root':
        rate = values.pow(4)
    elif transform == 'log1p':
        rate = values.expm1()
    else:
        raise ValueError(f'Unknown count transform: {transform}')
    return rate * (population / 100000)


def transform_proportions(proportions, transform):
    """ED proportions into model space; logit clamps to the decoder's bounds."""
    if transform == 'linear':
        return proportions
    if transform == 'logit':
        return torch.logit(proportions.clamp(*ED_BOUNDS))
    if transform == 'fourth_root':
        return proportions.clamp_min(0).pow(.25)
    raise ValueError(f'Unknown ED transform: {transform}')


def positive_residual(anchor, delta):
    """Softplus residual around a positive anchor [N,C,L]; zero delta returns the anchor."""
    positive = anchor.clamp_min(.001)
    base = positive + torch.log(-torch.expm1(-positive))  # inverse softplus
    return F.softplus(base[None, :, None] + delta)


def affine(modulation, z, local=None, local_z=None, local_scale=None):
    """Per-member scale and shift broadcast as [M,N,H,L,C,width]."""
    gamma, beta = modulation(z)[:, :, None, None, None, :].chunk(2, -1)
    if local is not None:
        local_gamma, local_beta = (local(local_z) * local_scale)[:, :, None, :, None, :].chunk(2, -1)
        gamma, beta = gamma + local_gamma, beta + local_beta
    return gamma, beta


def modulation_layer(inputs, width):
    layer = nn.Linear(inputs, 2 * width)
    nn.init.normal_(layer.weight, std=.02)
    nn.init.zeros_(layer.bias)
    return layer


def softplus_inverse(value):
    return torch.log(torch.expm1(torch.tensor(float(value))))


class TemporalEncoder(nn.Module):
    """Two small convolutions shared over weeks; pool mean and latest features."""
    def __init__(self, inputs, width):
        super().__init__()
        self.filters = nn.Sequential(nn.Conv1d(inputs, width, 3, padding=1), nn.SiLU(),
                                      nn.Conv1d(width, width, 3, padding=1), nn.SiLU())
        self.project = nn.Linear(2 * width, width)

    def forward(self, x):
        h = self.filters(x)
        return self.project(torch.cat((h.mean(-1), h[..., -1]), -1))


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


class SpatialBlock(nn.Module):
    """One pre-norm attention block across the location tokens of each episode."""
    def __init__(self, width, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('Spatial attention width must be divisible by its four heads')
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, h):  # [N, L, width]
        query = self.norms[0](h)
        h = h + self.attention(query, query, query, need_weights=False)[0]
        return h + self.feedforward(self.norms[1](h))


class ModulatedDecoder(nn.Module):
    """Two residual blocks, each modulated by the same episode-level latent."""
    def __init__(self, width, latent, local=0):
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
        if local:
            self.local_modulations = nn.ModuleList([modulation_layer(local, width) for _ in range(2)])
            self.local_scale = nn.Parameter(softplus_inverse(1))

    def forward(self, h, z, local_z=None):
        hidden = h[None]
        local = getattr(self, 'local_modulations', [None, None])
        scale = F.softplus(self.local_scale) if hasattr(self, 'local_scale') else None
        for norm, modulation, local_modulation, block in zip(self.norms, self.modulations, local, self.blocks):
            gamma, beta = affine(modulation, z, local_modulation, local_z, scale)
            hidden = hidden + block(norm(hidden) * (1 + gamma) + beta)
        return self.output(hidden)


class ForecastHead(nn.Module):
    """Identical per-group designs; each group learns its own readout and modulation."""
    def __init__(self, width, latent, local, kind):
        super().__init__()
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
        local = getattr(self, 'local_modulate', None)
        scale = F.softplus(self.local_scale) if local is not None else None
        gamma, beta = affine(self.modulate, z, local, local_z, scale)
        return self.output(h[None] * (1 + gamma) + beta)


def fair_crps_cells(samples, truth, mask):
    """Return masked native-unit scores [N,H,C,L]; samples [M,N,H,C,L]."""
    m = samples.shape[0]
    if m < 2:
        raise ValueError('Fair CRPS needs at least two independent members')
    samples = torch.where(mask.bool().unsqueeze(0), samples, 0)
    truth = torch.where(mask.bool(), truth, 0)
    ordered = samples.reshape(m, -1).sort(dim=0).values.reshape_as(samples)
    weights = (2 * torch.arange(m, device=samples.device) - m + 1).to(samples.dtype)
    spread = (ordered * weights.reshape(m, 1, 1, 1, 1)).sum(0) / (m * (m - 1))
    score = (samples - truth).abs().mean(0) - spread
    return score * mask


def fair_crps(samples, truth, mask):
    """Unweighted per-channel diagnostic; fitting uses explicit cell weights."""
    return fair_crps_cells(samples, truth, mask).sum((0, 1, 3)) / mask.sum((0, 1, 3)).clamp_min(1)


def draw_dropout(available, rng, probabilities=(.50, .25, .15, .10), scenario=None):
    """A and returned D are [episode, week, channel, location]; V = A & ~D.

    Recent reports/outages affect all locations; local gaps affect one history.
    D marks naturally available cells only, and is never supplied to the network.
    """
    import numpy as np
    a = np.asarray(available, dtype=bool)
    if a.ndim != 4 or a.shape[2] != 6:
        raise ValueError('Availability must be [episode, week, 6, location]')
    probabilities = np.asarray(probabilities, float)
    if probabilities.shape != (4,) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError('Four masking probabilities must sum to one')
    if scenario is not None and scenario not in MASK_SCENARIOS:
        raise ValueError(f'Unknown mask scenario: {scenario}')
    d = np.zeros_like(a)
    for n in range(len(a)):
        kind = scenario or rng.choice(MASK_SCENARIOS, p=probabilities)
        c = int(rng.integers(6))
        if kind == 'recent':
            channels = [c] if rng.random() < .5 else list(range(0, 3) if c < 3 else range(3, 6))
            weeks = int(rng.integers(1, min(2, a.shape[1]) + 1))
            d[n, -weeks:, channels, :] = True
        elif kind == 'gap':
            length = int(rng.integers(1, min(3, a.shape[1]) + 1))
            start = int(rng.integers(a.shape[1] - length + 1))
            d[n, start:start + length, c, int(rng.integers(a.shape[-1]))] = True
        elif kind == 'outage':
            d[n, :, c] = True
    return d & a


class Model(nn.Module):
    def __init__(self, lookback=8, horizons=(1, 2, 3, 4), width=64, latent=16, scale=None,
                 count_transform='raw', populations=None, geography=False, dynamics=False, input_scale=None,
                 encoder='mlp', heads='shared', decoder='legacy', ed_transform='linear', spatial='none',
                 noise='global', us_error='none', input_offset=None, head_sharing='shared',
                 annual_calendar=True, location_embedding=0, location_ids=None, supplied_final=False,
                 covariate_names=(), covariate_offset=None, covariate_scale=None, covariate_trained=None):
        super().__init__()
        self.config = dict(lookback=lookback, horizons=list(horizons), width=width, latent=latent,
                           supplied_final=supplied_final)
        if (encoder not in ('mlp', 'conv', 'multiscale_conv') or heads not in ('shared', 'state_us') or decoder not in ('legacy', 'residual2')
                or spatial not in ('none', 'attention', 'pathogen_spatial', 'target_spatial', 'joint_location_target') or noise not in ('global', 'local')
                or us_error not in ('none', 'shared_factor') or head_sharing not in ('shared', 'pathogen', 'target')):
            raise ValueError('Unknown model architecture option')
        if min(lookback, width, latent) < 1:
            raise ValueError('Model dimensions must be positive')
        if count_transform not in COUNT_TRANSFORMS:
            raise ValueError('Unknown count transform')
        if ed_transform not in ED_TRANSFORMS:
            raise ValueError('Unknown ED transform')
        if (count_transform != 'raw' or geography) and not populations:
            raise ValueError('Population metadata required')
        if populations and any(not torch.isfinite(torch.tensor(float(v))) or float(v) <= 0 for v in populations.values()):
            raise ValueError('Populations must be finite and positive')
        self.config.update(count_transform=count_transform, populations=populations,
                           geography=geography, dynamics=dynamics, encoder=encoder, heads=heads, decoder=decoder,
                           ed_transform=ed_transform, spatial=spatial, noise=noise, us_error=us_error,
                           head_sharing=head_sharing, annual_calendar=annual_calendar,
                           location_embedding=location_embedding, location_ids=location_ids or list(populations or {}))
        covariate_names = list(covariate_names)
        if len(set(covariate_names)) != len(covariate_names):
            raise ValueError('Covariate names must be unique')
        self.config['covariate_names'] = covariate_names
        self.register_buffer('input_scale', self._per_location(input_scale, 1.))
        if ed_transform == 'logit':
            self.register_buffer('input_offset', self._per_location(input_offset, 0.))
        self.register_buffer('scale', self._per_location(scale, 1.))
        if covariate_names:
            self.register_buffer('covariate_offset', self._per_covariate(covariate_offset, len(covariate_names), 0.))
            self.register_buffer('covariate_scale', self._per_covariate(covariate_scale, len(covariate_names), 1.))
            trained = self._per_covariate(covariate_trained, len(covariate_names), 1.).bool()
            self.register_buffer('covariate_trained', trained)
        else:
            self.covariate_offset = torch.empty(0, 1)
            self.covariate_scale = torch.empty(0, 1)
            self.covariate_trained = torch.empty(0, 1, dtype=torch.bool)
        self.config['scale'] = self.scale.tolist()
        self.config['input_scale'] = self.input_scale.tolist()
        if ed_transform == 'logit':
            self.config['input_offset'] = self.input_offset.tolist()
        self.config['covariate_offset'] = self.covariate_offset.tolist()
        self.config['covariate_scale'] = self.covariate_scale.tolist()
        self.config['covariate_trained'] = self.covariate_trained.tolist()
        temporal = MultiscaleEncoder if encoder == 'multiscale_conv' else TemporalEncoder
        fields_per_cell = 3 if supplied_final else 2
        extra_width = 3 * annual_calendar + 2 * geography + 30 * dynamics + location_embedding
        context_fields = 6 * fields_per_cell + 2 * len(covariate_names)
        self.context = nn.Sequential(nn.Linear((lookback * context_fields if encoder == 'mlp' else width) + extra_width, width),
                                     nn.SiLU(), nn.Linear(width, width))
        self.focal = (nn.Sequential(nn.Linear(lookback * fields_per_cell, width), nn.SiLU(), nn.Linear(width, width))
                      if encoder == 'mlp' else temporal(fields_per_cell, width))
        if encoder != 'mlp':
            self.temporal_context = temporal(context_fields, width)
        self.source = nn.Embedding(6, width)
        self.horizon = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)
        if location_embedding:
            if not self.config['location_ids']:
                raise ValueError('Location embeddings require ordered location IDs')
            self.location_id = nn.Embedding(len(self.config['location_ids']), location_embedding)
        self.output_groups = ([list(range(6))] if head_sharing == 'shared' else
                              [[0, 3], [1, 4], [2, 5]] if head_sharing == 'pathogen' else [[c] for c in range(6)])
        local = LOCAL_LATENT if noise == 'local' else 0
        self.output_heads = nn.ModuleList([ForecastHead(width, latent, local, decoder) for _ in self.output_groups])
        if heads == 'state_us':
            self.us_output_heads = nn.ModuleList([ForecastHead(width, latent, local, decoder) for _ in self.output_groups])
        if spatial != 'none':
            self.spatial = SpatialBlock(width)
        if spatial not in ('none', 'attention'):
            scope_channels = 2 if spatial == 'pathogen_spatial' else 1
            self.remote = (nn.Sequential(nn.Linear(lookback * scope_channels * fields_per_cell, width), nn.SiLU(), nn.Linear(width, width))
                           if encoder == 'mlp' else temporal(scope_channels * fields_per_cell, width))
            self.remote_identity = nn.Embedding(3 if spatial == 'pathogen_spatial' else 6, width)
            if geography or location_embedding:
                self.remote_geo = nn.Linear(2 * geography + location_embedding, width)
        if us_error == 'shared_factor':
            self.national_scale = nn.Parameter(softplus_inverse(.1).expand(6).clone())

    @staticmethod
    def _per_location(value, default):
        if value is None:
            return torch.full((6, 1), float(default))
        tensor = torch.as_tensor(value).float()
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.ndim != 2 or tensor.shape[0] != 6:
            raise ValueError(f'Per-location scales must be [6] or [6, locations], got {tuple(tensor.shape)}')
        return tensor.contiguous()

    @staticmethod
    def _per_covariate(value, count, default):
        if value is None:
            return torch.full((count, 1), float(default))
        tensor = torch.as_tensor(value).float()
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.ndim != 2 or tensor.shape[0] != count:
            raise ValueError(f'Covariate statistics must be [{count}] or [{count}, locations], got {tuple(tensor.shape)}')
        return tensor.contiguous()

    def local_noise_scales(self):
        return {name: float(F.softplus(value.detach())) for name, value in self.named_parameters() if name.endswith('local_scale')}

    def forward(self, x=None, calendar=None, members=8, z=None, locations=None, local_z=None, national_z=None,
                covariates=None, *, values=None, available=None, known_final=None, vintaged=True):
        """Native samples [member, episode, output week, component target, location].

        Either pass a pre-packed `x` [N,P,C,fields,L] (this model's own format,
        `fields` = 2 or 3 matching `supplied_final`), or pass `values`/`available`
        (and, when `vintaged=True`, `known_final`) and let this wrapper pack them:
        `vintaged=False` treats every visible cell as known-final (a finalized-input
        scenario); `vintaged=True` uses the supplied flags as-is.
        """
        if x is None:
            if values is None or available is None:
                raise ValueError('Provide either a packed x tensor, or values and available')
            available = available.bool()
            if self.config['supplied_final']:
                if vintaged:
                    known_final = available if known_final is None else known_final.bool() & available
                else:
                    known_final = available
                x = torch.stack((values, available.to(values.dtype), known_final.to(values.dtype)), dim=3)
            else:
                x = torch.stack((values, available.to(values.dtype)), dim=3)
        return self._forward(x, calendar, members=members, z=z, locations=locations,
                              local_z=local_z, national_z=national_z, covariates=covariates)

    def _forward(self, x, calendar, members=8, z=None, locations=None, local_z=None, national_z=None,
                 covariates=None):
        n, p, c, _, l = x.shape
        config = self.config
        if config['heads'] == 'state_us' and (locations is None or len(locations) != l):
            raise ValueError('Separate heads require ordered location IDs')
        mask = x[:, :, :, 1, :]
        valid = mask.bool()
        raw = torch.where(valid, x[:, :, :, 0, :], 0)
        population = None
        if config['count_transform'] != 'raw' or config['geography']:
            if locations is None or len(locations) != l:
                raise ValueError('Provide ordered location IDs for population metadata')
            population = x.new_tensor([config['populations'][loc] for loc in locations])
        input_scale = self.input_scale
        offset = getattr(self, 'input_offset', None)
        if input_scale.shape[-1] not in (1, l):
            raise ValueError(f'Input scale holds {input_scale.shape[-1]} locations, not {l}')
        transformed = torch.cat((transform_counts(raw[:, :, :3], population, config['count_transform']),
                                 transform_proportions(raw[:, :, 3:], config['ed_transform'])), 2)
        if offset is not None:
            transformed = transformed - offset[None, None, :, :]
        values = torch.where(valid, transformed / input_scale[None, None, :, :], 0)
        fields = torch.stack((values, mask, (x[:, :, :, 2, :].bool() & valid).to(values.dtype))
                             if config['supplied_final'] else (values, mask), dim=-1)
        fpc = fields.shape[-1]
        cov_fields = None
        k = len(config['covariate_names'])
        if k:
            expected = (n, p, k, 2, l)
            if covariates is None or tuple(covariates.shape) != expected:
                raise ValueError(f'Model expects covariates {expected}')
            cov_available = covariates[:, :, :, 1].bool() & self.covariate_trained[None, None]
            if self.covariate_scale.shape[-1] not in (1, l):
                raise ValueError(f'Covariate scale holds {self.covariate_scale.shape[-1]} locations, not {l}')
            cov_values = (covariates[:, :, :, 0] - self.covariate_offset[None, None]) / self.covariate_scale[None, None]
            cov_values = torch.where(cov_available, cov_values, 0)
            cov_fields = torch.stack((cov_values, cov_available.to(cov_values.dtype)), -1)
        elif covariates is not None and covariates.shape[2]:
            raise ValueError('Model was fitted without covariates')
        if config['encoder'] == 'mlp':
            pieces = [fields.permute(0, 3, 1, 2, 4).reshape(n, l, -1)]
            if cov_fields is not None:
                pieces.append(cov_fields.permute(0, 3, 1, 2, 4).reshape(n, l, -1))
            context = torch.cat(pieces, -1)
        else:
            pieces = [fields.permute(0, 3, 2, 4, 1).reshape(n * l, c * fpc, p)]
            if cov_fields is not None:
                pieces.append(cov_fields.permute(0, 3, 2, 4, 1).reshape(n * l, 2 * k, p))
            context = self.temporal_context(torch.cat(pieces, 1)).reshape(n, l, -1)
        extras, geo_features = [], []
        if config['annual_calendar']:
            if calendar.shape[-1] != 3:
                raise ValueError('Calendar requires annual phase and Christmas timing')
            extras.append(calendar[:, None, :].expand(-1, l, -1))
        if config['geography']:
            geo = torch.stack(((population / 100000).log(), x.new_tensor([loc == 'US' for loc in locations])), -1)
            geo_features.append(geo[None].expand(n, -1, -1))
        if config['location_embedding']:
            ids = torch.tensor([config['location_ids'].index(loc) for loc in locations], device=x.device)
            geo_features.append(self.location_id(ids)[None].expand(n, -1, -1))
        extras.extend(geo_features)
        if config['dynamics']:
            extras.append(recent_dynamics(values, mask).permute(0, 2, 1))
        context = self.context(torch.cat([context, *extras], -1))
        if config['encoder'] == 'mlp':
            focal = self.focal(fields.permute(0, 3, 2, 1, 4).reshape(n, l, c, p * fpc))
        else:
            focal = self.focal(fields.permute(0, 3, 2, 4, 1).reshape(n * l * c, fpc, p)).reshape(n, l, c, -1)
        if config['spatial'] == 'attention':
            context = self.spatial(context)
        h = context[:, :, None, :] + focal + self.source.weight[None, None, :, :]
        if config['spatial'] not in ('none', 'attention'):
            scope = config['spatial']
            groups = ([[0, 3], [1, 4], [2, 5]] if scope == 'pathogen_spatial' else [[i] for i in range(6)])
            tokens = []
            for gi, channels in enumerate(groups):
                f = fields[:, :, channels]
                remote_input = (f.permute(0, 3, 1, 2, 4).reshape(n, l, -1) if config['encoder'] == 'mlp'
                                else f.permute(0, 3, 2, 4, 1).reshape(n * l, len(channels) * fpc, p))
                token = self.remote(remote_input).reshape(n, l, -1)
                token = token + self.remote_identity.weight[gi]
                if geo_features:
                    token = token + self.remote_geo(torch.cat(geo_features, -1))
                tokens.append(token)
            remote = torch.stack(tokens, 2)  # N,L,group,W
            if scope == 'joint_location_target':
                remote = self.spatial(remote.reshape(n, l * len(groups), -1)).reshape_as(remote)
            else:
                remote = self.spatial(remote.permute(0, 2, 1, 3).reshape(n * len(groups), l, -1)).reshape(n, len(groups), l, -1).permute(0, 2, 1, 3)
            mapping = [next(i for i, group in enumerate(groups) if c in group) for c in range(6)]
            h = h + remote[:, :, mapping]
        offsets = x.new_tensor(config['horizons']).reshape(-1, 1) / 4
        h = self.norm(h[:, None, :, :, :] + self.horizon(offsets)[None, :, None, None, :])
        if z is None:
            z = torch.randn(members, n, config['latent'], device=x.device)
        if config['noise'] == 'local':
            if local_z is None:
                local_z = torch.randn(z.shape[0], n, l, LOCAL_LATENT, device=x.device)
            if local_z.shape != (z.shape[0], n, l, LOCAL_LATENT):
                raise ValueError('Local latent must have shape [members, episodes, locations, 4]')

        def decode(heads, context=h):
            outputs = [head(context[:, :, :, group], z, local_z)
                       for head, group in zip(heads, self.output_groups)]
            order = [channel for group in self.output_groups for channel in group]
            return torch.cat(outputs, -2)[..., [order.index(c) for c in range(6)], :]
        delta = decode(self.output_heads)
        if config['heads'] == 'state_us':
            us = decode(self.us_output_heads)
            is_us = torch.tensor([loc == 'US' for loc in locations], device=x.device)
            delta = torch.where(is_us[None, None, None, :, None, None], us, delta)
        if config['us_error'] == 'shared_factor':
            if national_z is None:
                national_z = torch.randn(delta.shape[0], n, c, device=x.device, dtype=delta.dtype)
            if national_z.shape != (delta.shape[0], n, c):
                raise ValueError('National noise must have shape [members, episodes, channels]')
            delta = delta + national_z[:, :, None, None, :, None] * F.softplus(self.national_scale)[None, None, None, None, :, None]
        delta = delta.squeeze(-1).permute(0, 1, 2, 4, 3)
        idx = (mask * torch.arange(1, p + 1, device=x.device)[None, :, None, None]).argmax(1)
        anchor = values.gather(1, idx[:, None]).squeeze(1)
        anchor = torch.where(mask.any(1), anchor, anchor.new_full((), .01))
        counts = positive_residual(anchor[:, :3], delta[:, :, :, :3]) * input_scale[None, None, None, :3, :]
        counts = invert_counts(counts, population, config['count_transform'])
        if config['ed_transform'] == 'fourth_root':
            root = positive_residual(anchor[:, 3:], delta[:, :, :, 3:]) * input_scale[None, None, None, 3:, :]
            ed = root.pow(4).clamp(max=1)
        else:
            if config['ed_transform'] == 'logit':
                logit = anchor[:, 3:] * input_scale[None, 3:, :] + offset[None, 3:, :]
            else:
                proportion = (anchor[:, 3:] * input_scale[None, 3:, :]).clamp(*ED_BOUNDS)
                logit = torch.logit(proportion)
            ed = torch.sigmoid(logit[None, :, None] + delta[:, :, :, 3:])
        return torch.cat((counts, ed), dim=3)


def recent_dynamics(values, mask):
    """Causal last two weekly slopes, acceleration and age, with validity flags."""
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


class IndependentBundle(nn.Module):
    """`fit_partition != 'all'`: separately fitted component models, one per channel group."""
    def __init__(self, models, groups):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.groups = groups
        self.config = models[0].config

    def forward(self, *args, members=8, **kwargs):
        outputs = [model(*args, members=members, **kwargs)[:, :, :, group]
                   for model, group in zip(self.models, self.groups)]
        order = [c for group in self.groups for c in group]
        return torch.cat(outputs, 3)[:, :, :, [order.index(c) for c in range(6)]]


def checkpoint(model, metadata):
    import json
    metadata = json.loads(json.dumps(metadata))
    if isinstance(model, IndependentBundle):
        return dict(groups=model.groups, components=[checkpoint(m, {}) for m in model.models], metadata=metadata)
    return dict(config=model.config, state_dict={k: v.cpu() for k, v in model.state_dict().items()}, metadata=metadata)


def load_model(saved):
    if 'components' in saved:
        return IndependentBundle([load_model(c) for c in saved['components']], saved['groups'])
    model = Model(**saved['config'])
    model.load_state_dict(saved['state_dict'])
    return model
