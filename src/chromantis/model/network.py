"""The probabilistic forecasting network, its transforms and checkpoint format."""
import torch
from .geographic import GeographicMessage, COORDINATES
from .covariates import CovariateEncoder, signed_log, trailing_mean, multiscale_features
from torch import nn
import torch.nn.functional as F

COUNT_TRANSFORMS = ('raw', 'rate', 'sqrt', 'fourth_root', 'log1p')
ED_TRANSFORMS = ('linear', 'logit', 'fourth_root')
ED_BOUNDS = (.0001, .9999)
LOCAL_LATENT = 4


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


def horizon_residual(anchor, delta):
    """`positive_residual` with a per-horizon anchor [N,H,C,L]; identical when horizons share it."""
    positive = anchor.clamp_min(.001)
    base = positive + torch.log(-torch.expm1(-positive))
    return F.softplus(base[None] + delta)


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


def pooled_context(context, observed, locations):
    """Equal-weight observed state/DC mean plus a separate native-US token."""
    states = context.new_tensor([loc != 'US' for loc in locations], dtype=torch.bool)
    keep = observed & states[None]
    pooled = (context * keep[..., None]).sum(1) / keep.sum(1).clamp_min(1)[:, None]
    national = torch.zeros_like(pooled)
    if 'US' in locations:
        us = locations.index('US')
        national = context[:, us] * observed[:, us, None]
    return torch.cat((pooled, national), -1)


class SpatialBlock(nn.Module):
    """One pre-norm attention block across the location tokens of each episode."""
    def __init__(self, width, heads=4):
        super().__init__()
        if width % heads:
            raise ValueError('Spatial attention width must be divisible by its four heads')
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.feedforward = nn.Sequential(nn.Linear(width, 2 * width), nn.SiLU(), nn.Linear(2 * width, width))

    def forward(self, h, observed=None):  # [N, L, width]
        original = h
        query = self.norms[0](h)
        padding = None
        if observed is not None:
            any_observed = observed.any(-1)
            # Attention needs one unmasked key even in an entirely missing episode.
            # Its result is discarded for that episode below.
            padding = ~observed.clone()
            padding[~any_observed, 0] = False
        h = h + self.attention(query, query, query, key_padding_mask=padding, need_weights=False)[0]
        h = h + self.feedforward(self.norms[1](h))
        return h if observed is None else torch.where(any_observed[:, None, None], h, original)


class PooledMessage(nn.Module):
    """Native-US broadcast or recipient-gated state-mean plus native-US message."""
    def __init__(self, width, kind):
        super().__init__()
        self.kind = kind
        self.message = nn.Linear(width if kind == 'national_broadcast' else 2 * width, width, bias=False)
        if kind == 'gated_pool':
            self.gate = nn.Linear(3 * width, width)

    def forward(self, context, observed, locations):
        summary = pooled_context(context, observed, locations)
        if self.kind == 'national_broadcast':
            return context + self.message(summary.chunk(2, -1)[1])[:, None]
        message = self.message(summary)[:, None]
        gate = torch.sigmoid(self.gate(torch.cat((context, summary[:, None].expand(-1, context.shape[1], -1)), -1)))
        return context + gate * message


class ModulatedDecoder(nn.Module):
    """Two residual blocks, each modulated by the same episode-level latent."""
    def __init__(self, width, latent, local=0, blocks=2):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(blocks)])
        self.modulations = nn.ModuleList([nn.Linear(latent, 2 * width) for _ in range(blocks)])
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                                    nn.Linear(width, width)) for _ in range(blocks)])
        self.output = nn.Linear(width, 1)
        for layer in self.modulations:
            nn.init.normal_(layer.weight, std=.02)
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output.weight, std=.01)
        nn.init.zeros_(self.output.bias)
        if local:
            self.local_modulations = nn.ModuleList([modulation_layer(local, width) for _ in range(blocks)])
            self.local_scale = nn.Parameter(softplus_inverse(1))

    def forward(self, h, z, local_z=None):
        hidden = h[None]
        local = getattr(self, 'local_modulations', [None] * len(self.blocks))
        scale = F.softplus(self.local_scale) if hasattr(self, 'local_scale') else None
        for norm, modulation, local_modulation, block in zip(self.norms, self.modulations, local, self.blocks):
            gamma, beta = affine(modulation, z, local_modulation, local_z, scale)
            hidden = hidden + block(norm(hidden) * (1 + gamma) + beta)
        return self.output(hidden)


class ForecastHead(nn.Module):
    """Identical per-group designs; each group learns its own readout and modulation."""
    def __init__(self, width, latent, local, kind, blocks=0):
        super().__init__()
        self.kind = 'residual2' if blocks and kind == 'legacy' else kind
        kind = self.kind
        if kind in ('quantile', 'quantile_small'):
            if blocks:
                self.residual_blocks = nn.ModuleList([
                    nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
                    for _ in range(blocks)])
            self.output = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3 if kind == 'quantile_small' else 23))
            nn.init.normal_(self.output[-1].weight, std=.01)
            nn.init.zeros_(self.output[-1].bias)
            if kind == 'quantile_small':
                # Two shared monotone shapes; only median and two spreads vary by forecast.
                self.shape_gaps = nn.Parameter(torch.zeros(2, 11))
        elif kind == 'residual2':
            self.output = ModulatedDecoder(width, latent, local, blocks or 2)
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
        for block in getattr(self, 'residual_blocks', []):
            h = h + block(h)
        if self.kind == 'quantile_small':
            raw = self.output(h)
            shape = F.softplus(self.shape_gaps).cumsum(-1)
            shape = shape / shape[:, -1:]
            median = raw[..., :1]
            lower = median - F.softplus(raw[..., 1:2]) * shape[0].flip(-1)
            upper = median + F.softplus(raw[..., 2:3]) * shape[1]
            return torch.cat((lower, median, upper), -1).movedim(-1, 0).unsqueeze(-1)
        if self.kind == 'quantile':
            raw = self.output(h)
            median = raw[..., 11:12]
            lower = median - torch.flip(torch.cumsum(F.softplus(torch.flip(raw[..., :11], [-1])) * .05, -1), [-1])
            upper = median + torch.cumsum(F.softplus(raw[..., 12:]) * .05, -1)
            return torch.cat((lower, median, upper), -1).movedim(-1, 0).unsqueeze(-1)
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


class Model(nn.Module):
    def __init__(self, lookback=8, horizons=(1, 2, 3, 4), width=64, latent=16, scale=None,
                 count_transform='raw', populations=None, geography=False, dynamics=False, input_scale=None,
                 encoder='mlp', heads='shared', decoder='legacy', ed_transform='linear', spatial='none',
                 noise='global', us_error='none', input_offset=None, head_sharing='shared',
                 annual_calendar=True, location_embedding=0, location_ids=None,
                 covariate_names=(), covariate_offset=None, covariate_scale=None, covariate_trained=None,
                 covariate_encoder='raw', coordinates=False, signal_features='none', direct_quantiles=False,
                 input_names=(), input_units=(), input_groups=(), target_names=(), target_units=(),
                 target_groups=(), target_input_indices=(), growth_anchor=False, decoder_blocks=0):
        super().__init__()
        input_names, input_units, input_groups = map(list, (input_names, input_units, input_groups))
        target_names, target_units, target_groups = map(list, (target_names, target_units, target_groups))
        target_input_indices = list(map(int, target_input_indices))
        if not input_names or not target_names:
            raise ValueError('Model needs named inputs and targets')
        if len(input_names) != len(input_units) or len(input_names) != len(input_groups):
            raise ValueError('Each input needs a unit and group')
        if len(target_names) != len(target_units) or len(target_names) != len(target_groups):
            raise ValueError('Each target needs a unit and group')
        if len(target_input_indices) != len(target_names) or any(i < 0 or i >= len(input_names) for i in target_input_indices):
            raise ValueError('Each target needs a valid input-history index')
        inputs, targets = len(input_names), len(target_names)
        self.config = dict(lookback=lookback, horizons=list(horizons), width=width, latent=latent,
                           input_names=input_names, input_units=input_units, input_groups=input_groups,
                           target_names=target_names, target_units=target_units, target_groups=target_groups,
                           target_input_indices=target_input_indices,
                           growth_anchor=growth_anchor, decoder_blocks=decoder_blocks)
        if (encoder not in ('mlp', 'conv', 'multiscale_conv') or heads not in ('shared', 'state_us') or decoder not in ('legacy', 'residual2', 'quantile', 'quantile_small')
                or spatial not in ('neighbors', 'distance', 'gravity', 'none', 'pooled', 'national_broadcast', 'gated_pool', 'attention', 'pathogen_spatial', 'target_spatial', 'joint_location_target') or noise not in ('global', 'local')
                or us_error not in ('none', 'shared_factor') or head_sharing not in ('shared', 'pathogen', 'target')):
            raise ValueError('Unknown model architecture option')
        if decoder in ('quantile', 'quantile_small'):
            self.config['direct_quantiles'] = True
            if noise != 'global' or us_error != 'none':raise ValueError('Direct quantiles cannot add sampled output noise')
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
                           geography=geography, coordinates=coordinates, dynamics=dynamics, encoder=encoder, heads=heads, decoder=decoder,
                           ed_transform=ed_transform, spatial=spatial, noise=noise, us_error=us_error,
                           head_sharing=head_sharing, annual_calendar=annual_calendar,
                           location_embedding=location_embedding, location_ids=location_ids or list(populations or {}))
        covariate_names = list(covariate_names)
        if len(set(covariate_names)) != len(covariate_names):
            raise ValueError('Covariate names must be unique')
        if covariate_encoder not in ('raw', 'smooth', 'summary', 'shared', 'growth'):
            raise ValueError('Unknown covariate encoder')
        if signal_features not in ('none', 'multiscale', 'smooth_multiscale'):
            raise ValueError('Unknown signal features')
        self.config['signal_features'] = signal_features
        self.config['covariate_encoder'] = covariate_encoder
        self.config['covariate_names'] = covariate_names
        self.register_buffer('input_scale', self._per_location(input_scale, inputs, 1.))
        if ed_transform == 'logit':
            self.register_buffer('input_offset', self._per_location(input_offset, inputs, 0.))
        self.register_buffer('scale', self._per_location(scale, targets, 1.))
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
        fields_per_cell = 2  # value and availability
        extra_width = 3 * annual_calendar + 2 * geography + 5 * inputs * dynamics + location_embedding
        if signal_features != 'none':
            extra_width += 18 * (inputs + len(covariate_names))
        compact_covariates = bool(covariate_names) and covariate_encoder in ('summary', 'shared', 'growth')
        if compact_covariates:
            self.covariate_encoder = CovariateEncoder(lookback, covariate_encoder)
            extra_width += 6 * len(covariate_names)
        context_fields = inputs * fields_per_cell + (0 if compact_covariates else 2 * len(covariate_names))
        self.context = nn.Sequential(nn.Linear((lookback * context_fields if encoder == 'mlp' else width) + extra_width, width),
                                     nn.SiLU(), nn.Linear(width, width))
        self.focal = (nn.Sequential(nn.Linear(lookback * fields_per_cell, width), nn.SiLU(), nn.Linear(width, width))
                      if encoder == 'mlp' else temporal(fields_per_cell, width))
        if encoder != 'mlp':
            self.temporal_context = temporal(context_fields, width)
        self.source = nn.Embedding(targets, width)
        self.horizon = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)
        if location_embedding:
            if not self.config['location_ids']:
                raise ValueError('Location embeddings require ordered location IDs')
            self.location_id = nn.Embedding(len(self.config['location_ids']), location_embedding)
        if head_sharing == 'shared':
            self.output_groups = [list(range(targets))]
        elif head_sharing == 'pathogen':
            labels = list(dict.fromkeys(target_groups))
            self.output_groups = [[i for i, group in enumerate(target_groups) if group == label] for label in labels]
        else:
            self.output_groups = [[c] for c in range(targets)]
        local = LOCAL_LATENT if noise == 'local' else 0
        self.output_heads = nn.ModuleList([ForecastHead(width, latent, local, decoder, decoder_blocks) for _ in self.output_groups])
        if heads == 'state_us':
            self.us_output_heads = nn.ModuleList([ForecastHead(width, latent, local, decoder, decoder_blocks) for _ in self.output_groups])
        if coordinates:
            self.coordinate_project = nn.Linear(3, width, bias=False)
        if spatial in ('neighbors', 'distance', 'gravity'):
            self.geographic_message = GeographicMessage(width, spatial)
        if spatial == 'pooled':
            self.national_context = nn.Linear(2 * width, width)
        elif spatial in ('national_broadcast', 'gated_pool'):
            self.pooled_message = PooledMessage(width, spatial)
        elif spatial not in ('none', 'neighbors', 'distance', 'gravity'):
            self.spatial = SpatialBlock(width)
        if spatial in ('pathogen_spatial', 'target_spatial', 'joint_location_target'):
            if spatial == 'pathogen_spatial':
                labels = list(dict.fromkeys(target_groups))
                remote_groups = [[i for i, group in enumerate(input_groups) if group == label] for label in labels]
                target_remote = [labels.index(group) for group in target_groups]
            else:
                remote_groups = [[index] for index in target_input_indices]
                target_remote = list(range(targets))
            if not remote_groups or len({len(group) for group in remote_groups}) != 1 or not remote_groups[0]:
                raise ValueError('Spatial target groups need the same nonzero number of input histories')
            self.config['remote_groups'] = remote_groups
            self.config['target_remote'] = target_remote
            scope_channels = len(remote_groups[0])
            self.remote = (nn.Sequential(nn.Linear(lookback * scope_channels * fields_per_cell, width), nn.SiLU(), nn.Linear(width, width))
                           if encoder == 'mlp' else temporal(scope_channels * fields_per_cell, width))
            self.remote_identity = nn.Embedding(len(remote_groups), width)
            if covariate_names:
                covariate_width = len(covariate_names) * (6 if compact_covariates else 2 * lookback)
                self.remote_covariates = nn.Sequential(nn.Linear(covariate_width, width), nn.SiLU(),
                                                       nn.Linear(width, width))
            if geography or location_embedding:
                self.remote_geo = nn.Linear(2 * geography + location_embedding, width)
        if us_error == 'shared_factor':
            self.national_scale = nn.Parameter(softplus_inverse(.1).expand(targets).clone())

    @staticmethod
    def _per_location(value, count, default):
        if value is None:
            return torch.full((count, 1), float(default))
        tensor = torch.as_tensor(value).float()
        if tensor.ndim == 1:
            tensor = tensor[:, None]
        if tensor.ndim != 2 or tensor.shape[0] != count:
            raise ValueError(f'Per-location scales must be [{count}] or [{count}, locations], got {tuple(tensor.shape)}')
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
                covariates=None, *, values=None, available=None):
        """Native samples [member, episode, output week, component target, location].

        Either pass a pre-packed `x` [N,P,C,2,L] (value, availability), or pass
        `values`/`available` and let this wrapper pack them.
        """
        if x is None:
            if values is None or available is None:
                raise ValueError('Provide either a packed x tensor, or values and available')
            x = torch.stack((values, available.bool().to(values.dtype)), dim=3)
        return self._forward(x, calendar, members=members, z=z, locations=locations,
                              local_z=local_z, national_z=national_z, covariates=covariates)

    def _forward(self, x, calendar, members=8, z=None, locations=None, local_z=None, national_z=None,
                 covariates=None):
        n, p, c, _, l = x.shape
        config = self.config
        targets = len(config['target_names'])
        if c != len(config['input_names']):
            raise ValueError(f'Model expects {len(config["input_names"])} input signals, got {c}')
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
        transformed = torch.zeros_like(raw)
        count_inputs = [i for i, unit in enumerate(config['input_units']) if unit == 'count']
        proportion_inputs = [i for i, unit in enumerate(config['input_units']) if unit == 'proportion']
        if count_inputs:
            transformed[:, :, count_inputs] = transform_counts(raw[:, :, count_inputs], population, config['count_transform'])
        if proportion_inputs:
            transformed[:, :, proportion_inputs] = transform_proportions(raw[:, :, proportion_inputs], config['ed_transform'])
        if offset is not None:
            transformed = transformed - offset[None, None, :, :]
        values = torch.where(valid, transformed / input_scale[None, None, :, :], 0)
        fields = torch.stack([values, mask], dim=-1)
        fpc = fields.shape[-1]
        cov_fields = cov_summary = None
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
            feature_cov_values = signed_log(cov_values)
            feature_cov_available = cov_available
            kind = config['covariate_encoder']
            if kind != 'raw':
                cov_values = signed_log(cov_values)
            if kind == 'smooth':
                cov_values, cov_available = trailing_mean(cov_values, cov_available)
            if kind == 'growth':
                # Log of raw values, floored at 10% of the training-only SD (no zero/negative logs).
                raw_cov = covariates[:, :, :, 0].clamp_min(0)
                log_cov = torch.log(raw_cov + .1 * self.covariate_scale[None, None])
                cov_summary = self.covariate_encoder(cov_values, cov_available, log_cov)
            elif kind in ('summary', 'shared'):
                cov_summary = self.covariate_encoder(cov_values, cov_available)
            else:
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
        if config['signal_features'] != 'none':
            smooth_features = config['signal_features'] == 'smooth_multiscale'
            extras.append(multiscale_features(values, valid, smooth_features))
            if k:
                extras.append(multiscale_features(feature_cov_values, feature_cov_available, smooth_features))
        if cov_summary is not None:
            extras.append(cov_summary)
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
        target_fields = fields[:, :, config['target_input_indices']]
        if config['encoder'] == 'mlp':
            focal = self.focal(target_fields.permute(0, 3, 2, 1, 4).reshape(n, l, targets, p * fpc))
        else:
            focal = self.focal(target_fields.permute(0, 3, 2, 4, 1).reshape(n * l * targets, fpc, p)).reshape(n, l, targets, -1)
        if config['coordinates']:
            coordinates = context.new_tensor([[*COORDINATES.get(loc, [0., 0.]), float(loc != 'US')] for loc in locations])
            coordinates[:, :2] /= 180
            context = context + self.coordinate_project(coordinates)[None]
        if config['spatial'] in ('neighbors', 'distance', 'gravity', 'pooled', 'national_broadcast', 'gated_pool', 'attention'):
            observed = valid.any(dim=(1, 2))
            if k:
                observed = observed | cov_available.any(dim=(1, 2))
        if config['spatial'] in ('neighbors', 'distance', 'gravity'):
            context = self.geographic_message(context, observed, locations, config['populations'])
        if config['spatial'] == 'pooled':
            national = pooled_context(context, observed, locations)
            context = context + self.national_context(national)[:, None]
        if config['spatial'] in ('national_broadcast', 'gated_pool'):
            context = self.pooled_message(context, observed, locations)
        if config['spatial'] == 'attention':
            context = self.spatial(context, observed)
        h = context[:, :, None, :] + focal + self.source.weight[None, None, :, :]
        if config['spatial'] in ('pathogen_spatial', 'target_spatial', 'joint_location_target'):
            scope = config['spatial']
            groups = config['remote_groups']
            remote_covariates = None
            if k:
                cov_input = (cov_summary if cov_summary is not None else
                             cov_fields.permute(0, 3, 1, 2, 4).reshape(n, l, -1))
                remote_covariates = self.remote_covariates(cov_input)
                remote_covariates = torch.where(cov_available.any(dim=(1, 2))[..., None], remote_covariates, 0)
            tokens = []
            for gi, channels in enumerate(groups):
                f = fields[:, :, channels]
                remote_input = (f.permute(0, 3, 1, 2, 4).reshape(n, l, -1) if config['encoder'] == 'mlp'
                                else f.permute(0, 3, 2, 4, 1).reshape(n * l, len(channels) * fpc, p))
                token = self.remote(remote_input).reshape(n, l, -1)
                token = token + self.remote_identity.weight[gi]
                if remote_covariates is not None:
                    token = token + remote_covariates
                if geo_features:
                    token = token + self.remote_geo(torch.cat(geo_features, -1))
                tokens.append(token)
            remote = torch.stack(tokens, 2)  # N,L,group,W
            observed = torch.stack([valid[:, :, channels].any(dim=(1, 2)) for channels in groups], -1)
            if k:
                observed = observed | cov_available.any(dim=(1, 2))[..., None]
            if scope == 'joint_location_target':
                remote = self.spatial(remote.reshape(n, l * len(groups), -1), observed.reshape(n, -1)).reshape_as(remote)
            else:
                remote = self.spatial(remote.permute(0, 2, 1, 3).reshape(n * len(groups), l, -1),
                                      observed.permute(0, 2, 1).reshape(n * len(groups), l)).reshape(n, len(groups), l, -1).permute(0, 2, 1, 3)
            h = h + remote[:, :, config['target_remote']]
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
            return torch.cat(outputs, -2)[..., [order.index(c) for c in range(targets)], :]
        delta = decode(self.output_heads)
        if config['heads'] == 'state_us':
            us = decode(self.us_output_heads)
            is_us = torch.tensor([loc == 'US' for loc in locations], device=x.device)
            delta = torch.where(is_us[None, None, None, :, None, None], us, delta)
        if config['us_error'] == 'shared_factor':
            if national_z is None:
                national_z = torch.randn(delta.shape[0], n, targets, device=x.device, dtype=delta.dtype)
            if national_z.shape != (delta.shape[0], n, targets):
                raise ValueError('National noise must have shape [members, episodes, channels]')
            delta = delta + national_z[:, :, None, None, :, None] * F.softplus(self.national_scale)[None, None, None, None, :, None]
        delta = delta.squeeze(-1).permute(0, 1, 2, 4, 3)
        target_values = values[:, :, config['target_input_indices']]
        target_mask = mask[:, :, config['target_input_indices']]
        idx = (target_mask * torch.arange(1, p + 1, device=x.device)[None, :, None, None]).argmax(1)
        anchor = target_values.gather(1, idx[:, None]).squeeze(1)
        anchor = torch.where(target_mask.any(1), anchor, anchor.new_full((), .01))
        # Per-horizon anchor [N,H,C,L]. Damped growth (2026-10-06): add the latest
        # two-week slope in model space, damped by 1/2 per week ahead (1, 1.5, 1.75, 1.875);
        # only where the newest and two-weeks-earlier context cells are both observed.
        anchor = anchor[:, None].expand(-1, len(config['horizons']), -1, -1)
        if config.get('growth_anchor') and p >= 3:
            target_valid = valid[:, :, config['target_input_indices']]
            ok = target_valid[:, -1] & target_valid[:, -3]
            slope = torch.where(ok, (target_values[:, -1] - target_values[:, -3]) / 2, torch.zeros_like(target_values[:, -1]))
            damping = x.new_tensor([sum(.5 ** i for i in range(max(0, t))) for t in config['horizons']])
            anchor = anchor + slope[:, None] * damping[None, :, None, None]
        result = torch.zeros_like(delta)
        target_scales = input_scale[config['target_input_indices']]
        target_offsets = offset[config['target_input_indices']] if offset is not None else None
        count_targets = [i for i, unit in enumerate(config['target_units']) if unit == 'count']
        proportion_targets = [i for i, unit in enumerate(config['target_units']) if unit == 'proportion']
        if count_targets:
            counts = horizon_residual(anchor[:, :, count_targets], delta[:, :, :, count_targets])
            counts = counts * target_scales[None, None, None, count_targets]
            result[:, :, :, count_targets] = invert_counts(counts, population, config['count_transform'])
        if proportion_targets:
            if config['ed_transform'] == 'fourth_root':
                root = horizon_residual(anchor[:, :, proportion_targets], delta[:, :, :, proportion_targets])
                proportions = (root * target_scales[None, None, None, proportion_targets]).pow(4).clamp(max=1)
            else:
                if config['ed_transform'] == 'logit':
                    logit = (anchor[:, :, proportion_targets] * target_scales[None, None, proportion_targets]
                             + target_offsets[None, None, proportion_targets])
                else:
                    value = (anchor[:, :, proportion_targets] * target_scales[None, None, proportion_targets]).clamp(*ED_BOUNDS)
                    logit = torch.logit(value)
                proportions = torch.sigmoid(logit[None] + delta[:, :, :, proportion_targets])
            result[:, :, :, proportion_targets] = proportions
        return result


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
        combined = torch.cat(outputs, 3)
        targets = len(self.config['target_names'])
        result = combined.new_zeros((*combined.shape[:3], targets, combined.shape[-1]))
        result[:,:,:,order] = combined
        return result


def checkpoint(model, metadata):
    import json
    metadata = json.loads(json.dumps(metadata))
    if isinstance(model, IndependentBundle):
        return dict(groups=model.groups, components=[checkpoint(m, {}) for m in model.models], metadata=metadata)
    return dict(config=model.config, state_dict={k: v.cpu() for k, v in model.state_dict().items()}, metadata=metadata)


def load_model(saved):
    if 'components' in saved:
        return IndependentBundle([load_model(c) for c in saved['components']], saved['groups'])
    config = dict(saved['config'])
    # Input flags removed on 2026-10-08 with the routes that set them; every pilot-route
    # checkpoint (B3-B7, System2) has both False and loads unchanged.
    for retired in ('supplied_final', 'supplied_estimated'):
        if config.pop(retired, False):
            raise ValueError(f'Checkpoint uses {retired}, removed on 2026-10-08')
    factory = Model
    if config.get('encoder') in ('series_mlp', 'series_mixer'):
        from .series import SeriesModel
        factory = SeriesModel
    model = factory(**config)
    model.load_state_dict(saved['state_dict'])
    return model
