"""B0 formulations with masked Wednesday inputs and paired stochastic B1 stages."""
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .b0 import (B0, ED_BOUNDS, LOCAL_LATENT, TemporalEncoder, SpatialBlock,
                 transform_counts, transform_proportions, invert_counts, modulation_layer, softplus_inverse)
from .architecture import MultiscaleEncoder

MASK_PROBABILITIES = (.50, .25, .15, .10)
MASK_SCENARIOS = ('natural', 'recent', 'gap', 'outage')


def draw_dropout(available, rng, probabilities=MASK_PROBABILITIES, scenario=None):
    """A and returned D are [episode, week, channel, location]; V = A & ~D.

    Recent reports/outages affect all locations; local gaps affect one history.
    D marks naturally available cells only, and is never supplied to the network.
    """
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
            # Half channel outages, half source-family outages.
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


class SampleHead(nn.Module):
    """B0 legacy/residual2 FiLM heads with an explicit member axis on context."""
    def __init__(self, width, latent, decoder='legacy', local=False):
        super().__init__()
        self.kind = decoder
        self.horizon = nn.Linear(1, width)
        self.norm = nn.LayerNorm(width)
        blocks = 1 if decoder == 'legacy' else 2
        self.modulations = nn.ModuleList([modulation_layer(latent, width) for _ in range(blocks)])
        if local:
            self.local_modulations = nn.ModuleList([modulation_layer(LOCAL_LATENT, width) for _ in range(blocks)])
            self.local_scale = nn.Parameter(softplus_inverse(1))
        if decoder == 'residual2':
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(blocks)])
            self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
                                         for _ in range(blocks)])
            self.output = nn.Linear(width, 1)
        else:
            self.output = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        final = self.output if decoder == 'residual2' else self.output[-1]
        nn.init.normal_(final.weight, std=.01)
        nn.init.zeros_(final.bias)

    def forward(self, context, z, offsets, local_z=None):  # context M,N,L,C,W
        h = self.norm(context[:, :, None] + self.horizon(context.new_tensor(offsets)[:, None] / 4)[None, None, :, None, None])
        for i, modulation in enumerate(self.modulations):
            gamma, beta = modulation(z)[:, :, None, None, None].chunk(2, -1)
            if hasattr(self, 'local_modulations'):
                lg, lb = (self.local_modulations[i](local_z) * F.softplus(self.local_scale))[:, :, None, :, None].chunk(2, -1)
                gamma, beta = gamma + lg, beta + lb
            if self.kind == 'legacy':
                h = h * (1 + gamma) + beta
            else:
                h = h + self.blocks[i](self.norms[i](h) * (1 + gamma) + beta)
        return self.output(h).squeeze(-1)  # M,N,H,L,C


class B1(nn.Module):
    def __init__(self, target, populations, locations, lookback=12, width=64, latent=16,
                 input_scale=None, input_offset=None, scale=None, direct=False,
                 encoder='mlp', spatial='none', decoder='legacy', heads='shared', noise='global',
                 head_sharing='shared', count_transform='fourth_root', ed_transform='logit',
                 geography=True, dynamics=True, annual_calendar=True, location_embedding=0, us_error='none',
                 supplied_final=False, parallel_recent=False):
        super().__init__()
        targets = [target] if isinstance(target, int) else list(target)
        if not targets or len(set(targets)) != len(targets) or any(c not in range(6) for c in targets):
            raise ValueError('B1 target must be a channel or a unique group of channels')
        if lookback < 2 or width < 3 or latent < 1 or location_embedding < 0:
            raise ValueError('Invalid B1 dimensions')
        choices = dict(encoder=('mlp', 'conv', 'multiscale_conv'),
            spatial=('none', 'attention', 'pathogen_spatial', 'target_spatial', 'joint_location_target'),
            decoder=('legacy', 'residual2'), heads=('shared', 'state_us'), noise=('global', 'local'),
            head_sharing=('shared', 'pathogen', 'target'), count_transform=('sqrt', 'fourth_root', 'log1p'),
            ed_transform=('linear', 'logit', 'fourth_root'), us_error=('none', 'shared_factor'))
        options = dict(encoder=encoder, spatial=spatial, decoder=decoder, heads=heads, noise=noise,
            head_sharing=head_sharing, count_transform=count_transform, ed_transform=ed_transform,
            geography=geography, dynamics=dynamics, annual_calendar=annual_calendar,
            location_embedding=location_embedding, us_error=us_error)
        for name, values in choices.items():
            if options[name] not in values:
                raise ValueError(f'Invalid B1 {name}: {options[name]}')
        if any(loc not in populations or not np.isfinite(populations[loc]) or populations[loc] <= 0 for loc in locations):
            raise ValueError('Finite positive population required for every location')
        self.targets = targets
        self.config = dict(target=targets, populations=populations, locations=list(locations), lookback=lookback,
            width=width, latent=latent, direct=direct, supplied_final=supplied_final, parallel_recent=parallel_recent, input_scale=input_scale, input_offset=input_offset, scale=scale, **options)
        self.register_buffer('input_scale', B0._per_location(input_scale, 1))
        self.register_buffer('input_offset', B0._per_location(input_offset, 0))
        self.register_buffer('scale', B0._per_location(scale, 1))
        self.register_buffer('population', torch.tensor([populations[loc] for loc in locations], dtype=torch.float32))
        self.register_buffer('geography', torch.tensor([[np.log(populations[loc] / 100000), float(loc == 'US')]
                                                       for loc in locations], dtype=torch.float32))
        if direct or parallel_recent:
            # The finalized-vs-Wednesday control must change inputs, not the
            # predictor: reuse B0's dynamics, decoder and missing-history prior.
            self.direct_model = B0(lookback=lookback, width=width, latent=latent, scale=scale,
                populations=populations, input_scale=input_scale, input_offset=input_offset,
                location_ids=list(locations), supplied_final=supplied_final, parallel_recent=parallel_recent, **options)
            return
        temporal = MultiscaleEncoder if encoder == 'multiscale_conv' else TemporalEncoder
        extra = 3 * annual_calendar + 2 * geography + 24 * dynamics + location_embedding
        self.context = nn.Sequential(nn.Linear((lookback * 18 if encoder == 'mlp' else width) + extra, width),
                                     nn.SiLU(), nn.Linear(width, width))
        self.focal = (nn.Sequential(nn.Linear(lookback * 3, width), nn.SiLU(), nn.Linear(width, width))
                      if encoder == 'mlp' else temporal(3, width))
        if encoder != 'mlp':
            self.temporal_context = temporal(18, width)
        self.source = nn.Embedding(6, width)
        self.anchor_flags = nn.Linear(3, width)
        self.baseline = nn.Linear(width, 1)
        if location_embedding:
            self.location_id = nn.Embedding(len(locations), location_embedding)
        if spatial != 'none':
            self.spatial = SpatialBlock(width)
        if spatial not in ('none', 'attention'):
            remote_channels = 2 if spatial == 'pathogen_spatial' else 1
            self.remote = (nn.Sequential(nn.Linear(lookback * remote_channels * 3, width), nn.SiLU(), nn.Linear(width, width))
                           if encoder == 'mlp' else temporal(remote_channels * 3, width))
            self.remote_identity = nn.Embedding(3 if spatial == 'pathogen_spatial' else 6, width)
            if geography or location_embedding:
                self.remote_geo = nn.Linear(2 * geography + location_embedding, width)
        # Output grouping is separate from independently fitted component grouping.
        groups = ([targets] if head_sharing == 'shared' else
                  [[c for c in targets if c % 3 == p] for p in range(3)] if head_sharing == 'pathogen' else [[c] for c in targets])
        self.output_groups = [[targets.index(c) for c in group] for group in groups if group]
        def heads_for_stage():
            return nn.ModuleList([SampleHead(width, latent, decoder, noise == 'local') for _ in self.output_groups])
        self.future = heads_for_stage()
        if heads == 'state_us':
            self.us_future = heads_for_stage()
        self.recent = heads_for_stage()
        self.corrections = nn.Sequential(nn.Linear(2, width), nn.SiLU(), nn.Linear(width, width))
        if heads == 'state_us':
            self.us_recent = heads_for_stage()
        if us_error == 'shared_factor':
            self.national_scale = nn.Parameter(softplus_inverse(.1).expand(2, len(targets)).clone())

    def normalized(self, raw):
        transformed = torch.cat((transform_counts(raw[:, :, :3], self.population, self.config['count_transform']),
                                 transform_proportions(raw[:, :, 3:], self.config['ed_transform'])), 2)
        return (transformed - self.input_offset) / self.input_scale

    def working_to_native(self, working):  # [..., target, location]
        results = []
        for i, c in enumerate(self.targets):
            value = working[..., i, :]
            if c < 3:
                value = invert_counts(F.softplus(value) * self.input_scale[c], self.population, self.config['count_transform'])
            elif self.config['ed_transform'] == 'fourth_root':
                value = (F.softplus(value) * self.input_scale[c]).pow(4).clamp(max=1)
            else:
                value = value.sigmoid()
            results.append(value)
        return torch.stack(results, -2)

    def encode(self, values, visible, calendar, known_final=None):
        n, p, _, l = values.shape
        config = self.config
        raw = torch.where(visible, values, 0)
        v = torch.where(visible, self.normalized(raw), 0)
        known_final = torch.zeros_like(visible) if known_final is None else known_final & visible
        fields = torch.stack((v, visible.to(v.dtype), known_final.to(v.dtype)), -1)  # N,P,C,L,3
        context = (fields.permute(0, 3, 1, 2, 4).reshape(n, l, -1) if config['encoder'] == 'mlp' else
                   self.temporal_context(fields.permute(0, 3, 2, 4, 1).reshape(n * l, 18, p)).reshape(n, l, -1))
        extras, geo = [], []
        if config['annual_calendar']:
            extras.append(calendar[:, None].expand(-1, l, -1))
        if config['geography']:
            geo.append(self.geography[None].expand(n, -1, -1))
        if config['location_embedding']:
            geo.append(self.location_id.weight[None].expand(n, -1, -1))
        extras.extend(geo)
        if config['dynamics']:
            valid = visible[:, -1] & visible[:, -2]
            slope = (v[:, -1] - v[:, -2]) * valid
            pv = visible[:, -2] & visible[:, -3] if p >= 3 else torch.zeros_like(valid)
            prior = (v[:, -2] - v[:, -3]) * pv if p >= 3 else torch.zeros_like(slope)
            av = valid & pv
            acceleration = (slope - prior) * av
            extras.append(torch.stack((slope, acceleration, valid, av), -1).permute(0, 2, 1, 3).reshape(n, l, 24))
        context = self.context(torch.cat([context, *extras], -1))
        if config['spatial'] == 'attention':
            context = self.spatial(context)
        focal = (self.focal(fields.permute(0, 3, 2, 1, 4).reshape(n, l, 6, p * 3)) if config['encoder'] == 'mlp' else
                 self.focal(fields.permute(0, 3, 2, 4, 1).reshape(n * l * 6, 3, p)).reshape(n, l, 6, -1))
        h = context[:, :, None] + focal + self.source.weight[None, None]
        if config['spatial'] not in ('none', 'attention'):
            groups = [[0, 3], [1, 4], [2, 5]] if config['spatial'] == 'pathogen_spatial' else [[c] for c in range(6)]
            tokens = []
            for g, channels in enumerate(groups):
                f = fields[:, :, channels]
                inputs = (f.permute(0, 3, 1, 2, 4).reshape(n, l, -1) if config['encoder'] == 'mlp' else
                          f.permute(0, 3, 2, 4, 1).reshape(n * l, len(channels) * 3, p))
                token = self.remote(inputs).reshape(n, l, -1) + self.remote_identity.weight[g]
                if geo:
                    token = token + self.remote_geo(torch.cat(geo, -1))
                tokens.append(token)
            remote = torch.stack(tokens, 2)
            if config['spatial'] == 'joint_location_target':
                remote = self.spatial(remote.reshape(n, l * len(groups), -1)).reshape_as(remote)
            else:
                remote = self.spatial(remote.permute(0, 2, 1, 3).reshape(n * len(groups), l, -1)).reshape(n, len(groups), l, -1).permute(0, 2, 1, 3)
            mapping = [next(g for g, channels in enumerate(groups) if c in channels) for c in range(6)]
            h = h + remote[:, :, mapping]
        h = h[:, :, self.targets]
        focal_valid = visible[:, :, self.targets]
        any_valid = focal_valid.any(1)
        flags = torch.cat((focal_valid[:, -2:].permute(0, 3, 2, 1), any_valid.permute(0, 2, 1)[..., None]), -1)
        h = h + self.anchor_flags(flags.to(v.dtype))
        working = []
        for c in self.targets:
            if c < 3 or config['ed_transform'] == 'fourth_root':
                positive = v[:, :, c].clamp_min(.001)
                w = positive + torch.log(-torch.expm1(-positive))
            elif config['ed_transform'] == 'logit':
                w = v[:, :, c] * self.input_scale[c] + self.input_offset[c]
            else:
                w = torch.logit((v[:, :, c] * self.input_scale[c]).clamp(*ED_BOUNDS))
            working.append(w)
        working = torch.stack(working, 2)
        idx = (focal_valid * torch.arange(1, p + 1, device=v.device)[None, :, None, None]).argmax(1)
        last = working.gather(1, idx[:, None]).squeeze(1)
        last = torch.where(any_valid, last, self.baseline(h).squeeze(-1).permute(0, 2, 1))
        anchors = torch.where(focal_valid[:, -2:], working[:, -2:], last[:, None])
        return h, anchors, last

    def decode(self, stage, context, noise, offsets):
        z, local_z, national_z = noise
        def apply(heads):
            outputs = [head(context[:, :, :, group], z, offsets, local_z) for head, group in zip(heads, self.output_groups)]
            order = [c for group in self.output_groups for c in group]
            return torch.cat(outputs, -1)[..., [order.index(c) for c in range(len(self.targets))]]
        delta = apply(getattr(self, stage))
        if self.config['heads'] == 'state_us':
            national = apply(getattr(self, 'us_' + stage))
            is_us = self.geography[:, 1].bool()
            delta = torch.where(is_us[None, None, None, :, None], national, delta)
        delta = delta.permute(0, 1, 2, 4, 3)
        if self.config['us_error'] == 'shared_factor':
            index = 0 if stage == 'recent' else 1
            delta = delta + national_z[:, :, None, :, None] * F.softplus(self.national_scale[index])[None, None, None, :, None]
        return delta

    def forecast_from_recent(self, h, recent_working, z_future, local_future=None, national_future=None):
        """Every forecast target receives only its own paired recent corrections."""
        inputs = []
        for i, c in enumerate(self.targets):
            w = recent_working[..., i, :]
            if c < 3 or self.config['ed_transform'] == 'fourth_root':
                value = F.softplus(w)
            elif self.config['ed_transform'] == 'logit':
                value = (w - self.input_offset[c]) / self.input_scale[c]
            else:
                value = w.sigmoid() / self.input_scale[c]
            inputs.append(value)
        normalized = torch.stack(inputs, 3)  # M,N,2,C,L
        context = h[None] + self.corrections(normalized.permute(0, 1, 4, 3, 2))
        future = recent_working[:, :, -1:] + self.decode('future', context,
            (z_future, local_future, national_future), (1, 2, 3, 4))
        return self.working_to_native(future)

    def draw_noise(self, members, episodes, generator=None):
        """CPU-generator draws move to model device; validation freezes every term."""
        device = self.population.device
        def draw(shape):
            return torch.randn(shape, generator=generator, device='cpu' if generator is not None else device).to(device)
        shape = (members, episodes)
        return dict(z=draw((*shape, self.config['latent'])),
            local=draw((*shape, len(self.config['locations']), LOCAL_LATENT)) if self.config['noise'] == 'local' else None,
            national=draw((*shape, 6 if self.config['direct'] else len(self.targets))) if self.config['us_error'] == 'shared_factor' else None)

    def forward(self, values, available, calendar, members=128, dropout=None, z_recent=None, z_future=None,
                local_recent=None, local_future=None, national_recent=None, national_future=None, known_final=None):
        """Native samples [member, episode, output week, component target, location]."""
        expected = (values.shape[0], self.config['lookback'], 6, len(self.config['locations']))
        if tuple(values.shape) != expected or available.shape != values.shape:
            raise ValueError(f'B1 expects values/availability {expected}')
        if dropout is not None and dropout.shape != available.shape:
            raise ValueError('Artificial dropout must have the same shape as availability')
        visible = available.bool() if dropout is None else available.bool() & ~dropout.bool()
        if known_final is not None and known_final.shape != available.shape:
            raise ValueError('Known-final flags must have the same shape as availability')
        known_final = torch.zeros_like(visible) if known_final is None else known_final.bool() & visible
        if self.config['direct'] or self.config['parallel_recent']:
            x = torch.stack((values, visible.to(values.dtype), known_final.to(values.dtype))
                            if self.config['supplied_final'] else (values, visible.to(values.dtype)), dim=3)
            result = self.direct_model(x, calendar, members=members, z=z_future,
                locations=self.config['locations'], local_z=local_future,
                national_z=national_future)[:, :, :, self.targets]
            if self.config['parallel_recent']:
                recent = torch.where(known_final[:, -2:, self.targets][None],
                                     values[:, -2:, self.targets][None], result[:, :, :2])
                result = torch.cat((recent, result[:, :, 2:]), dim=2)
            return result
        h, anchors, last = self.encode(values, visible, calendar, known_final)
        if z_recent is not None:
            members = z_recent.shape[0]
        elif z_future is not None:
            members = z_future.shape[0]
        def complete(z, local, national):
            required = (z is None or (self.config['noise'] == 'local' and local is None)
                        or (self.config['us_error'] == 'shared_factor' and national is None))
            noise = self.draw_noise(members, len(values)) if required else {}
            return (noise.get('z') if z is None else z, noise.get('local') if local is None else local,
                    noise.get('national') if national is None else national)
        future_noise = complete(z_future, local_future, national_future)
        recent_noise = complete(z_recent, local_recent, national_recent)
        if recent_noise[0].shape != future_noise[0].shape:
            raise ValueError('Recent and future noise must have matching member/episode axes')
        recent = anchors[None] + self.decode('recent', h[None], recent_noise, (-1, 0))
        final_recent = known_final[:, -2:, self.targets][None]
        recent = torch.where(final_recent, anchors[None], recent)
        future = self.forecast_from_recent(h, recent, *future_noise)
        # Return exact native finals, including genuine zeros and ED endpoints.
        native_recent = torch.where(final_recent, values[:, -2:, self.targets][None], self.working_to_native(recent))
        return torch.cat((native_recent, future), dim=2)
