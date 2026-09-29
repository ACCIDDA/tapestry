"""Shared tensor preparation, fitting and scoring. No job management or pipeline orchestration."""
import csv
import json

import numpy as np
import torch

from tapestry.dataset.build import CHANNELS, covariate_names_for
from tapestry.dataset.episodes import calendar
from tapestry.evaluation.quantiles import LEVELS
from tapestry.model.network import Model, fair_crps_cells, draw_dropout, IndependentBundle
from tapestry.model.objective import LOSS_WEIGHTS, loss_cell_weights, loss_scales, LOSS_DEFINITION
from tapestry.model.pipeline import HistorySamples, CovariateHistory, forecast_histories, NowcastForecast
from .provenance import save

GROUPS = {'all': [list(range(len(CHANNELS)))], 'pathogen': [[0, 3], [1, 4], [2, 5]],
          'target': [[i] for i in range(len(CHANNELS))]}
EVAL_CHUNK = 32


def populations(path, locations):
    values = {}
    with open(path) as stream:
        for row in csv.DictReader(stream):
            loc, value = row.get('abbreviation') or row['location'], float(row['population'])
            if loc in values or not np.isfinite(value) or value <= 0:
                raise ValueError(f'Invalid or duplicate population for {loc}')
            values[loc] = value
    return {loc: values[loc] for loc in locations}


def to_tensors(batch, device):
    values = torch.as_tensor(np.stack([e['values'] for e in batch]), device=device)
    available = torch.as_tensor(np.stack([e['available'] for e in batch]), device=device)
    known_final = torch.as_tensor(np.stack([e['known_final'] for e in batch]), device=device)
    y = torch.as_tensor(np.stack([e['target_values'] for e in batch]), device=device)
    y_mask = torch.as_tensor(np.stack([e['target_available'] for e in batch]), device=device)
    cal = torch.as_tensor(calendar([e['context_dates'][-1] for e in batch]), device=device)
    covariates = (torch.as_tensor(np.stack([e['covariates'] for e in batch]), device=device)
                  if 'covariates' in batch[0] else None)
    return values, available, known_final, y, y_mask, cal, covariates


def covariate_scales(batch):
    """Per-covariate/location mean and SD from observed fitting cells only."""
    c = np.stack([e['covariates'] for e in batch])
    values, available = c[..., 0, :], c[..., 1, :].astype(bool)
    support = available.sum(axis=(0, 1))
    total = np.where(available, values, 0.).sum(axis=(0, 1))
    offset = np.divide(total, support, out=np.zeros_like(total), where=support > 0)
    squared = np.where(available, (values - offset[None, None]) ** 2, 0.).sum(axis=(0, 1))
    variance = np.divide(squared, support, out=np.zeros_like(squared), where=support > 0)
    scale = np.sqrt(variance)
    scale = np.where(scale > 1e-6, scale, 1.)
    return offset.tolist(), scale.tolist(), (support > 0).tolist()


def model_options(batch, scenario, pop):
    covariate_names = covariate_names_for(scenario.covariate_set)
    options = dict(populations=pop, location_ids=list(batch[0]['locations']), lookback=scenario.lookback,
                   width=scenario.width, latent=scenario.latent, covariate_names=list(covariate_names),
                   **scenario.model_options())
    if scenario.input_normalization == 'b0':
        normalization = [dict(e, X=np.stack((e['values'], e['available']), axis=2)) for e in batch]
        options.update(input_scales(normalization, scenario.count_transform, scenario.ed_transform, pop))
    if 'history_samples' in batch[0]:
        options['supplied_estimated'] = True
    if covariate_names:
        offset, scale, trained = covariate_scales(batch)
        options.update(covariate_offset=offset, covariate_scale=scale, covariate_trained=trained)
    return options


def covariate_history(batch, values, names):
    if values is None:
        return None
    return CovariateHistory(values, tuple(names), tuple(batch[0]['locations']),
                            tuple(e['context_dates'] for e in batch), tuple(e['issuance'] for e in batch))


def training_samples(model, episodes_, ids, values, available, known_final, cal, cov, members, vintaged):
    if 'history_samples' not in episodes_[0]:
        return model(values=values, available=available, known_final=known_final, calendar=cal,
                     covariates=cov, locations=list(episodes_[0]['locations']), members=members, vintaged=vintaged)
    batch = [episodes_[i] for i in ids.tolist()]
    bank = torch.as_tensor(np.stack([e['history_samples'] for e in batch]), device=values.device)
    # Independent choices for each predictive member and episode; never shuffle cells.
    choices = torch.randint(bank.shape[1], (members, len(batch)), device=values.device)
    draws = bank[torch.arange(len(batch), device=values.device)[None], choices]
    estimated = torch.as_tensor(np.stack([e['estimated'] for e in batch]), device=values.device)
    histories = HistorySamples(draws, available, known_final & available, estimated & available,
                               tuple(episodes_[0]['locations']), tuple(e['context_dates'] for e in batch),
                               tuple(e['issuance'] for e in batch))
    return forecast_histories(model, histories, cal, covariate_history(batch, cov, model.config['covariate_names']))


def fit_component(train, validation, channels, component, options, scenario, seed, device, epochs=None):
    seed = seed + 10000 * component
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    selecting = validation is not None
    budget = scenario.epochs if epochs is None else epochs
    model = Model(horizons=scenario.horizons, scale=loss_scales(unique_truth(train)), **options).to(device)
    weights_by_channel = LOSS_WEIGHTS[scenario.loss_weights]
    values, available, known_final, y, y_mask, cal, cov = to_tensors(train, device)
    weights = torch.as_tensor(loss_cell_weights(train, weights_by_channel), device=device)[:, :, channels]
    if selecting:
        vvalues, vavailable, vknown_final, vy, vy_mask, vcal, vcov = to_tensors(validation, device)
        vweights = torch.as_tensor(loss_cell_weights(validation, weights_by_channel), device=device)[:, :, channels]
        if not float(vweights.sum()) > 0:
            raise ValueError(f'Component {component} (channels {channels}) has no weighted validation labels; '
                             'early stopping cannot select an epoch')
    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(budget):
        dropout = None
        a = available.cpu().numpy()
        if scenario.mask_rate and scenario.input_mode != 'scheduled_final':
            dropout = torch.as_tensor(draw_dropout(a, rng, scenario.mask_probabilities), device=device)
        epoch_cov = cov
        if scenario.mask_rate and scenario.input_mode == 'scheduled_final':
            dropout = torch.zeros_like(available)
            epoch_cov = None if cov is None else cov.clone()
            state_ids = [i for i, loc in enumerate(train[0]['locations']) if loc != 'US']
            for n in range(len(train)):
                if rng.random() >= scenario.mask_rate:
                    continue
                loc = int(rng.choice(state_ids))
                whole = rng.random() < .25
                recent = int(rng.integers(1, 3))
                weeks = slice(None) if whole else slice(-recent, None)
                dropout[n, weeks, :, loc] = True
                if epoch_cov is not None:
                    for k, name in enumerate(covariate_names_for(scenario.covariate_set)):
                        lag = int(name in ('ilinet_ili', 'clinical_lab_flu_pct_positive', 'flusurv_flu_rate'))
                        window = slice(None) if whole else slice(-recent - lag, -lag if lag else None)
                        epoch_cov[n, window, k, :, loc] = 0
        model.train()
        total = 0.
        for ids in torch.randperm(len(train), device=device).split(scenario.batch_size):
            optimizer.zero_grad()
            visible = available[ids] if dropout is None else available[ids] & ~dropout[ids]
            samples = training_samples(model, train, ids, values[ids], visible, known_final[ids], cal[ids],
                                       None if epoch_cov is None else epoch_cov[ids], scenario.members,
                                       scenario.input_mode == 'vintaged')
            score = fair_crps_cells(samples[:, :, :, channels], y[ids][:, :, channels], y_mask[ids][:, :, channels])
            loss = (weights[ids] * score / model.scale[channels]).sum() * len(train) / len(ids)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids) / len(train)
        val = None
        if selecting:
            model.eval()
            val = 0.
            with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device)] if str(device).startswith("cuda") else []):
                torch.manual_seed(seed + 900000)
                for ids in torch.arange(len(validation), device=device).split(scenario.batch_size):
                    samples = training_samples(model, validation, ids, vvalues[ids], vavailable[ids],
                                               vknown_final[ids], vcal[ids], None if vcov is None else vcov[ids],
                                               scenario.validation_members, scenario.input_mode == 'vintaged')
                    score = fair_crps_cells(samples[:, :, :, channels], vy[ids][:, :, channels], vy_mask[ids][:, :, channels])
                    val += float((vweights[ids] * score / model.scale[channels]).sum())
            if not np.isfinite(val):
                raise ValueError(f'Nonfinite validation loss at epoch {epoch + 1}, component {component}')
            if val < best:
                best, best_epoch, best_state = val, epoch + 1, {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append(dict(epoch=epoch + 1, loss=total, validation_loss=val))
        print(json.dumps(dict(component=component, phase='select' if selecting else 'refit', **history[-1])), flush=True)
        if selecting and scenario.patience and epoch + 1 - best_epoch >= scenario.patience:
            break
    if selecting and scenario.patience and best_state is not None:
        model.load_state_dict(best_state)
    record = dict(channels=channels, best_epoch=best_epoch,
                 selected_epoch=(best_epoch if scenario.patience else scenario.epochs) if selecting else budget,
                 phase='select' if selecting else 'refit', epochs=budget, history=history,
                 fitting_episodes=len(train), validation_episodes=len(validation) if selecting else 0)
    return model.cpu(), record


def unique_truth(episodes_):
    """[dates, C, value/available, L]: the truth of every calendar date, pooled across episodes.

    Only truth enters (loss scales, 2026-09-22): target cells, and context cells flagged
    known-final. The as-of context values of vintaged episodes (not known-final) are
    skipped; before this fix the first value seen per date was used, which in vintaged
    mode was usually the earliest as-of value. Finalized episodes hold truth only, so
    their scales are unchanged."""
    by_date = {}

    def add(values, available, day):
        stored, seen = by_date.setdefault(day, (np.zeros_like(values), np.zeros(values.shape, bool)))
        new = available & ~seen
        stored[new], seen[new] = values[new], True

    for e in episodes_:
        for values, available, day in zip(e['target_values'], e['target_available'], e['target_dates']):
            add(values, available, day)
        for values, available, final, day in zip(e['values'], e['available'], e['known_final'], e['context_dates']):
            add(values, available & final, day)
    panel = np.zeros((len(by_date), len(CHANNELS), 2, len(episodes_[0]['locations'])), np.float32)
    for i, (values, seen) in enumerate(by_date.values()):
        panel[i, :, 0] = values
        panel[i, :, 1] = seen
    return panel


def fit_models(train, validation, full_train, scenario, seed, device, pop):
    """Select epochs if requested, then fit independent channel groups on full_train."""
    groups = GROUPS[scenario.fit_partition]
    models, records = [], []
    for i, channels in enumerate(groups):
        selected = scenario.epochs
        if validation is not None:
            _, record = fit_component(train, validation, channels, i,
                                      model_options(train, scenario, pop), scenario, seed, device)
            records.append(record)
            selected = record['selected_epoch']
        if selected < 1:
            raise ValueError(f'Selected {selected} epochs for component {i}; refusing to refit with no training')
        model, record = fit_component(full_train, None, channels, i, model_options(full_train, scenario, pop),
                                      scenario, seed, device, epochs=selected)
        models.append(model)
        records.append(record)
    model = models[0] if scenario.fit_partition == 'all' else IndependentBundle(models, groups)
    return model, records


def evaluate(model, eps, eval_members, device, output):
    """Held-out quantiles from `eval_members` draws (in chunks of EVAL_CHUNK) per episode.

    Admission quantiles (channels 0-2) are rounded to integers (counts); ED proportions
    are not rounded."""
    model.eval()
    quantiles, truths, masks, crps = [], [], [], []
    nowcasting = max(model.config['horizons']) <= 0
    for e in eps:
        values = torch.as_tensor(e['values'][None], device=device)
        available = torch.as_tensor(e['available'][None], device=device)
        known_final = torch.as_tensor(e['known_final'][None], device=device)
        cal = torch.as_tensor(calendar([e['context_dates'][-1]]), device=device)
        cov = torch.as_tensor(e['covariates'][None], device=device) if 'covariates' in e else None
        metadata = {}
        if isinstance(model, NowcastForecast):
            metadata = dict(context_dates=(e['context_dates'],), issuances=(e['issuance'],))
            cov = covariate_history([e], cov, model.config['covariate_names'])
        with torch.no_grad():
            samples = torch.cat([model(values=values, available=available, calendar=cal,
                                       members=min(EVAL_CHUNK, eval_members - j), known_final=known_final,
                                       covariates=cov, locations=list(e['locations']), vintaged=True, **metadata).cpu()
                                 for j in range(0, eval_members, EVAL_CHUNK)], dim=0).numpy()[:, 0]
        if nowcasting:
            crps.append(fair_crps_cells(torch.as_tensor(samples[:, None]), torch.as_tensor(e['target_values'][None]),
                                        torch.as_tensor(e['target_available'][None]))[0].numpy())
        q = np.quantile(samples, LEVELS, axis=0)
        q[:, :, :3] = np.floor(q[:, :, :3] + .5)
        quantiles.append(q)
        truths.append(e['target_values'])
        masks.append(e['target_available'])
    q = np.stack(quantiles, axis=1)
    y, mask = np.stack(truths), np.stack(masks)
    np.savez_compressed(output / 'forecasts.npz', quantiles=q, quantile_levels=LEVELS,
                        truth=y, mask=mask, context_end=[e['context_dates'][-1] for e in eps],
                        target_dates=[e['target_dates'] for e in eps], locations=eps[0]['locations'])
    if nowcasting:
        scales = (model.models[0].scale if isinstance(model, IndependentBundle) else model.scale).detach().cpu().numpy()
        score = float((np.stack(crps) * loss_cell_weights(eps) / scales).sum())
        save(output / 'nowcast_scores.json', dict(normalized_crps=score, definition=LOSS_DEFINITION))
    return [dict(context_end=e['context_dates'][-1]) for e in eps]


def input_scales(episodes, transform, ed_transform, populations):
    """Training-only scales (and logit centers) in each channel's model space."""
    import torch
    from tapestry.model.network import transform_counts, transform_proportions
    locations = episodes[0]['locations']
    # Count each context date once; do not overweight dates occurring in more windows.
    # The caller has already masked all observations outside the fitting partition.
    by_date = {}
    for episode in episodes:
        if episode['locations'] != locations:
            raise ValueError('Training episodes must share location order')
        for day, row in zip(episode['context_dates'], episode['X']):
            if day not in by_date:
                by_date[day] = row.copy()
            else:
                valid_row = row[:, 1].astype(bool)
                by_date[day][:, 0][valid_row] = row[:, 0][valid_row]
                by_date[day][:, 1][valid_row] = 1
    panel = torch.as_tensor(np.stack(list(by_date.values())))  # date, channel, (value, mask), location
    valid = panel[:, :, 1].bool()
    values = torch.where(valid, panel[:, :, 0], 0)
    population = torch.tensor([populations[loc] for loc in locations]) if populations else None
    transformed = torch.cat((transform_counts(values[:, :3], population, transform),
                             transform_proportions(values[:, 3:], ed_transform)), 1).numpy()
    valid = valid.numpy()
    # One scale per channel AND location. Pooling locations set a single scale from
    # state-sized counts, so the US entered the model ~40x too large and its intervals
    # collapsed; each location is now normalized by its own history. Falls back to the
    # pooled value where a location has no observations of a channel.
    scales, offsets = [], []
    for c in range(6):
        pooled = transformed[:, c][valid[:, c]]
        logit_channel = c >= 3 and ed_transform == 'logit'
        floor = .1 if logit_channel else (1 if c < 3 and transform == 'raw' else .001)
        if logit_channel:
            fallback = max(float(pooled.std()) if pooled.size else 0, floor)
            pooled_offset = float(pooled.mean()) if pooled.size else 0.
        else:
            fallback = max(float(np.quantile(pooled, .95)) if pooled.size else 0, floor)
            pooled_offset = 0.
        channel_scales, channel_offsets = [], []
        for li in range(len(locations)):
            observed = transformed[:, c, li][valid[:, c, li]]
            if not observed.size:
                channel_scales.append(fallback)
                channel_offsets.append(pooled_offset)
            elif logit_channel:
                channel_offsets.append(float(observed.mean()))
                channel_scales.append(max(float(observed.std()), floor))
            else:
                channel_offsets.append(0.)
                channel_scales.append(max(float(np.quantile(observed, .95)), floor))
        scales.append(channel_scales)
        offsets.append(channel_offsets)
    result = dict(input_scale=scales)
    if ed_transform == 'logit':
        result['input_offset'] = offsets
    return result
