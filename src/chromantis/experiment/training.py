"""Tensor preparation, the fitting loop and forecast evaluation. No job management (planner) or
training-history treatment (fit.py)."""
import csv
import json

import numpy as np
import torch

from chromantis.dataset.build import CHANNELS, covariate_names_for
from chromantis.dataset.episodes import calendar
from chromantis.evaluation.quantiles import LEVELS
from chromantis.model.network import Model, fair_crps_cells, IndependentBundle
from chromantis.model.objective import LOSS_WEIGHTS, loss_cell_weights, loss_scales

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
    y = torch.as_tensor(np.stack([e['target_values'] for e in batch]), device=device)
    y_mask = torch.as_tensor(np.stack([e['target_available'] for e in batch]), device=device)
    cal = torch.as_tensor(calendar([e['context_dates'][-1] for e in batch]), device=device)
    covariates = (torch.as_tensor(np.stack([e['covariates'] for e in batch]), device=device)
                  if 'covariates' in batch[0] else None)
    return values, available, y, y_mask, cal, covariates


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
    if covariate_names:
        offset, scale, trained = covariate_scales(batch)
        options.update(covariate_offset=offset, covariate_scale=scale, covariate_trained=trained)
    return options


def objective_weights(batch, scenario, channels):
    if not scenario.reconstruction_labels:
        return loss_cell_weights(batch, LOSS_WEIGHTS[scenario.loss_weights])[:, :, channels]
    # Normalize the two objectives separately, so adding reconstruction ages does
    # not silently reduce the forecast objective or change its season weights.
    result = np.zeros_like(np.stack([e['target_values'] for e in batch]))
    h = np.asarray(scenario.horizons)
    for use, factor in ((h > 0, 1.), (h <= 0, scenario.joint_weight)):
        subset = [dict(e, Y=e['Y'][use], target_dates=tuple(np.asarray(e['target_dates'])[use])) for e in batch]
        result[:, use] = factor * loss_cell_weights(subset, LOSS_WEIGHTS[scenario.loss_weights])
    return result[:, :, channels]


def prediction_loss(model, predictions, y, mask):
    if model.config.get('direct_quantiles'):
        from chromantis.model.series import quantile_loss
        return quantile_loss(predictions, y, mask)
    return fair_crps_cells(predictions, y, mask)


def sum_wis_weights(episodes, horizons):
    """One complete four-week admission total per issuance; equal seasons/geographies."""
    from chromantis.dataset.cv import season
    future = np.flatnonzero(np.asarray(horizons) > 0)
    if len(future) != 4:
        raise ValueError('Sum WIS needs exactly four future weeks')
    totals = []
    for e in episodes:
        y = e['Y'][future]
        valid = y[:, :, 1].all(0)
        dates = [e['target_dates'][i] for i in future]
        if len({season(d) for d in dates}) != 1:
            valid[:] = False
        total = np.stack((y[:, :, 0].sum(0), valid), axis=1)[None]
        totals.append(dict(e, Y=total, target_dates=(dates[-1],)))
    return loss_cell_weights(totals, [1, 0, 0, 0, 0, 0])[:, :, :1]


def four_week_sum_wis(samples, truth, mask, horizons):
    """Sum intact sample trajectories, then take quantiles (never sum quantiles)."""
    from chromantis.model.series import quantile_loss
    future = np.flatnonzero(np.asarray(horizons) > 0).tolist()
    draws = samples[:, :, future, :1].sum(2, keepdim=True)
    y = truth[:, future, :1].sum(1, keepdim=True)
    valid = mask[:, future, :1].all(1, keepdim=True)
    q = torch.quantile(draws, draws.new_tensor(LEVELS), dim=0)
    return quantile_loss(q, y, valid)


def fit_component(train, validation, channels, component, options, scenario, seed, device, epochs=None, augmenter=None):
    seed = seed + 10000 * component
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    augmentation_rng = np.random.default_rng(seed + 700000)
    selecting = validation is not None
    budget = scenario.epochs if epochs is None else epochs
    if augmenter is not None and augmenter.transport_missingness and 'covariate_trained' in options:
        options = dict(options)
        possible = (~augmenter.cov_support | augmenter.cov_visible).any(axis=(0, 1))
        options['covariate_trained'] = (np.asarray(options['covariate_trained']) & possible).tolist()
    factory = Model
    if scenario.encoder in ('series_mlp', 'series_mixer'):
        from chromantis.model.series import SeriesModel
        factory = SeriesModel
        options = dict(options, growth_anchor=scenario.growth_anchor)
    model = factory(horizons=scenario.horizons, scale=loss_scales(unique_truth(train)), **options).to(device)
    ili = None
    if scenario.ili_training != 'none' and scenario.encoder == 'mlp':
        pretrain_historical_panel(model, train, scenario, channels, seed, device)
        torch.manual_seed(seed)  # Match modern minibatch ordering after transfer initialization.
    elif scenario.ili_training != 'none':
        from chromantis.model.series import HistoricalILI
        ili = HistoricalILI(scenario.ili_path, scenario.lookback, device, units=scenario.ili_units, steps=scenario.ili_steps)
        if scenario.ili_training == 'pretrain':
            ili.pretrain(model, seed)
            torch.manual_seed(seed)  # Match modern minibatch ordering after transfer initialization.
    values, available, y, y_mask, cal, cov = to_tensors(train, device)
    weights = torch.as_tensor(objective_weights(train, scenario, channels), device=device)
    total_weights = torch.as_tensor(sum_wis_weights(train, scenario.horizons), device=device) if scenario.sum_wis_weight and 0 in channels else None
    if selecting:
        vvalues, vavailable, vy, vy_mask, vcal, vcov = to_tensors(validation, device)
        vweights = torch.as_tensor(objective_weights(validation, scenario, channels), device=device)
        if scenario.reconstruction_labels:
            # Select epochs on finalized future labels; reconstruction is auxiliary.
            vweights[:, np.asarray(scenario.horizons) <= 0] = 0
        if not float(vweights.sum()) > 0:
            raise ValueError(f'Component {component} (channels {channels}) has no weighted validation labels; '
                             'early stopping cannot select an epoch')
    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)
    best, best_state, best_epoch = float('inf'), None, 0
    history = []
    for epoch in range(budget):
        model.train()
        total = 0.
        # Large members x batch products are split into gradient-accumulation chunks of at most
        # 1024 member-episodes (2026-10-06, GPU memory); the summed gradient equals the full
        # minibatch gradient. Default 128 members x 8 episodes is one chunk (unchanged).
        chunk = max(1, 1024 // scenario.members)
        for batch_ids in torch.randperm(len(train), device=device).split(scenario.batch_size):
            optimizer.zero_grad()
            batch_loss = 0.
            chunks = batch_ids.split(chunk)
            first = chunks[0]
            for ids in chunks:
                scale = len(train) / len(batch_ids)
                batch_values, batch_available = values[ids], available[ids]
                batch_cov = None if cov is None else cov[ids]
                if augmenter is not None:
                    batch = augmenter.batch([train[i] for i in ids.tolist()], augmentation_rng)
                    batch_values, batch_available, _, _, _, batch_cov = to_tensors(batch, device)
                    if batch_cov is not None and cov is not None:
                        cov_visible = batch_cov[..., 1, :].bool() & cov[ids][..., 1, :].bool()
                        batch_cov[..., 0, :] *= cov_visible
                        batch_cov[..., 1, :] = cov_visible
                if scenario.covariate_dropout and batch_cov is not None:
                    # Whole-episode covariate outage (e.g. Kinsa feed missing), values and masks.
                    hidden = torch.as_tensor(rng.random(len(ids)) < scenario.covariate_dropout, device=device)
                    batch_cov = torch.where(hidden[:, None, None, None, None], torch.zeros_like(batch_cov), batch_cov)
                samples = model(values=batch_values, available=batch_available, calendar=cal[ids],
                                covariates=batch_cov, locations=list(train[0]['locations']), members=scenario.members)
                score = prediction_loss(model, samples[:, :, :, channels], y[ids][:, :, channels], y_mask[ids][:, :, channels])
                loss = (weights[ids] * score / model.scale[channels]).sum() * scale
                admissions = [i for i, c in enumerate(channels) if c < 3]
                if scenario.log_loss_weight and admissions:
                    # Weekly admissions on the log(1 + count) scale, the Hub log score's scale.
                    # Log-unit CRPS/pinball is already relative, so it is not divided by a level scale.
                    chosen = [channels[i] for i in admissions]
                    log_score = prediction_loss(model, torch.log1p(samples[:, :, :, chosen].clamp_min(0)),
                                                torch.log1p(y[ids][:, :, chosen].clamp_min(0)), y_mask[ids][:, :, chosen])
                    loss = loss + scenario.log_loss_weight * (weights[ids][:, :, admissions] * log_score).sum() * scale
                if total_weights is not None:
                    sum_score = four_week_sum_wis(samples, y[ids], y_mask[ids], scenario.horizons)
                    loss = loss + scenario.sum_wis_weight * (total_weights[ids] * sum_score / (4 * model.scale[:1])).sum() * scale
                if ili is not None and scenario.ili_training == 'joint' and ids is first:
                    loss = loss + scenario.ili_weight * ili.loss(model)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                batch_loss += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += batch_loss * len(batch_ids) / len(train)
        val = None
        if selecting:
            model.eval()
            val = 0.
            with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device)] if str(device).startswith("cuda") else []):
                torch.manual_seed(seed + 900000)
                for ids in torch.arange(len(validation), device=device).split(scenario.batch_size):
                    samples = model(values=vvalues[ids], available=vavailable[ids], calendar=vcal[ids],
                                    covariates=None if vcov is None else vcov[ids],
                                    locations=list(validation[0]['locations']), members=scenario.validation_members)
                    score = prediction_loss(model, samples[:, :, :, channels], vy[ids][:, :, channels], vy_mask[ids][:, :, channels])
                    val += float((vweights[ids] * score / model.scale[channels]).sum())
                    admissions = [i for i,c in enumerate(channels) if c < 3]
                    if scenario.log_loss_weight and admissions:
                        chosen=[channels[i] for i in admissions]
                        log_score=prediction_loss(model,torch.log1p(samples[:,:,:,chosen].clamp_min(0)),
                            torch.log1p(vy[ids][:,:,chosen].clamp_min(0)),vy_mask[ids][:,:,chosen])
                        val += scenario.log_loss_weight * float((vweights[ids][:,:,admissions] * log_score).sum())
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


def historical_panel(path, train, scenario):
    """Pre-August-2022 state ILI as pseudo flu admission/ED panel episodes (flu MLP pretraining).

    Each location's ILI is multiplied by (fit-only modern Q95 of that location's flu
    admissions, or flu ED proportion) / (historical Q95 of its ILI), so the pseudo series
    lives in the modern channel's units; ED pseudo-values are clipped to [0, 1]. Only the
    flu admission and flu ED channels exist; COVID/RSV, covariates and the US (no national
    ILI series in the archive) are unavailable. Only the four future horizons are labels.
    This transfers the shape of past epidemics; it is not observed historical admissions."""
    from chromantis.dataset.episodes import episodes
    locations = list(train[0]['locations'])
    with np.load(path) as d:
        ili, dates = d['values'].astype(float), d['dates'].astype(str)
        sources = [str(v).upper() for v in d['locations']]
    if max(dates) >= '2022-08-01':
        raise ValueError('Historical ILI must end before the modern panel')
    truth = unique_truth(train)  # [dates, C, value/available, L], fit-only truth
    targets = np.full((len(dates), len(locations), len(CHANNELS)), np.nan, dtype=np.float32)
    for li, loc in enumerate(locations):
        if loc not in sources:
            continue
        series = ili[:, sources.index(loc)]
        if not np.isfinite(series).sum() or not np.nanquantile(series, .95) > 0:
            continue
        for c in (0, 3):
            observed = truth[:, c, 0, li][truth[:, c, 1, li].astype(bool)]
            if observed.size:
                pseudo = series * np.quantile(observed, .95) / np.nanquantile(series, .95)
                targets[:, li, c] = np.clip(pseudo, 0, 1) if c == 3 else pseudo
    names = covariate_names_for(scenario.covariate_set)
    with np.load('data/processed/panel.npz') as p:
        state_names, national_names = p['covariate_names'], p['covariate_national_names']
    panel = dict(dates=dates, locations=np.array(locations), targets=targets,
                 covariates=np.full((len(dates), len(locations), len(state_names)), np.nan, dtype=np.float32),
                 covariates_national=np.full((len(dates), len(national_names)), np.nan, dtype=np.float32),
                 covariate_names=state_names, covariate_national_names=national_names)
    result = []
    future = np.asarray(scenario.horizons) > 0
    for e in episodes(panel, scenario.lookback, 'scheduled_final', names, horizons=scenario.horizons):
        e['target_available'] = e['target_available'] & future[:, None, None]
        e['target_values'] = np.where(e['target_available'], e['target_values'], 0).astype(np.float32)
        e['Y'] = np.stack((e['target_values'], e['target_available']), axis=2)
        if e['target_available'].any():
            result.append(e)
    return result


def pretrain_historical_panel(model, train, scenario, channels, seed, device):
    """`ili_steps` Adam updates of the modern loss, each on `batch_size` historical pseudo-panel episodes.

    Batch size matches modern training (a 64-episode batch exceeded GPU memory at 256 members, 2026-10-06)."""
    hist = historical_panel(scenario.ili_path, train, scenario)
    weights = torch.as_tensor(loss_cell_weights(hist, LOSS_WEIGHTS[scenario.loss_weights])[:, :, channels], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=scenario.lr, weight_decay=scenario.weight_decay)
    rng = np.random.default_rng(seed + 31000)
    model.train()
    for step in range(scenario.ili_steps):
        batch_ids = torch.as_tensor(rng.choice(len(hist), min(scenario.batch_size, len(hist)), replace=False), device=device)
        optimizer.zero_grad()
        for ids in batch_ids.split(max(1, 1024 // scenario.members)):  # same accumulation as modern training
            values, available, y, y_mask, cal, cov = to_tensors([hist[i] for i in ids.tolist()], device)
            samples = model(values=values, available=available, calendar=cal, covariates=cov,
                            locations=list(hist[0]['locations']), members=scenario.members)
            score = prediction_loss(model, samples[:, :, :, channels], y[:, :, channels], y_mask[:, :, channels])
            loss = (weights[ids] * score / model.scale[channels]).sum() * len(hist) / len(batch_ids)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite historical pretraining loss')
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
    print(json.dumps(dict(phase='historical_panel_pretrain', steps=scenario.ili_steps, episodes=len(hist),
                          last_loss=float(loss.detach()))), flush=True)


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


def fit_models(train, validation, full_train, scenario, seed, device, pop, augmenter=None, selection_augmenter=None,
               keep_selection=False):
    """Select epochs if requested, then fit independent channel groups on full_train.

    `keep_selection=True` also returns the early-stopped inner-fit model, which never
    saw the validation weeks' labels (used to calibrate on those weeks)."""
    groups = [[c for c in group if LOSS_WEIGHTS[scenario.loss_weights][c] > 0] for group in GROUPS[scenario.fit_partition]]
    groups = [g for g in groups if g]
    models, records, selectors = [], [], []
    for i, channels in enumerate(groups):
        selected = scenario.epochs
        if validation is not None:
            selector, record = fit_component(train, validation, channels, i,
                                             model_options(train, scenario, pop), scenario, seed, device, augmenter=selection_augmenter)
            selectors.append(selector)
            records.append(record)
            selected = record['selected_epoch']
        if selected < 1:
            raise ValueError(f'Selected {selected} epochs for component {i}; refusing to refit with no training')
        model, record = fit_component(full_train, None, channels, i, model_options(full_train, scenario, pop),
                                      scenario, seed, device, epochs=selected, augmenter=augmenter)
        models.append(model)
        records.append(record)
    bundle = lambda parts: parts[0] if scenario.fit_partition == 'all' else IndependentBundle(parts, groups)
    model = bundle(models)
    if keep_selection:
        return model, records, bundle(selectors) if selectors else None
    return model, records


def evaluate(model, eps, eval_members, device, output):
    """Held-out quantiles from `eval_members` draws (in chunks of EVAL_CHUNK) per episode.

    Admission quantiles (channels 0-2) are rounded to integers (counts); ED proportions
    are not rounded."""
    model.eval()
    quantiles, truths, masks = [], [], []
    sum_quantiles = []
    for e in eps:
        values = torch.as_tensor(e['values'][None], device=device)
        available = torch.as_tensor(e['available'][None], device=device)
        cal = torch.as_tensor(calendar([e['context_dates'][-1]]), device=device)
        cov = torch.as_tensor(e['covariates'][None], device=device) if 'covariates' in e else None
        initial_rng=torch.get_rng_state()
        initial_cuda=torch.cuda.get_rng_state(device) if str(device).startswith('cuda') else None
        with torch.no_grad():
            if model.config.get('direct_quantiles') and 'history_samples' in e:
                # One quantile forecast per sampled history, then their equal-weight mixture.
                bank = torch.as_tensor(e['history_samples'], device=device)
                m = len(bank)
                repeat = lambda x: None if x is None else x.expand(m, *x.shape[1:])
                each = model(values=bank, available=repeat(available), calendar=repeat(cal),
                             covariates=repeat(cov), locations=list(e['locations'])).cpu().numpy()
                direct = mixture_many(each)
                samples = None
            elif model.config.get('direct_quantiles'):
                direct = model(values=values, available=available, calendar=cal, covariates=cov, locations=list(e['locations']))[:, 0].cpu().numpy()
                samples = None
            elif 'history_samples' in e:
                # One conditional future per intact history (correction_noise > 0).
                bank=e['history_samples']
                if len(bank)!=eval_members:
                    raise ValueError('Evaluation history bank must match eval_members')
                chunks=[]
                for j in range(0,eval_members,EVAL_CHUNK):
                    history=torch.as_tensor(bank[j:j+EVAL_CHUNK],device=device)
                    m=len(history)
                    repeat=lambda x: None if x is None else x.expand(m,*x.shape[1:])
                    chunks.append(model(values=history,available=repeat(available),calendar=repeat(cal),
                        members=1,covariates=repeat(cov),locations=list(e['locations']))[0].cpu())
                samples=torch.cat(chunks).numpy()
            else:
                samples = torch.cat([model(values=values, available=available, calendar=cal,
                                       members=min(EVAL_CHUNK, eval_members - j),
                                       covariates=cov, locations=list(e['locations'])).cpu()
                                 for j in range(0, eval_members, EVAL_CHUNK)], dim=0).numpy()[:, 0]
        q = direct if model.config.get('direct_quantiles') else np.quantile(samples, LEVELS, axis=0)
        if 'mixture_episode' in e:
            other=e['mixture_episode']
            ov=torch.as_tensor(other['values'][None],device=device)
            oa=torch.as_tensor(other['available'][None],device=device)
            with torch.no_grad(), torch.random.fork_rng(devices=[torch.device(device)] if initial_cuda is not None else []):
                torch.set_rng_state(initial_rng)
                if initial_cuda is not None:torch.cuda.set_rng_state(initial_cuda,device)
                if model.config.get('direct_quantiles'):
                    oq=model(values=ov,available=oa,calendar=cal,covariates=cov,locations=list(e['locations']))[:,0].cpu().numpy()
                    q=mixture_quantiles(q,oq)
                else:
                    draws=torch.cat([model(values=ov,available=oa,calendar=cal,members=min(EVAL_CHUNK,eval_members-j),
                        covariates=cov,locations=list(e['locations'])).cpu()
                        for j in range(0,eval_members,EVAL_CHUNK)]).numpy()[:,0]
                    samples=np.concatenate((samples[:eval_members//2],draws[eval_members//2:]))
                    q=np.quantile(samples,LEVELS,axis=0)
        if samples is not None and len(e['target_dates']) == 4:
            sum_quantiles.append(np.quantile(samples[:, :, 0].sum(axis=1), LEVELS, axis=0))
        q[:, :, :3] = np.floor(q[:, :, :3] + .5)
        quantiles.append(q)
        truths.append(e['target_values'])
        masks.append(e['target_available'])
    q = np.stack(quantiles, axis=1)
    y, mask = np.stack(truths), np.stack(masks)
    # Standard reported inputs record which context cells received a finalized value
    # because nothing was archived by the deadline (reports star them).
    inputs = {name: np.stack([e[name] for e in eps]) for name in ('filled', 'available', 'covariates_filled')
              if name in eps[0]}
    if 'covariates' in eps[0] and 'covariates_filled' in inputs:
        inputs['covariates_available'] = np.stack([e['covariates'][..., 1, :] for e in eps]).astype(bool)
    if 'forecast_cutoff_utc' in eps[0]:
        inputs['forecast_cutoff_utc'] = np.array([e['forecast_cutoff_utc'] for e in eps])
    if sum_quantiles:
        inputs['flu_admission_sum_quantiles'] = np.stack(sum_quantiles, axis=1)
    np.savez_compressed(output / 'forecasts.npz', quantiles=q, quantile_levels=LEVELS,
                        truth=y, mask=mask, context_end=[e['context_dates'][-1] for e in eps],
                        target_dates=[e['target_dates'] for e in eps], locations=eps[0]['locations'], **inputs)
    return [dict(context_end=e['context_dates'][-1]) for e in eps]


def input_scales(episodes, transform, ed_transform, populations):
    """Training-only scales (and logit centers) in each channel's model space."""
    import torch
    from chromantis.model.network import transform_counts, transform_proportions
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


def evaluate_hubs(model, panel, scenario, held_out, seed, eval_members, device, output, prepare=None, first=None):
    """Standard forecast evaluation: each pathogen scored on inputs visible at its own Hub's deadline.

    Evaluates the held-out season once per Hub (`dataset.build.HUBS`, Wednesday reports
    from `cv.score_episodes`; `first` reuses already cut FluSight episodes), each with the
    same evaluation seed, so issuances with identical inputs get identical draws. Writes
    one `forecasts.npz` whose channel c comes from the evaluation of `CHANNEL_HUBS[c]`.
    `prepare` maps episodes before evaluation (e.g. a correction stage). `filled`/`available`
    hold channel c at its own Hub's deadline; `<name>_<hub>` keep every channel and
    covariate a Hub's forecasts saw, since a model reads all six target histories."""
    from pathlib import Path
    import shutil
    from chromantis.dataset import cv
    from chromantis.dataset.build import HUBS, CHANNEL_HUBS
    output = Path(output)
    parts = {}
    for hub in HUBS:
        eps = first if hub == HUBS[0] and first is not None else cv.score_episodes(panel, scenario, held_out, hub)
        eps = prepare(eps) if prepare else eps
        folder = output / f'hub-{hub}'
        folder.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed + 1000)
        evaluate(model, eps, eval_members, device, folder)
        with np.load(folder / 'forecasts.npz', allow_pickle=False) as data:
            parts[hub] = {k: data[k] for k in data.files}
        shutil.rmtree(folder)
    merged = dict(parts[HUBS[0]])
    for name in ('context_end', 'target_dates', 'locations', 'truth', 'mask'):
        if any(not np.array_equal(parts[h][name], merged[name]) for h in HUBS):
            raise ValueError(f'Hub evaluations differ in {name}; Hub deadlines must only change inputs')
    for c, hub in enumerate(CHANNEL_HUBS):
        merged['quantiles'][:, :, :, c] = parts[hub]['quantiles'][:, :, :, c]
        for name, axis in (('filled', 2), ('available', 2)):
            if name in merged:
                np.moveaxis(merged[name], axis, 0)[c] = np.moveaxis(parts[hub][name], axis, 0)[c]
    for hub in HUBS:  # every input a Hub's forecasts saw, all channels, at that Hub's deadline
        if 'forecast_cutoff_utc' in parts[hub]:
            merged[f'forecast_cutoff_utc_{hub}'] = parts[hub]['forecast_cutoff_utc']
        for name in ('filled', 'available', 'covariates_filled', 'covariates_available'):
            if name in parts[hub]:
                merged[f'{name}_{hub}'] = parts[hub][name]
    for name in ('forecast_cutoff_utc', 'covariates_filled', 'covariates_available'):
        merged.pop(name, None)
    np.savez_compressed(output / 'forecasts.npz', **merged)
    return [dict(context_end=d) for d in merged['context_end']]


def mixture_many(q):
    """Equal mixture over axis 1 of quantile functions q [levels, members, ...] (linear, constant tails)."""
    u = (np.arange(256) + .5) / 256
    w = np.stack([np.interp(u, LEVELS, np.eye(len(LEVELS))[k]) for k in range(len(LEVELS))], 1)
    draws = np.tensordot(w, q, axes=(1, 0))  # [u, members, ...]
    return np.quantile(draws.reshape(-1, *q.shape[2:]), LEVELS, axis=0)


def mixture_quantiles(a,b):
    """Equal CDF mixture, linearly interpolated quantile functions, constant tails."""
    shape=a.shape
    a=a.reshape(len(LEVELS),-1);b=b.reshape(len(LEVELS),-1)
    # Deterministic probability quadrature avoids noisy resampling of direct heads.
    u=(np.arange(2048)+.5)/2048
    out=np.empty_like(a)
    for i in range(a.shape[1]):
        draws=np.r_[np.interp(u,LEVELS,a[:,i]),np.interp(u,LEVELS,b[:,i])]
        out[:,i]=np.quantile(draws,LEVELS)
    return out.reshape(shape)
