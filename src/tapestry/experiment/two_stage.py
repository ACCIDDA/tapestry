"""Season-cross-fitted nowcasts for an independently trained forecaster.

Each stage uses its resolved epoch/early-stopping settings. Forecast validation
never trains a nowcaster; any nowcaster validation split stays inside its permitted
fitting partition.
"""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tapestry.dataset import cv
from tapestry.dataset.build import load, covariate_names_for
from tapestry.dataset.episodes import episodes, restrict_labels
from tapestry.model.network import checkpoint
from tapestry.model.pipeline import NowcastForecast, reconstruct
from . import training
from .provenance import save, environment, sha256


def nowcast_training(panel, scenario, keep):
    """Mask by reference week before cutting episodes or fitting preprocessing."""
    dates = np.asarray(panel['dates']).astype(str)
    allowed = set(dates[keep])
    result = episodes(cv.masked(panel, keep), scenario.lookback, 'vintaged',
                      covariate_names_for(scenario.covariate_set), scenario.lookback,
                      horizons=scenario.horizons)
    return [e for e in result if e['context_dates'][-1] in allowed]


def fit_nowcaster(panel, scenario, keep, seed, device, pop):
    train = nowcast_training(panel, scenario, keep)
    if not train:
        raise ValueError('No nowcast training episodes in cross-fitting partition')
    selection, validation = train, None
    if scenario.patience:
        dates = np.asarray(panel['dates']).astype(str)
        hidden = keep & (cv.week_roles(dates, scenario, held_out='') == 'validation')
        selection = nowcast_training(panel, scenario, keep & ~hidden)
        validation = [e for e in (restrict_labels(e, set(dates[hidden])) for e in train) if e is not None]
        if not selection or not validation:
            raise ValueError('Nowcaster early stopping needs fitting and validation labels in this partition')
    return training.fit_models(selection, validation, train, scenario, seed, device, pop)


def select_covariates(episode, all_names, selected_names):
    out = dict(episode)
    out.pop('covariates', None)
    if selected_names:
        out['covariates'] = episode['covariates'][:, [all_names.index(n) for n in selected_names]]
    return out


def reconstructed_episodes(model, batch, all_names, forecast_names, members, device, lookback):
    """Cache whole joint draws; preserve dates, labels and the original missingness outside R."""
    model.to(device).eval()
    result = []
    for e in batch:
        source = select_covariates(e, all_names, model.config['covariate_names'])
        values, available, final, _, _, cal, cov = training.to_tensors([source], device)
        with torch.no_grad():
            histories = reconstruct(model, values=values, available=available, known_final=final,
                                    calendar=cal, covariates=training.covariate_history([source], cov, model.config['covariate_names']),
                                    locations=e['locations'], members=members,
                                    context_dates=(e['context_dates'],), issuances=(e['issuance'],))
        out = select_covariates(e, all_names, forecast_names)
        out.update(history_samples=histories.values[:, 0].cpu().numpy(),
                   values=histories.values[0, 0].cpu().numpy(),
                   available=histories.available[0].cpu().numpy(),
                   known_final=histories.known_final[0].cpu().numpy(),
                   estimated=histories.estimated[0].cpu().numpy())
        for key in ('values', 'available', 'known_final', 'estimated', 'covariates'):
            if key in out:
                out[key] = out[key][-lookback:]
        out['context_dates'] = out['context_dates'][-lookback:]
        out['history_samples'] = out['history_samples'][:, -lookback:]
        result.append(out)
    model.cpu()
    return result


def cross_fit_keep(panel, keep, label, batch, recent):
    """Exclude prediction season and boundary-spanning reconstructed/forecast labels."""
    dates = np.asarray(panel['dates']).astype(str)
    labels = np.array([cv.season(d) for d in dates])
    excluded = {d for e in batch for d in (*e['context_dates'][-recent:], *e['target_dates'])}
    return keep & (labels != label) & ~np.isin(dates, list(excluded))


def cross_fitted_histories(panel, scenario, keep, batch, all_names, seed, device, pop):
    """For each input season, train its nowcaster on other permitted seasons only."""
    origins = np.array([cv.season(e['context_dates'][-1]) for e in batch])
    result, records = [], []
    ns = scenario.stage('nowcast')
    forecast_names = list(covariate_names_for(scenario.stage('forecast').covariate_set))
    for label in sorted(set(origins)):
        selected = [e for e, origin in zip(batch, origins) if origin == label]
        fitting = cross_fit_keep(panel, keep, label, selected, scenario.nowcast_weeks)
        model, fit_records = fit_nowcaster(panel, ns, fitting, seed, device, pop)
        result.extend(reconstructed_episodes(model, selected, all_names, forecast_names,
                                             scenario.nowcast_members, device, scenario.stage('forecast').lookback))
        records.append(dict(prediction_season=label, fitting_dates=np.asarray(panel['dates'])[fitting].astype(str).tolist(),
                            records=fit_records))
    return result, records


def fit_pipeline(scenario, seed, held_out, eval_members, device, output, dataset, population_file):
    panel = load(dataset)
    ns, fs = scenario.stage('nowcast'), scenario.stage('forecast')
    episode_scenario = scenario.episode_scenario()
    all_names = list(covariate_names_for(episode_scenario.covariate_set))
    full = cv.fold(panel, episode_scenario, held_out)
    pop = training.populations(population_file, full.train[0]['locations'])
    roles = cv.week_roles(panel['dates'], fs, held_out)
    full_keep = np.isin(roles, ['fit', 'validation'])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    forecast_names = list(covariate_names_for(fs.covariate_set))
    train, cross_records = cross_fitted_histories(panel, scenario, full_keep, full.train,
                                                 all_names, seed, device, pop)
    inner_train, validation, inner_records = train, None, None
    if fs.patience:
        inner = cv.fold(panel, episode_scenario, held_out, inner=True)
        inner_keep = roles == 'fit'
        inner_train, inner_cross = cross_fitted_histories(panel, scenario, inner_keep, inner.train,
                                                         all_names, seed, device, pop)
        validation_nowcaster, validation_records = fit_nowcaster(panel, ns, inner_keep, seed, device, pop)
        validation = reconstructed_episodes(validation_nowcaster, inner.validation, all_names,
                                              forecast_names, scenario.nowcast_members, device, fs.lookback)
        inner_records = dict(fold=inner.info, cross_fits=inner_cross, nowcaster=validation_records)
    forecaster, forecast_records = training.fit_models(inner_train, validation, train, fs, seed, device, pop)
    nowcaster, nowcast_records = fit_nowcaster(panel, ns, full_keep, seed, device, pop)
    model = NowcastForecast(nowcaster, forecaster)
    metadata = dict(scenario=scenario.scenario_string, run_id=scenario.run_id, config=asdict(scenario),
                    seed=seed, held_out_season=held_out, protocol='independent_stages_cross_fitted_v2',
                    fold=full.info, cross_fits=cross_records, inner=inner_records,
                    records=dict(nowcast=nowcast_records, forecast=forecast_records),
                    nowcast_config=asdict(ns), forecast_config=asdict(fs), loss=training.LOSS_DEFINITION,
                    dataset=str(dataset), dataset_sha256=sha256(dataset),
                    channels=panel['target_names'].tolist(), locations=list(full.train[0]['locations']),
                    eval_members=eval_members, **environment())
    for name, stage in [('nowcaster', nowcaster), ('forecaster', forecaster), ('model', model)]:
        torch.save(checkpoint(stage, dict(metadata, stage=name)), output / f'{name}.pt')
    # Pipeline input covariates follow its declared union, which can differ from panel order.
    score = [select_covariates(e, all_names, model.config['covariate_names']) for e in full.score]
    training.evaluate(model.to(device), score, eval_members, device, output)
    nowcast_fold = cv.fold(panel, ns, held_out)
    nowcast_output = output / 'nowcast'
    nowcast_output.mkdir(exist_ok=True)
    training.evaluate(nowcaster.to(device), nowcast_fold.score, eval_members, device, nowcast_output, panel=panel)
    save(output / 'manifest.json', metadata)
    return output / 'model.pt'
