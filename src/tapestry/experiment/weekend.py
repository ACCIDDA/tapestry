"""B2-sized revision experiments using the common planner, fitter and scorer."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from tapestry.dataset import cv
from tapestry.dataset.build import load, covariate_names_for
from tapestry.dataset.episodes import select_covariates
from tapestry.dataset.reporting_error import ReportingErrors
from tapestry.model.network import checkpoint
from tapestry.model.revision_regression import RevisionRegression
from . import training
from .provenance import save, sha256, environment


def scheduled_reports(e, panel, names):
    """Preserve B2 availability and source lags; replace visible values by reports.

    Archived gaps use explicitly assumed final proxies. They are not revision
    observations and never enter the empirical reporting-error library.
    """
    dates = panel['dates'].astype(str)
    issue = str(np.datetime64(e['context_dates'][-1]) + np.timedelta64(4, 'D'))
    issues = panel['issuance_dates'].astype(str)
    wi = np.searchsorted(issues, issue)
    if wi >= len(issues) or issues[wi] != issue:
        raise ValueError(f'No evaluation issuance {issue}')
    ti = np.searchsorted(dates, e['context_dates'])
    if (ti >= len(dates)).any() or not np.array_equal(dates[ti], e['context_dates']):
        raise ValueError('Evaluation dates do not align')
    raw = np.moveaxis(panel['asof_targets'][wi, ti], -1, -2)
    available = e['available'].copy()
    out = dict(e, values=np.where(available, np.where(np.isfinite(raw), raw, e['values']), 0),
               known_final=np.zeros_like(available), issuance=issue)
    proxy = int((available & ~np.isfinite(raw)).sum())
    if names:
        v, a = select_covariates(panel['asof_covariates'][wi, ti], panel['asof_covariates_national'][wi, ti],
            panel['covariate_names'], panel['covariate_national_names'], panel['locations'], names)
        cov = e['covariates'].copy()
        cov[..., 0, :] = np.where(cov[..., 1, :], np.where(a, v, cov[..., 0, :]), 0)
        out['covariates'] = cov
    return out, proxy


class CorrectedErrors:
    """Fresh synthetic reports each epoch, corrected by season-cross-fitted models."""
    transport_missingness = False
    def __init__(self, bank, models):
        self.bank, self.models = bank, models
    def batch(self, batch, rng):
        return [self.models[cv.season(e['context_dates'][-1])].apply(self.bank.draw(e, rng)) for e in batch]


def fit_correctors(train, bank, scenario, seed):
    # The empirical error distribution belongs to the outer training fold. Each
    # trajectory season is excluded from its correction regressor's label fit.
    rng = np.random.default_rng(seed + 410000)
    examples = [(bank.draw(e, rng), e) for e in train for _ in range(2)]
    def build(ex):
        return RevisionRegression(scenario.nowcast_weeks, scenario.correction_penalty,
                                  scenario.correction_strength, scenario.correction_features).fit(ex)
    labels = sorted({cv.season(e['context_dates'][-1]) for e in train})
    by_season = {}
    for label in labels:
        permitted = []
        for x, t in examples:
            if cv.season(t['context_dates'][-1]) == label:
                continue
            label_mask = np.array([cv.season(d) != label for d in t['context_dates']])[:, None, None]
            permitted.append((x, dict(t, available=t['available'] & label_mask)))
        by_season[label] = build(permitted)
    final = build(examples)
    return by_season, final


def subset_labels(e, indices):
    out = dict(e)
    for name in ('target_values', 'target_available', 'Y'):
        out[name] = e[name][indices]
    out['target_dates'] = tuple(np.asarray(e['target_dates'])[indices])
    return out


class ForecastSlice(torch.nn.Module):
    def __init__(self, model, indices):
        super().__init__()
        self.model, self.indices = model, indices
        self.config = dict(model.config, horizons=[model.config['horizons'][i] for i in indices])
        self.scale = model.models[0].scale if hasattr(model, 'models') else model.scale
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)[:, :, self.indices]


def fit(scenario, seed, held_out, members, device, output, dataset):
    panel = load(dataset)
    full = cv.fold(panel, scenario, held_out)
    inner = cv.fold(panel, scenario, held_out, inner=True) if scenario.patience else None
    roles = cv.week_roles(panel['dates'], scenario, held_out)
    keep = np.isin(roles, ['fit', 'validation'])
    outer = ReportingErrors(panel, scenario, held_out, keep)
    selection = ReportingErrors(panel, scenario, held_out, roles == 'fit') if inner else None
    names = list(covariate_names_for(scenario.covariate_set))
    score, proxies = zip(*(scheduled_reports(e, panel, names) for e in full.score))
    score = list(score)
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    correction_records = None
    if scenario.weekend_family == 'two_stage':
        cross, final = fit_correctors(full.train, outer, scenario, seed)
        augmentation = CorrectedErrors(outer, cross)
        selection_augmentation = None
        if inner:
            inner_cross, inner_final = fit_correctors(inner.train, selection, scenario, seed)
            selection_augmentation = CorrectedErrors(selection, inner_cross)
            raw = selection.batch(inner.validation, np.random.default_rng(seed + 800000))
            inner.validation = [inner_final.apply(e) for e in raw]
        score = [final.apply(e) for e in score]
        correction_records = dict(full=final.record(), cross_fits={k:v.record() for k,v in cross.items()},
            inner=inner_final.record() if inner else None,
            error_library='Shared fold-permitted donor errors; regression labels exclude the corrected trajectory season')
    else:
        augmentation, selection_augmentation = outer, selection
        if inner:
            inner.validation = selection.batch(inner.validation, np.random.default_rng(seed + 800000))
    pop = training.populations('data/metadata/locations.csv', full.train[0]['locations'])
    model, records = training.fit_models(inner.train if inner else full.train,
        inner.validation if inner else None, full.train, scenario, seed, device, pop,
        augmenter=augmentation, selection_augmenter=selection_augmentation)
    metadata = dict(scenario=scenario.scenario_string, run_id=scenario.run_id, config=asdict(scenario),
        seed=seed, held_out_season=held_out, protocol='b2_weekend_scheduled_reports_v1',
        fold=full.info, inner_fold=inner.info if inner else None, records=records,
        reporting_errors=outer.metadata, selection_errors=selection.metadata if selection else None,
        corrections=correction_records, target_proxy_cells=sum(proxies),
        labels='Finalized future values; joint models additionally learn finalized recent context values',
        evaluation='Wednesday archived values on B2 source schedule; absent archives use final proxies',
        loss=training.LOSS_DEFINITION, joint_weight=scenario.joint_weight,
        dataset=str(dataset), dataset_sha256=sha256(dataset), eval_members=members,
        channels=panel['target_names'].tolist(), locations=list(full.train[0]['locations']), **environment())
    torch.save(checkpoint(model, metadata), out / 'model.pt')
    save(out / 'manifest.json', metadata)
    model.to(device)
    torch.manual_seed(seed + 1000)
    if scenario.weekend_family == 'joint':
        horizons = np.asarray(scenario.horizons)
        indices = np.flatnonzero(horizons > 0).tolist()
        training.evaluate(ForecastSlice(model, indices), [subset_labels(e, indices) for e in score], members, device, out)
        indices = np.flatnonzero(horizons <= 0).tolist()
        recent = out / 'nowcast'; recent.mkdir(exist_ok=True)
        training.evaluate(ForecastSlice(model, indices), [subset_labels(e, indices) for e in score], members, device, recent, panel=panel)
    else:
        training.evaluate(model, score, members, device, out)
    return out / 'model.pt'
