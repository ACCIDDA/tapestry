"""Fit independent signal finalizers and score boundary-week reference values."""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tapestry.dataset.build import load
from tapestry.dataset.finalization import signals, boundary_rows, split, training_statistics, validation_split, gap_features
from tapestry.model.finalization import triangle_predictions, seasonal_predictions, seasonal_gap_predictions, proxy_gap_predictions, choose_shrinkage, fit_gap, predict_gap
from tapestry.model.baselinenowcast import predictions as baselinenowcast_predictions
from .provenance import save, sha256


def upper_bound(name):
    if name.startswith('nssp_') or name.endswith('_pct_rank'):
        return 1.
    if name.startswith(('inpatient_', 'outpatient_')) or name in ('ilinet_ili', 'clinical_lab_flu_pct_positive'):
        return 100.
    return None


def baseline_predictions(rows, training, scales, fallback):
    history = rows['baseline_history']
    mask = np.isfinite(history)
    index = np.where(mask, np.arange(history.shape[1]), -1).max(1)
    li = rows['location']
    baseline = np.where(index >= 0, history[np.arange(len(history)), index.clip(0)], fallback[li])
    visible = np.isfinite(history[:, -1])
    corrected = baseline.copy()
    biases = {}
    for age in np.unique(rows['age']):
        use = training & visible & (rows['age'] == age)
        residual = np.log1p(rows['truth'][use] / scales[li[use]]) - np.log1p(baseline[use] / scales[li[use]])
        bias = float(np.median(residual)) if len(residual) else 0.
        apply = visible & (rows['age'] == age)
        corrected[apply] = np.maximum(0, np.expm1(np.log1p(baseline[apply] / scales[li[apply]]) + bias)) * scales[li[apply]]
        biases[int(age)] = bias
    return baseline, corrected, biases


def metrics(frame):
    """Native errors plus training-scale errors; equal weight for each location."""
    records = []
    if 'age' not in frame:
        frame = frame.assign(age=0)
    for (signal, age, kind), group in frame.groupby(['signal', 'age', 'kind'], sort=True):
        for method in ('prediction', 'persistence', 'median_revision', 'baselinenowcast'):
            if method not in group:
                continue
            error = group[method] - group.truth
            temp = pd.DataFrame(dict(location=group.location, ae=error.abs(), se=error**2,
                                     nae=error.abs() / group.scale, nse=(error / group.scale)**2,
                                     bias=error))
            means = temp.groupby('location').mean().mean()
            records.append(dict(signal=signal, age=int(age), kind=kind, method=method, cells=len(group),
                                locations=group.location.nunique(), weeks=group.issuance.nunique(),
                                mae=float(means.ae), rmse=float(np.sqrt(means.se)),
                                normalized_mae=float(means.nae), normalized_rmse=float(np.sqrt(means.nse)),
                                bias=float(means.bias)))
    return pd.DataFrame(records)


def reporting_support(panel, frame):
    """Classify target rows by visible history and prior reporting continuity."""
    f=frame[frame.signal.str.startswith(('nhsn_', 'nssp_'))].copy()
    wi=np.searchsorted(panel['issuance_dates'].astype(str),f.issuance)
    ti=np.searchsorted(panel['dates'].astype(str),f.boundary)
    locations={v:i for i,v in enumerate(panel['locations'].astype(str))}
    channels={v:i for i,v in enumerate(panel['target_names'].astype(str))}
    li=f.location.map(locations).to_numpy();ki=f.signal.map(channels).to_numpy()
    tt=ti[:,None]-np.arange(12)
    visible=panel['asof_targets'][wi[:,None],tt.clip(0),li[:,None],ki[:,None]]
    f['complete_history_12']=(np.isfinite(visible)&(tt>=0)).all(1)
    ww=wi[:,None]-np.arange(8);tt=ti[:,None]-np.arange(8)
    visible=panel['asof_targets'][ww.clip(0),tt.clip(0),li[:,None],ki[:,None]]
    f['uninterrupted_8']=(np.isfinite(visible)&(tt>=0)&(ww>=0)).all(1)
    return f


def target_percent_metrics(frame):
    """Newest-week errors with missing-data proximity explicit in every row."""
    frame=frame[frame.signal.str.startswith(('nhsn_', 'nssp_')) & frame.age.eq(0)].copy()
    frame['geography']=np.where(frame.location.eq('US'),'US','states')
    strata={'all':np.ones(len(frame),bool)}
    if 'kind' in frame:
        strata['reported']=frame.kind.eq('reported')
    if 'complete_history_12' in frame:
        strata['complete_history_12']=frame.complete_history_12
        strata['uninterrupted_8']=frame.uninterrupted_8
        strata['complete_and_uninterrupted']=frame.complete_history_12&frame.uninterrupted_8
    records=[]
    for stratum,keep in strata.items():
        for (signal,geography),g in frame[keep].groupby(['signal','geography']):
            for method in ('prediction','persistence','baselinenowcast'):
                if method not in g:
                    continue
                err=(g[method]-g.truth).abs()
                loc=pd.DataFrame(dict(location=g.location,ae=err,truth=g.truth,nae=err/g.scale,
                    within5=np.where(g.truth>0,err<=.05*g.truth,np.nan))).groupby('location')
                totals=loc[['ae','truth']].sum()
                records.append(dict(stratum=stratum,signal=signal,geography=geography,method=method,
                    cells=len(g),weeks=g.issuance.nunique() if 'issuance' in g else 0,locations=g.location.nunique(),
                    wape=err.sum()/g.truth.sum() if g.truth.sum()>0 else np.nan,
                    location_wape=(totals.ae/totals.truth.replace(0,np.nan)).mean(),
                    normalized_mae=loc.nae.mean().mean(),within5=loc.within5.mean().mean()))
    return pd.DataFrame(records)


def fit(scenario, seed, fold, device, output, dataset):
    if scenario.finalization_loss != 'mae':
        raise ValueError('Triangle calibration optimizes MAE; finalization_loss must be mae')
    torch.set_num_threads(int(os.environ.get('TAPESTRY_TORCH_THREADS', '1')))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panel = load(dataset)
    checkpoints, frames, coverage = {}, [], []
    reference_name='baselinenowcast_point_v2'+('_nssp_grid' if scenario.finalization_quantize else '')
    fold_info = None
    for si, (name, truth, asof, locations) in enumerate(signals(panel)):
        if scenario.finalization_scope == 'targets' and not name.startswith(('nhsn_', 'nssp_')):
            continue
        rows = boundary_rows(panel, name, truth, asof, locations, scenario.lookback, scenario.finalization_weeks)
        training, scoring, fold_info = split(panel, rows, scenario, fold)
        train_weeks = np.unique(rows['issuance'][training])
        record = dict(signal=name, lag=rows['lag'], training_cells=int(training.sum()),
                      training_weeks=len(train_weeks), eligible_score_cells=int(scoring.sum()),
                      training_weeks_by_age={int(a): len(np.unique(rows['issuance'][training & (rows['age'] == a)]))
                                             for a in range(scenario.finalization_weeks)})
        if len(train_weeks) < 6 or not scoring.any():
            coverage.append(dict(record, status='skipped', reason='fewer than six training Wednesdays or no score labels'))
            continue
        scales, fallback, trained = training_statistics(rows, training)
        scoring &= trained[rows['location']]
        record['untrained_location_cells'] = record['eligible_score_cells'] - int(scoring.sum())
        if not scoring.any():
            coverage.append(dict(record, status='skipped', reason='no trained locations on score support'))
            continue
        # Compute triangles only for rows that can enter this fit/evaluation.
        used = training | scoring
        rows = {k: (v[used] if isinstance(v,np.ndarray) and k != 'locations' else v) for k,v in rows.items()}
        training, scoring = training[used], scoring[used]
        inner, validation = validation_split(rows, training, maturity=scenario.finalization_maturity)
        # Validation labels never influence scales or feature transformations.
        inner_scales, inner_fallback, _ = training_statistics(rows, inner)
        inner_anchor, _, _ = baseline_predictions(rows, inner, inner_scales, inner_fallback)
        baseline, correction, bias = baseline_predictions(rows, training, scales, fallback)
        cap = upper_bound(name)
        if cap is not None:
            baseline, correction = np.minimum(baseline, cap), np.minimum(correction, cap)
            inner_anchor = np.minimum(inner_anchor, cap)
        li = rows['location']
        integer_reports = name.startswith('nhsn_')
        reference, reference_status, reference_support = baselinenowcast_predictions(
            panel, asof, rows, baseline, integer=integer_reports)
        if cap is not None:
            reference = np.minimum(reference, cap)
        if integer_reports and not np.all(rows['truth'][np.isfinite(rows['truth'])] == np.rint(rows['truth'][np.isfinite(rows['truth'])])):
            raise ValueError(f'{name}: report quantization requires integer reference counts')
        geography = np.where(locations[li]=='US','US','states')
        reported = np.isfinite(rows['baseline_history'][:, -1])
        priority = (name.startswith(('nhsn_','nssp_','inpatient_')) or
            name.endswith('_wval_like') or name in
            ('kinsa_ili','ilinet_ili','clinical_lab_flu_pct_positive','flusurv_flu_rate'))
        common_factors=np.ones(len(baseline))
        context_diagnostics={}
        estimator = triangle_predictions if scenario.finalization_model == 'triangle' else seasonal_predictions
        options = {} if scenario.finalization_model == 'triangle' else dict(mode='adaptive_chain' if scenario.finalization_model=='context_residual' else scenario.finalization_model,
            halflife=scenario.finalization_halflife,prior_weeks=scenario.finalization_pool,statistic=scenario.finalization_statistic,growth_bandwidth=scenario.finalization_growth)
        if scenario.finalization_model == 'context_residual':
            from tapestry.model.context_nowcast import context_predictions
            estimator = context_predictions
            options = dict(penalty=scenario.finalization_penalty,halflife=scenario.finalization_halflife,
                           features_mode=scenario.finalization_features,gate=scenario.finalization_gate,diagnostics=context_diagnostics,strength=scenario.finalization_strength,
                           growth_weight=scenario.finalization_growth_weight,residual_halflife=scenario.finalization_residual_halflife,
                           prior_weeks=scenario.finalization_pool,statistic=scenario.finalization_statistic)
        raw, support, factors = (estimator(panel,asof,rows,baseline,integer=integer_reports,common_factors=common_factors,**options)
            if priority else (baseline.astype(float).copy(),np.zeros(len(baseline),int),np.ones(len(baseline))))
        # The triangle alone leaves missing cells at the fallback.
        delta = raw-baseline
        shrinkage, validation_scores = {}, {}
        predicted = baseline.astype(float).copy()
        report_start=str(np.datetime64(max(rows['issuance'][training]))-np.timedelta64(51*7,'D'))
        report_training=training & (rows['issuance']>=report_start)
        for age in range(scenario.finalization_weeks):
            for group in ('US','states'):
                apply = reported & (rows['age']==age) & (geography==group)
                use = report_training & apply
                key = f'reported:{group}:{age}'
                supported = len(np.unique(rows['issuance'][use]))>=6
                alpha, losses = (choose_shrinkage(rows['truth'][use],baseline[use],delta[use],
                    scales[li[use]],li[use],upper_bound=cap,integer=integer_reports)
                    if supported and scenario.finalization_model == 'triangle' else (0.,[]))
                if scenario.finalization_model != 'triangle':
                    alpha, losses = 1., []  # Causal curves do not calibrate on frozen future labels.
                shrinkage[key],validation_scores[key] = alpha,losses
                predicted[apply] += alpha*delta[apply]
        # Train on observed as well as missing examples, without the current
        # target report as an input. This learns imputation rather than identity.
        gap_saved = None
        seasonal_gap = scenario.finalization_gap != 'ridge' and name.startswith(('nhsn_', 'nssp_'))
        if seasonal_gap:
            candidate = (proxy_gap_predictions(panel, asof, rows, baseline, name, integer=integer_reports)
                         if scenario.finalization_gap == 'proxy' else
                         seasonal_gap_predictions(panel, asof, rows, baseline, integer=integer_reports,
                                                  trend=scenario.finalization_gap == 'trend'))
            predicted[~reported] = candidate[~reported]
        if not seasonal_gap and priority and (~reported & scoring).any() and inner.sum()>=100 and validation.any():
            x,names,normalizers=gap_features(panel,rows,name,inner)
            past=dict(rows,baseline_history=rows['baseline_history'][:,:-1])
            ia,_,_=baseline_predictions(past,inner,inner_scales,inner_fallback)
            oa,_,_=baseline_predictions(past,training,scales,fallback)
            beta=fit_gap(x[inner],rows['truth'][inner],ia[inner],inner_scales[li[inner]],li[inner],rows['issuance'][inner])
            candidate=predict_gap(x,beta,ia,inner_scales[li])
            missing_alpha={}
            for group in ('US','states'):
                use=validation & ~reported & (geography==group)
                supported=len(np.unique(rows['issuance'][use]))>=4
                alpha,losses=(choose_shrinkage(rows['truth'][use],inner_anchor[use],
                    candidate[use]-inner_anchor[use],inner_scales[li[use]],li[use],upper_bound=cap)
                    if supported else (0.,[]))
                missing_alpha[group]=alpha
                shrinkage['missing:'+group]=alpha
                validation_scores['missing:'+group]=losses
            if any(missing_alpha.values()):
                beta=fit_gap(x[training],rows['truth'][training],oa[training],scales[li[training]],li[training],rows['issuance'][training])
                candidate=predict_gap(x,beta,oa,scales[li])
                for group,alpha in missing_alpha.items():
                    use=~reported & (geography==group)
                    predicted[use]=baseline[use]+alpha*(candidate[use]-baseline[use])
                gap_saved=dict(coefficients=beta,normalizers=normalizers,feature_names=names)
        predicted = np.maximum(0,predicted)
        if cap is not None:
            predicted = np.minimum(predicted,cap)
        if integer_reports:
            predicted[reported] = np.rint(predicted[reported])
        if scenario.finalization_quantize and name.startswith('nssp_'):
            # The archived source and reference use 0.01 percentage-point steps.
            # Apply the same point-output resolution to all fitted comparators.
            predicted=np.rint(predicted*10000)/10000
            reference=np.rint(reference*10000)/10000
            correction=np.rint(correction*10000)/10000
        checkpoint = dict(finalization_quantize=scenario.finalization_quantize,finalization_statistic=scenario.finalization_statistic,finalization_gap=scenario.finalization_gap,finalization_model=scenario.finalization_model,gap_model=gap_saved,shrinkage=shrinkage,max_delay=12,
            window=52 if scenario.finalization_model=='triangle' else 104,
            prior_weeks=12 if scenario.finalization_model=='triangle' else scenario.finalization_pool,
            seasonal_bandwidth=None if scenario.finalization_model=='triangle' else 8,
            recency_halflife=scenario.finalization_halflife if scenario.finalization_model.startswith('adaptive') else None,
            integer_reports=integer_reports,scales=scales.tolist(),fallback=fallback.tolist(),
            upper_bound=cap,locations=locations.tolist(),trained_locations=trained.tolist(),
            lag=rows['lag'],reconstruction_weeks=scenario.finalization_weeks)
        predicted = predicted[scoring]
        if not np.isfinite(predicted).all():
            raise ValueError(f'Nonfinite predictions: {name}')
        record.update(inner_training_cells=int(inner.sum()), validation_cells=int(validation.sum()),
            validation_start=str(min(rows['issuance'][validation])) if validation.any() else None,
            shrinkage=shrinkage, calibration_losses=validation_scores,report_calibration_start=report_start,
            report_calibration_cells=int(report_training.sum()))
        history = rows['baseline_history'][scoring]
        kind = np.where(np.isfinite(history[:, -1]), 'reported',
                        np.where(np.isfinite(history).any(1), 'missing_recent_history', 'missing_no_recent_history'))
        frames.append(pd.DataFrame(dict(signal=name, issuance=rows['issuance'][scoring],
            boundary=rows['boundary'][scoring], age=rows['age'][scoring],
            elapsed_weeks=rows['elapsed_weeks'][scoring], location=locations[rows['location'][scoring]],
            kind=kind, report=history[:, -1], truth=rows['truth'][scoring], prediction=predicted,
            baselinenowcast=reference[scoring], baselinenowcast_status=reference_status[scoring],
            baselinenowcast_support=reference_support[scoring],
            residual_alpha=context_diagnostics.get('alpha',np.ones(len(baseline)))[scoring],
            triangle_raw=raw[scoring], triangle_support=support[scoring], development_factor=factors[scoring],shared_development_factor=common_factors[scoring],
            persistence=baseline[scoring], median_revision=correction[scoring], scale=scales[rows['location'][scoring]])))
        checkpoints[name] = checkpoint
        coverage.append(dict(record, status='complete', score_cells=int(scoring.sum())))
        print(json.dumps(dict(fold=fold, **coverage[-1])), flush=True)
    if not frames:
        raise ValueError(f'No finalization signals have training and scoring support in {fold}')
    frame = pd.concat(frames, ignore_index=True)
    # Serialize all native values at the same precision (float32/64 CSV repr differs).
    for column in ('truth','report','persistence','median_revision','scale'):
        frame[column] = frame[column].astype(float)
    frame.to_csv(output / 'finalizations.csv.gz', index=False)
    scored = metrics(frame)
    scored.to_csv(output / 'finalization-metrics.csv', index=False)
    save(output / 'baselinenowcast_scores.json', dict(
        baseline=reference_name,
        status_counts=frame.baselinenowcast_status.value_counts().to_dict(),
        metrics=scored[scored.method == 'baselinenowcast'].to_dict('records')))
    save(output / 'finalization_scores.json', dict(coverage=coverage, rows=len(frame), signals=len(checkpoints)))
    metadata = dict(formulation=scenario.finalization_model, baseline=reference_name, scenario=scenario.scenario_string, seed=seed, fold=fold, **fold_info,
                    dataset=str(dataset), dataset_sha256=sha256(dataset),
                    output='native_unit_point_estimates', coverage=coverage)
    save(output / 'manifest.json', metadata)
    torch.save(dict(signals=checkpoints, metadata=metadata), output / 'model.pt')
    return output / 'model.pt'


def rank(folder, runs, make_plots=True):
    """Compare common cell support within each CV protocol; never rank unlike folds together."""
    from tapestry.model.scenario import Scenario
    output = folder / 'finalization-ranking'
    output.mkdir(exist_ok=True)
    scores, cell_support, diagnostics, coverage, target_scores = [], {}, [], [], []
    trajectory_scores = []
    panel=load(json.loads((folder/'experiment.json').read_text())['dataset'])
    for row in runs:
        scenario = Scenario.from_string(row['scenario'])
        for fold in scenario.scored_seasons:
            manifest = json.loads((folder / row['attempt'] / f'eval_{fold}' / 'manifest.json').read_text())
            coverage.extend(dict(c, scenario=row['scenario'], seed=row['seed'], fold=fold,
                                 cv=scenario.finalization_cv) for c in manifest['coverage'])
            frame = pd.read_csv(folder / row['attempt'] / f'eval_{fold}' / 'finalizations.csv.gz')
            if 'age' not in frame:
                frame['age'] = 0
            support = set(map(tuple, frame[['signal', 'issuance', 'boundary', 'location']].to_numpy()))
            key = (scenario.finalization_cv, scenario.finalization_maturity, scenario.finalization_weeks, fold)
            if key in cell_support and cell_support[key] != support:
                raise ValueError(f'Finalization scoring support differs for {key}; compare matched configurations')
            cell_support[key] = support
            scored = metrics(frame)
            scored['scenario'], scored['seed'], scored['fold'] = row['scenario'], row['seed'], fold
            scored['cv'] = scenario.finalization_cv
            scored['maturity'] = scenario.finalization_maturity
            scored['lookback'], scored['width'] = scenario.lookback, scenario.width
            scored['reconstruction_weeks'] = scenario.finalization_weeks
            scores.append(scored)
            supported = reporting_support(panel,frame)
            target_scored=target_percent_metrics(supported)
            if scenario.finalization_weeks >= 4:
                from tapestry.evaluation.trajectory import trajectory_cells, trajectory_metrics
                trajectory_scores.append(trajectory_metrics(trajectory_cells(supported)).assign(
                    scenario=row['scenario'], seed=row['seed'], fold=fold))
            target_scored['scenario'],target_scored['seed'],target_scored['fold']=row['scenario'],row['seed'],fold
            target_scores.append(target_scored)
            # Stream geography/season diagnostics one fold at a time.
            from tapestry.dataset.cv import season
            frame['geography'] = np.where(frame.location == 'US', 'US', 'states')
            seasons = {day: season(day) for day in frame.boundary.unique()}
            frame['season'] = frame.boundary.map(seasons)
            for (geography, season_name), group in frame.groupby(['geography', 'season']):
                m = metrics(group)
                m['scenario'], m['seed'], m['fold'] = row['scenario'], row['seed'], fold
                m['geography'], m['season'] = geography, season_name
                diagnostics.append(m)
    details = pd.concat(scores, ignore_index=True)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    details.to_csv(output / 'signal-scores.csv', index=False)
    # First average folds/seeds within signal; then give each signal equal weight.
    columns = ['cv', 'maturity', 'scenario', 'lookback', 'width', 'reconstruction_weeks', 'age', 'kind', 'method']
    by_signal = details.groupby(columns + ['signal'], as_index=False)[['normalized_mae', 'normalized_rmse']].mean()
    summary = by_signal.groupby(columns, as_index=False).agg(
        normalized_mae=('normalized_mae', 'mean'), normalized_rmse=('normalized_rmse', 'mean'),
        signals=('signal', 'nunique')).sort_values(['cv', 'kind', 'normalized_mae'])
    summary.to_csv(output / 'ranking.csv', index=False)
    run_scores = details.groupby(columns + ['seed', 'signal'], as_index=False).normalized_mae.mean()
    run_scores = run_scores.groupby(columns + ['seed'], as_index=False).normalized_mae.mean()
    run_scores.to_csv(output / 'run-scores.csv', index=False)
    pd.concat(diagnostics, ignore_index=True).to_csv(output / 'geography-season-scores.csv', index=False)
    if trajectory_scores:
        trajectory = pd.concat(trajectory_scores, ignore_index=True)
        trajectory.to_csv(output/'trajectory-performance.csv', index=False)
        trajectory.groupby(['scenario','fold','geography','stratum','method'], as_index=False)[
            ['point','level','growth','log_growth','trajectory']].mean().to_csv(output/'trajectory-ranking.csv', index=False)
    targets=pd.concat(target_scores,ignore_index=True)
    targets.to_csv(output/'target-performance.csv',index=False)
    target_rank=targets.groupby(['scenario','fold','geography','stratum','method'],as_index=False).agg(
        wape=('wape','mean'),location_wape=('location_wape','mean'),
        normalized_mae=('normalized_mae','mean'),within5=('within5','mean'),signals=('signal','nunique'))
    target_rank.to_csv(output/'target-ranking.csv',index=False)
    if make_plots:
        plot(details, output)
    print(summary[['cv', 'lookback', 'reconstruction_weeks', 'age', 'kind', 'method', 'normalized_mae', 'signals']].to_string(index=False))
    return output


def plot(details, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from tapestry.model.scenario import Scenario
    for (cv, maturity), data in details.groupby(['cv', 'maturity']):
        # Every curve retains reconstruction age; never average a two-week and
        # eight-week output window into a purported common task.
        means = data.groupby(['scenario', 'age', 'kind', 'method', 'signal']).normalized_mae.mean()
        means = means.groupby(['scenario', 'age', 'kind', 'method']).mean()
        kinds = sorted(data.kind.unique())
        fig, axes = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 5), squeeze=False)
        for ax, kind in zip(axes[0], kinds):
            table = means.xs(kind, level='kind').unstack('method')
            for scenario, group in table.groupby(level='scenario'):
                setting = Scenario.from_string(scenario)
                group = group.droplevel('scenario').sort_index()
                ratio = group.prediction / group.persistence.replace(0, np.nan)
                ax.plot(group.index, ratio, marker='o', markersize=3,
                        label=f'R={setting.finalization_weeks}, input={setting.lookback}')
                if 'baselinenowcast' in group:
                    ax.plot(group.index, group.baselinenowcast / group.persistence.replace(0, np.nan),
                            linestyle=':', label=f'baselinenowcast R={setting.finalization_weeks}, input={setting.lookback}')
            ax.axhline(1, color='black', linewidth=1, linestyle='--')
            ax.set_title(kind.replace('_', ' '))
            ax.set_xlabel('Weeks behind each signal’s T-X boundary')
            ax.set_ylabel('Normalized MAE / persistence (lower is better)')
            ax.set_xticks(range(int(data.age.max()) + 1))
            ax.grid(alpha=.2)
        axes[0, -1].legend(fontsize=7)
        fig.suptitle(f'Wednesday finalization: {cv}; {maturity}-week reference-age filter')
        fig.tight_layout()
        fig.savefig(output / f'{cv}-m{maturity}-age-performance.png', dpi=160)
        plt.close(fig)
        chosen = [s for s in data.scenario.unique()
                  if Scenario.from_string(s).finalization_weeks == 4 and Scenario.from_string(s).lookback == 3]
        if not chosen:
            continue
        current = data[(data.scenario == chosen[0]) & (data.kind == 'reported')]
        table = current.groupby(['age', 'signal', 'method']).normalized_mae.mean().unstack('method')
        ratio = (table.prediction / table.persistence.replace(0, np.nan)).unstack('signal')
        fig, ax = plt.subplots(figsize=(15, 5))
        im = ax.imshow(ratio.to_numpy(), aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=2)
        ax.set_xticks(range(len(ratio.columns)), ratio.columns, rotation=55, ha='right')
        ax.set_yticks(range(len(ratio)), [f'T-X-{a}' for a in ratio.index])
        for i in range(len(ratio)):
            for j in range(len(ratio.columns)):
                v = ratio.iloc[i, j]
                ax.text(j, i, f'{v:.2f}' if np.isfinite(v) else '—', ha='center', va='center', fontsize=7)
        ax.set_title(f'{cv}: four reconstructed weeks, three input weeks per estimate\nReported cells: normalized MAE / unchanged report')
        fig.colorbar(im, ax=ax, label='Error ratio; 1 = unchanged report; dash = zero baseline or no support')
        fig.tight_layout()
        fig.savefig(output / f'{cv}-m{maturity}-four-week-signals.png', dpi=160)
        plt.close(fig)
