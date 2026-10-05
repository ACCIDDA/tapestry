"""Evaluate fixed CV checkpoints on finalized, vintage, and corrected-vintage inputs."""
from dataclasses import asdict, replace
from pathlib import Path
import fcntl
import json

import numpy as np
import torch

from tapestry.dataset import cv
from tapestry.dataset.build import load, covariate_names_for
from tapestry.dataset.episodes import select_covariates
from tapestry.dataset.finalization import boundary_rows
from tapestry.model.finalization import seasonal_predictions, proxy_gap_predictions
from tapestry.model.network import load_model, checkpoint
from .provenance import save, sha256
from .training import evaluate


def selected_nowcasts(panel, wanted_issues, mode='adaptive_chain', growth_bandwidth=.2, penalty=10., growth_weight=1., residual_halflife=26., features_mode="basic", gate="none", strength=1.):
    """Causal selected estimator on all locations, independent of final-label support."""
    shape = (len(panel['issuance_dates']), 8, len(panel['locations']), len(panel['target_names']))
    result = np.full(shape, np.nan, np.float32)
    issues = panel['issuance_dates'].astype(str)
    for k, name in enumerate(panel['target_names'].astype(str)):
        asof = panel['asof_targets'][..., k]
        # Dummy labels: prediction construction must not depend on future truth.
        rows = boundary_rows(panel, name, np.zeros(asof.shape[1:], np.float32), asof,
                             panel['locations'].astype(str), 12, 8)
        use = np.isin(rows['issuance'], wanted_issues)
        rows = {key: value[use] if isinstance(value, np.ndarray) and key != 'locations' else value
                for key, value in rows.items()}
        integer = name.startswith('nhsn_')
        anchor = np.zeros(len(rows['location']))
        if mode == 'context_residual':
            from tapestry.model.context_nowcast import context_predictions
            prediction, _, _ = context_predictions(panel,asof,rows,anchor,integer=integer,penalty=penalty,
                growth_weight=growth_weight,residual_halflife=residual_halflife,features_mode=features_mode,gate=gate,strength=strength)
        else:
            prediction, _, _ = seasonal_predictions(panel, asof, rows, anchor, mode=mode,
                integer=integer, halflife=8., prior_weeks=4., statistic='median', growth_bandwidth=growth_bandwidth)
        reported = np.isfinite(rows['baseline_history'][:, -1])
        gap = proxy_gap_predictions(panel, asof, rows, anchor, name, integer=integer)
        prediction[~reported] = gap[~reported]
        prediction = np.maximum(0, prediction)
        if integer:
            prediction[reported] = np.rint(prediction[reported])
        else:
            prediction = np.rint(np.minimum(1, prediction) * 10000) / 10000
        # Never invent a zero-valued observation in a location with no archive history.
        prediction[~rows['active']] = np.nan
        wi = np.searchsorted(issues, rows['issuance'])
        result[wi, rows['age'], rows['location'], k] = prediction
        print(f'Prepared causal nowcasts: {name}', flush=True)
    return result


def cached_nowcasts(panel, folder, dataset_hash, mode='adaptive_chain', growth_bandwidth=.2, penalty=10., growth_weight=1., residual_halflife=26., features_mode="basic", gate="none", strength=1.):
    suffix = '' if mode == 'adaptive_chain' else f'-{mode}-g{growth_bandwidth:g}-p{penalty:g}-w{growth_weight:g}-h{residual_halflife:g}-{features_mode}-{gate}-s{strength:g}'
    path = folder / f'replay-nowcasts{suffix}.npz'
    stamp = dict(dataset_sha256=dataset_hash, mode=mode, growth_bandwidth=growth_bandwidth, penalty=penalty,
                 growth_weight=growth_weight,residual_halflife=residual_halflife,features_mode=features_mode,gate=gate,strength=strength,
                 model_sha256=sha256(Path(__file__).parents[1] / 'model' / 'finalization.py'),
                 context_sha256=sha256(Path(__file__).parents[1] / 'model' / 'context_nowcast.py'),
                 rows_sha256=sha256(Path(__file__).parents[1] / 'dataset' / 'finalization.py'),
                 replay_sha256=sha256(__file__))
    with (folder / f'replay-nowcasts{suffix}.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if path.exists():
            with np.load(path) as saved:
                if json.loads(str(saved['metadata'])) != stamp:
                    raise ValueError('Nowcast cache provenance changed; use a new experiment')
                return saved['predictions']
        issues = panel['issuance_dates'].astype(str)
        wanted = [d for d in issues if cv.season(str(np.datetime64(d) - np.timedelta64(4, 'D')))
                  in ('2024-2025', '2025-2026')]
        predictions = selected_nowcasts(panel, wanted, mode, growth_bandwidth, penalty, growth_weight, residual_halflife, features_mode, gate, strength)
        # Check against the actual selected model on every saved comparable cell.
        reference = Path('data/experiments/seasonal-nowcast-causal-20261001')
        import pandas as pd
        dates = panel['dates'].astype(str)
        locations = list(panel['locations'].astype(str))
        names = list(panel['target_names'].astype(str))
        checked = 0
        if mode == 'adaptive_chain':
            reference_files = reference.glob('*/s42/attempt-001/eval_202[45]-202[56]/finalizations.csv.gz')
        else:
            from tapestry.model.scenario import Scenario
            pinned=json.loads((folder/'replay-source.json').read_text())
            records=pinned.get('nowcaster_references',[pinned.get('nowcaster_reference')])
            expected=dict(model=mode,growth=growth_bandwidth,penalty=penalty,growth_weight=growth_weight,
                          residual_halflife=residual_halflife,features=features_mode,gate=gate,strength=strength)
            matches=[r for r in records if r and all(getattr(Scenario.from_string(r['scenario']),'finalization_'+key)==value
                                                    for key,value in expected.items())]
            if len(matches)!=1:
                raise ValueError('Candidate replay requires one exact completed nowcaster reference')
            record=matches[0]
            run = Scenario.from_string(record['scenario']).run_id
            reference_files = (Path(record['root'])/run/'s42'/'attempt-001').glob('eval_*/finalizations.csv.gz')
        for saved in reference_files:
            frame = pd.read_csv(saved)
            frame = frame[frame.issuance.isin(wanted)]
            wi = np.searchsorted(issues, frame.issuance)
            li = frame.location.map({v:i for i,v in enumerate(locations)}).to_numpy()
            ki = frame.signal.map({v:i for i,v in enumerate(names)}).to_numpy()
            values = predictions[wi, frame.age.to_numpy(), li, ki]
            if not np.allclose(values, frame.prediction, rtol=2e-6, atol=2e-6):
                raise ValueError('Replay nowcasts differ from selected completed nowcasts')
            checked += len(frame)
        if not checked:
            raise ValueError('Selected-model prediction artifacts required for parity check')
        save(folder / f'replay-nowcast-parity{suffix}.json', dict(checked_cells=checked, **stamp))
        np.savez_compressed(path, predictions=predictions, metadata=json.dumps(stamp))
        return predictions


def replay_episode(episode, panel, covariate_names, mode, corrections=None, flags='native', scheduled=False):
    """Replace inputs only. Labels and their masks stay exactly as in B2 CV."""
    if mode == 'finalized':
        return encode_flags(episode, flags)
    dates = panel['dates'].astype(str)
    issue = str(np.datetime64(episode['context_dates'][-1]) + np.timedelta64(4, 'D'))
    issues = panel['issuance_dates'].astype(str)
    wi = np.searchsorted(issues, issue)
    if wi >= len(issues) or issues[wi] != issue:
        raise ValueError(f'No matching vintage for {issue}')
    ti = np.searchsorted(dates, episode['context_dates'])
    if (ti >= len(dates)).any() or not np.array_equal(dates[ti], episode['context_dates']):
        raise ValueError('Replay context does not align to panel dates')
    raw = panel['asof_targets'][wi, ti].copy()
    observed = np.isfinite(raw)
    if mode == 'nowcast':
        if corrections is None:
            raise ValueError('Nowcast replay needs causal corrections')
        recent = min(8, len(raw))
        raw[-recent:] = corrections[wi, np.arange(recent - 1, -1, -1)]
    if scheduled:
        # Numerical proxy for assumed availability, never a training revision pair.
        raw = np.where(observed, raw, panel['targets'][ti])
    values = np.moveaxis(raw, -1, -2)
    result = dict(episode, values=np.nan_to_num(values), available=np.isfinite(values),
                  known_final=np.zeros_like(values, dtype=bool), issuance=issue)
    if covariate_names and not scheduled:
        v, a = select_covariates(panel['asof_covariates'][wi, ti],
            panel['asof_covariates_national'][wi, ti], panel['covariate_names'],
            panel['covariate_national_names'], panel['locations'], covariate_names)
        result['covariates'] = np.stack((v, a), axis=-2)
    return encode_flags(result, flags)


def encode_flags(episode, flags):
    """Counterfactual feature encoding only; never change values or availability."""
    if flags == 'native':
        return episode
    if flags not in ('available', 'off'):
        raise ValueError(f'Unknown flag encoding: {flags}')
    return dict(episode, known_final=episode['available'].copy() if flags == 'available'
                else np.zeros_like(episode['available'], dtype=bool))


def evaluate_replay(scenario, seed, held_out, members, device, output, dataset):
    source = replace(scenario, replay_from='', replay_inputs='none', replay_flags='native',
                     replay_nowcaster='selected', replay_growth=.2, replay_schedule=False, replay_penalty=10., replay_growth_weight=1., replay_residual_halflife=26.,
                     replay_features="basic",replay_gate="none",replay_uncertainty=0.,replay_strength=1.)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    folder = output.parents[3]
    pinned = json.loads((folder / 'replay-source.json').read_text())
    key = f'{source.run_id}/s{seed}/{held_out}'
    record = pinned['checkpoints'][key]
    path = Path(record['path'])
    if sha256(path) != record['sha256']:
        raise ValueError(f'Source checkpoint changed: {path}')
    dataset_hash = sha256(dataset)
    if pinned['dataset_sha256'] != dataset_hash:
        raise ValueError('Replay dataset differs from original B2 inputs')
    saved = torch.load(path, map_location='cpu', weights_only=False)
    original = saved['metadata']
    if original['scenario'] != source.scenario_string or original['held_out_season'] != held_out or original['seed'] != seed:
        raise ValueError('Source checkpoint scenario, seed, or CV season differs')
    panel = load(dataset)
    fold = cv.fold(panel, source, held_out)
    mode = 'adaptive_chain' if scenario.replay_nowcaster == 'selected' else scenario.replay_nowcaster
    corrections = cached_nowcasts(panel, folder, dataset_hash, mode, scenario.replay_growth, scenario.replay_penalty,
                                  scenario.replay_growth_weight, scenario.replay_residual_halflife, scenario.replay_features, scenario.replay_gate, scenario.replay_strength) if scenario.replay_inputs == 'nowcast' else None
    eps = [replay_episode(e, panel, covariate_names_for(source.covariate_set), scenario.replay_inputs, corrections,
                         scenario.replay_flags, scheduled=scenario.replay_schedule)
           for e in fold.score]
    if scenario.replay_uncertainty:
        from tapestry.model.revision_uncertainty import RevisionUncertainty,trajectory_distribution_scores
        from .finalization import reporting_support
        import pandas as pd
        uncertainty=RevisionUncertainty(panel,corrections)
        diagnostics=[];scores=[]
        for e in eps:
            histories,scale,record=uncertainty.draw(e['issuance'],e['values'],members,scenario.replay_uncertainty)
            e['history_samples']=histories
            diagnostics.append(dict(issuance=e['issuance'],**record))
            ti=np.searchsorted(panel['dates'].astype(str),e['context_dates'])
            truth=np.moveaxis(panel['targets'][ti],-1,-2)
            scored=trajectory_distribution_scores(histories,truth,e['values'],scale)
            for k,name in enumerate(panel['target_names']):
                for l,location in enumerate(panel['locations']):
                    scores.append(dict(signal=name,location=location,issuance=e['issuance'],
                        boundary=e['context_dates'][-1],**{n:v[k,l] for n,v in scored.items()}))
        save(output/'revision-uncertainty.json',dict(strength=scenario.replay_uncertainty,
            empirical_members=members,history_seed=42,diagnostics=diagnostics,
            labels='actual delay-12 reports, original causal predictions',
            weighting='104-week window; 26-week recency half-life; weighted median centered log errors',
            missing_donor='zero perturbation; fewer than twelve observed donor errors stays deterministic',
            scale='causal donor maturity Q95; raw observed-history Q95 before donor support; floor one count or 0.0001 ED fraction'))
        scored=reporting_support(panel,pd.DataFrame(scores))
        scored.to_csv(output/'trajectory-distributions.csv.gz',index=False)
    model = load_model(saved).to(device)
    torch.manual_seed(seed + 1000)
    evaluate(model, eps, members, device, output)
    metadata = dict(original, scenario=scenario.scenario_string, run_id=scenario.run_id,
                    config=asdict(scenario), replay=dict(mode=scenario.replay_inputs, source=record,
                    fixed_weights=True, target_correction_weeks=8, vintage_history_weeks=source.lookback,
                    all_covariates_vintaged=scenario.replay_inputs != 'finalized' and not scenario.replay_schedule,
                    scheduled_availability=scenario.replay_schedule, nowcaster=mode,
                    revision_uncertainty=scenario.replay_uncertainty,
                    missing_report_values='frozen_final_proxy' if scenario.replay_schedule else 'unfilled',
                    finality_encoding=scenario.replay_flags,
                    finality_encoding_is_diagnostic=scenario.replay_flags != 'native',
                    paired_evaluation_seed=seed + 1000),
                    eval_members=members, fold=fold.info)
    if scenario.replay_inputs == 'finalized':
        with np.load(path.parent / 'forecasts.npz') as old, np.load(output / 'forecasts.npz') as new:
            for name in ('truth', 'mask', 'context_end', 'target_dates', 'locations'):
                if not np.array_equal(old[name], new[name]):
                    raise ValueError(f'Finalized replay changed original {name}')
            delta = np.abs(new['quantiles'] - old['quantiles'])
            metadata['replay']['original_quantiles_max_abs_difference'] = float(delta.max())
    save(output / 'manifest.json', metadata)
    torch.save(checkpoint(model, metadata), output / 'model.pt')
    return output / 'model.pt'
