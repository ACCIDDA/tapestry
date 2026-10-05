"""Add point-baseline evaluations to completed fits without retraining them."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tapestry.dataset.build import load
from tapestry.dataset.finalization import signals, boundary_rows
from tapestry.model.baselinenowcast import predictions
from tapestry.model.scenario import Scenario
from .provenance import save, sha256, now


def run(folder):
    from .planner import read_jobs, attempts, check_pinned_inputs
    from .finalization import metrics, upper_bound
    settings = json.loads((folder / 'experiment.json').read_text())
    check_pinned_inputs(settings)
    panel = load(settings['dataset'])
    for job in read_jobs(folder):
        scenario = Scenario.from_string(job['scenario'])
        if scenario.task not in ('finalize', 'nowcast', 'pipeline'):
            continue
        for seed in job['seeds']:
            completed = [p for p in attempts(folder, job['scenario'], seed)
                         if (p / 'run.json').exists() and
                         json.loads((p / 'run.json').read_text()).get('status') == 'complete']
            if not completed:
                raise ValueError(f'No completed fit for {job["name"]}, seed {seed}')
            for fold in scenario.scored_seasons:
                output = completed[-1] / f'eval_{fold}'
                if scenario.task == 'finalize':
                    frame = pd.read_csv(output / 'finalizations.csv.gz')
                    frame['baselinenowcast'] = np.nan
                    frame['baselinenowcast_status'] = ''
                    frame['baselinenowcast_support'] = 0
                    keys = ['issuance', 'boundary', 'age', 'location']
                    for name, truth, asof, locations in signals(panel):
                        selected = frame.signal == name
                        if not selected.any():
                            continue
                        rows = boundary_rows(panel, name, truth, asof, locations,
                                             scenario.lookback, scenario.finalization_weeks)
                        index = pd.MultiIndex.from_arrays([rows['issuance'], rows['boundary'], rows['age'],
                                                           locations[rows['location']]], names=keys)
                        positions = index.get_indexer(pd.MultiIndex.from_frame(frame.loc[selected, keys]))
                        if (positions < 0).any():
                            raise ValueError(f'Cannot align saved nowcasts for {name}')
                        rows = {k: v[positions] if isinstance(v, np.ndarray) and k != 'locations' else v
                                for k, v in rows.items()}
                        np.testing.assert_allclose(rows['truth'], frame.loc[selected, 'truth'], rtol=1e-6)
                        point, status, support = predictions(panel, asof, rows,
                            frame.loc[selected, 'persistence'].to_numpy(), integer=name.startswith('nhsn_'))
                        cap = upper_bound(name)
                        if cap is not None:
                            point = np.minimum(point, cap)
                        frame.loc[selected, 'baselinenowcast'] = point
                        frame.loc[selected, 'baselinenowcast_status'] = status
                        frame.loc[selected, 'baselinenowcast_support'] = support
                    if not np.isfinite(frame.baselinenowcast).all():
                        raise ValueError('Baseline did not cover every saved cell')
                    frame.to_csv(output / 'finalizations.csv.gz', index=False)
                    scored = metrics(frame)
                    scored.to_csv(output / 'finalization-metrics.csv', index=False)
                    save(output / 'baselinenowcast_scores.json', dict(baseline='baselinenowcast_point_v2',
                        status_counts=frame.baselinenowcast_status.value_counts().to_dict(),
                        metrics=scored[scored.method == 'baselinenowcast'].to_dict('records')))
                else:
                    from tapestry.dataset import cv
                    from tapestry.model.network import load_model, IndependentBundle
                    from .nowcast_baseline import evaluate
                    ns = scenario.stage('nowcast') if scenario.task == 'pipeline' else scenario
                    model_path = output / ('nowcaster.pt' if scenario.task == 'pipeline' else 'model.pt')
                    model = load_model(torch.load(model_path, map_location='cpu', weights_only=False))
                    scales = (model.models[0].scale if isinstance(model, IndependentBundle) else model.scale).detach().numpy()
                    if scenario.task == 'pipeline':
                        output = output / 'nowcast'
                    eps = cv.fold(panel, ns, fold).score
                    with np.load(output / 'forecasts.npz') as saved:
                        np.testing.assert_array_equal(saved['target_dates'], [e['target_dates'] for e in eps])
                        np.testing.assert_array_equal(saved['locations'], eps[0]['locations'])
                        np.testing.assert_array_equal(saved['mask'], [e['target_available'] for e in eps])
                        np.testing.assert_allclose(saved['truth'], [e['target_values'] for e in eps])
                    evaluate(panel, eps, scales, output)
                sources = [Path(__file__), Path(__file__).with_name('nowcast_baseline.py'),
                           Path(__file__).parents[1] / 'model' / 'baselinenowcast.py']
                save(output / 'baselinenowcast-provenance.json', dict(created=now(),
                    dataset_sha256=settings['dataset_sha256'],
                    source_sha256={str(p): sha256(p) for p in sources},
                    baseline='baselinenowcast_point_v2', original_fit_unchanged=True))
                print(f'Stored baseline: {output}', flush=True)
