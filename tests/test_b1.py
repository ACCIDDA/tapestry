"""B1 scientific invariants: vintage leakage, units, masks, weights and pairing."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tapestry.model_data import wednesday as wd
from tapestry.models.b1 import B1, draw_dropout
from tapestry.models.b1_run import task_weights, unique_truth


def test_wednesday_snapshot_fallback_retractions_and_support():
    archive = wd.VintageArchive()
    # Older period has no Hub coverage and may use Delphi; an interior hole may not.
    for day in ('2023-10-28', '2023-11-04', '2023-11-11', '2023-11-18'):
        archive.add('delphi_nhsn', '2023-11-21', day, 0, 'NC', 80)
    archive.add('hub_flusight_current', '2023-11-15', '2023-11-04', 0, 'NC', 20)
    archive.add('hub_flusight_current', '2023-11-15', '2023-11-11', 0, 'NC', 30)
    # New full snapshot removes Nov 11 and explicitly nulls Nov 18.
    archive.add('hub_flusight_current', '2023-11-22T22:00:00', '2023-11-04', 0, 'NC', 21)
    archive.add('hub_flusight_current', '2023-11-22T22:00:00', '2023-11-18', 0, 'NC', None)
    archive.add('hub_flusight_current', '2023-11-23', '2023-11-18', 0, 'NC', 900)
    archive.add('delphi_nssp', '2023-11-21', '2023-11-04', 5, 'NC', .2)
    days = ('2023-10-28', '2023-11-04', '2023-11-11', '2023-11-18')
    x, a, p, r = archive.panel(days, ('NC',), archive.resolve('2023-11-22'))
    np.testing.assert_array_equal(x[:, 0, 0], [80, 21, 0, 0])
    np.testing.assert_array_equal(a[:, 0, 0], [True, True, False, False])
    assert r[2, 0, 0] == r[3, 0, 0] == 2
    assert not a[1, 5, 0] and r[1, 5, 0] == 4
    # Conflicting revisions remain missing, including after another identical row.
    archive.add('delphi_nhsn', '2023-11-21', '2023-10-28', 0, 'NC', 81)
    archive.add('delphi_nhsn', '2023-11-21', '2023-10-28', 0, 'NC', 80)
    assert not archive.panel(days, ('NC',), archive.resolve('2023-11-22'))[1][0, 0, 0]
    context, targets = wd.output_dates('2023-11-22')
    assert context[-1] == '2023-11-18'
    assert targets == ('2023-11-11', '2023-11-18', '2023-11-25', '2023-12-02', '2023-12-09', '2023-12-16')
    with pytest.raises(ValueError):
        archive.add('delphi_nhsn', '2023-11-22', '2023-11-05', 0, 'NC', 1)


def test_selected_native_units_source_fill_and_date_alias(tmp_path, monkeypatch):
    rows = {
        'delphi_nssp': [dict(signal='pct_ed_visits_influenza', geo_type='state', geo_value='nc',
                            fill_method='source', reference_time='2023-11-18', report_time='2023-11-22', value='2')],
        'hub_covid_current': [dict(target='wk inc covid prop ed visits', location='37',
                                  target_end_date='2023-11-18', as_of='2023-11-22', observation=.03)],
    }
    for key in rows:
        path = tmp_path / 'raw' / key / 'snapshots' / 'test'
        path.mkdir(parents=True);(path / 'manifest.json').write_text('{}')
    class Selected:
        audit = []
        def __init__(self, root): pass
        def selected_tables(self, dataset_key):
            if dataset_key not in rows: return
            hub = dataset_key.startswith('hub_')
            yield SimpleNamespace(snapshot_id='test', source_path='test',
                event_date_column='date' if hub else 'reference_time', vintage_column='as_of' if hub else 'report_time',
                geographic_resolutions=('state',), iter_rows=lambda: iter(rows[dataset_key]))
    monkeypatch.setattr(wd, 'SelectedData', Selected)
    archive = wd.read_archive(tmp_path)
    x, a, _, _ = archive.panel(('2023-11-18',), ('NC',), archive.resolve('2023-11-22'))
    assert x[0, 3, 0] == pytest.approx(.02)
    assert x[0, 4, 0] == pytest.approx(.03)
    assert a[0, 3:5, 0].all()


@pytest.mark.parametrize('options', [
    {}, {'encoder': 'conv'}, {'encoder': 'multiscale_conv'},
    {'encoder': 'conv', 'spatial': 'joint_location_target', 'head_sharing': 'pathogen',
     'decoder': 'residual2', 'heads': 'state_us', 'location_embedding': 4},
])
def test_hidden_values_do_not_leak_through_transforms_anchors_or_slopes(options):
    torch.manual_seed(1)
    for c in (0, 3):
        model = B1(c, {'NC': 1000000}, ['NC'], width=8, **options)
        x = torch.rand(2, 12, 6, 1)
        a = torch.ones_like(x, dtype=torch.bool)
        d = torch.zeros_like(a);d[:, -3:, :, :] = True;d[0, :, c] = True
        a[1, :2] = False
        hidden = ~a | d
        altered = x.clone();altered[hidden] = float('nan')
        z = torch.randn(4, 2, 16);zf = torch.randn_like(z);cal = torch.zeros(2, 3)
        expected = model(x, a, cal, dropout=d, z_recent=z, z_future=zf)
        actual = model(altered, a, cal, dropout=d, z_recent=z, z_future=zf)
        torch.testing.assert_close(actual, expected)
        assert actual.isfinite().all()
        actual[:, :, 2:].sum().backward()
        assert sum(p.grad.abs().sum() for p in model.recent.parameters() if p.grad is not None) > 0
        assert model.baseline.weight.grad.abs().sum() > 0


def test_member_pairing_and_future_noise():
    torch.manual_seed(7)
    model = B1(0, {'NC': 1000000}, ['NC'], width=8)
    x = torch.rand(1, 12, 6, 1);a = torch.ones_like(x, dtype=torch.bool);cal=torch.zeros(1,3)
    zn=torch.randn(5,1,16);zf=torch.randn_like(zn)
    original=model(x,a,cal,z_recent=zn,z_future=zf)
    permutation=torch.tensor([4,2,0,3,1])
    reordered=model(x,a,cal,z_recent=zn[permutation],z_future=zf[permutation])
    torch.testing.assert_close(reordered,original[permutation])
    changed=model(x,a,cal,z_recent=zn,z_future=zf+2)
    torch.testing.assert_close(changed[:,:,:2],original[:,:,:2])
    assert not torch.allclose(changed[:,:,2:],original[:,:,2:])
    h,_,_=model.encode(x,a,cal)
    recent=torch.randn(5,1,2,1,1,requires_grad=True)
    future=model.forecast_from_recent(h,recent,zf)
    future[2].sum().backward()
    assert recent.grad[2].abs().sum()>0
    assert recent.grad[[0,1,3,4]].abs().sum()==0


def test_dropout_only_hides_available_cells_and_keeps_task_weights_fixed():
    a=np.ones((80,12,6,2),dtype=bool);a[:,:4,5]=False
    for scenario in ('natural','recent','gap','outage'):
        d=draw_dropout(a,np.random.default_rng(3),scenario=scenario)
        assert not (d & ~a).any()
        np.testing.assert_array_equal(d,draw_dropout(a,np.random.default_rng(3),scenario=scenario))
        if scenario=='natural':assert not d.any()
        else:assert d.any()
    episodes=[]
    for issuance in ('2023-11-22','2024-11-20'):
        context,targets=wd.output_dates(issuance)
        y=np.ones((6,6,2,2),np.float32)
        y[-1,0,1,0]=0
        episodes.append(dict(Y=y,target_dates=targets,locations=('NC','US')))
    w=task_weights(episodes,0)
    assert w[:,:2].sum()==pytest.approx(.5)
    assert w[:,2:].sum()==pytest.approx(.5)
    np.testing.assert_allclose(w.sum((0,1,2)),[.8,.2],rtol=1e-6)
    assert not w[:,-1,0,0].any()
    before=w.copy()
    for e in episodes:e['X']=np.zeros((12,6,2,2))
    np.testing.assert_array_equal(before,task_weights(episodes,0))
    assert unique_truth(episodes).shape[1:]==(6,2,2)


def test_grouped_loss_keeps_target_geography_and_task_weights():
    from tapestry.models.b1_scenarios import B1Scenario
    context, dates = wd.output_dates('2025-01-08')
    episodes = [dict(Y=np.ones((6, 6, 2, 2), np.float32), target_dates=dates, locations=('NC', 'US'))]
    w = task_weights(episodes, [0, 3])
    # Admission coefficient 1, ED .5, with each task retaining half the mass.
    np.testing.assert_allclose(w.sum((0, 1, 3)), [2/3, 0, 0, 1/3, 0, 0], rtol=1e-6)
    np.testing.assert_allclose(w.sum((0, 1, 2)), [.8, .2], rtol=1e-6)
    assert w[:, :2].sum() == pytest.approx(.5)
    assert w[:, 2:].sum() == pytest.approx(.5)
    a = np.ones((100, 12, 6, 2), bool)
    off = draw_dropout(a, np.random.default_rng(5), B1Scenario(mask_rate=0.).mask_probabilities)
    on = draw_dropout(a, np.random.default_rng(5), B1Scenario(mask_rate=1.).mask_probabilities)
    assert not off.any()
    assert on.any((1, 2, 3)).all()
    # This is a probability per episode, not independent cell dropout.
    s = B1Scenario(mask_rate=.25)
    np.testing.assert_allclose(s.mask_probabilities, [.75, .125, .075, .05])


def test_grouped_predictions_return_canonical_channel_order():
    from tapestry.models.b1_run import sample
    context, dates = wd.output_dates('2025-01-08')
    e = dict(X=np.ones((12, 6, 2, 1), np.float32), Y=np.ones((6, 6, 2, 1), np.float32),
             context_dates=context, target_dates=dates)
    class ConstantComponent:
        config = {'lookback': 12}
        def __init__(self, targets): self.targets = targets
        def __call__(self, x, available, cal, members, dropout):
            return torch.tensor(self.targets, dtype=x.dtype)[None, None, None, :, None].expand(members, len(x), 6, -1, 1)
    models = [ConstantComponent(group) for group in ([0, 3], [1, 4], [2, 5])]
    samples, _ = sample(models, [e], members=5, seed=42, device='cpu', sample_batch=2)
    np.testing.assert_array_equal(samples[0, 0, 0, :, 0], np.arange(6))


def test_local_and_national_stage_noise_is_fixed_and_member_paired():
    torch.manual_seed(3)
    model = B1([0, 3], {'NC': 1000000, 'US': 300000000}, ['NC', 'US'], width=8,
               encoder='conv', decoder='residual2', noise='local', us_error='shared_factor', heads='state_us')
    x = torch.rand(2, 12, 6, 2);a = torch.ones_like(x, dtype=torch.bool);cal = torch.zeros(2, 3)
    generator = torch.Generator().manual_seed(123)
    fixed = {}
    for stage in ('recent', 'future'):
        for name, value in model.draw_noise(4, 2, generator).items():
            fixed[f'{name}_{stage}'] = value
    expected = model(x, a, cal, **fixed)
    state = torch.random.get_rng_state().clone()
    actual = model(x, a, cal, **fixed)
    torch.testing.assert_close(expected, actual)
    assert torch.equal(torch.random.get_rng_state(), state)
    perm = torch.tensor([3, 1, 0, 2])
    reordered = model(x, a, cal, **{k: v[perm] for k, v in fixed.items()})
    torch.testing.assert_close(reordered, expected[perm])
    changed = model(x, a, cal, **{**fixed, 'local_future': fixed['local_future'] + 2})
    torch.testing.assert_close(changed[:, :, :2], expected[:, :, :2])
    assert not torch.allclose(changed[:, :, 2:], expected[:, :, 2:])


@pytest.mark.parametrize('count_transform', ['sqrt', 'fourth_root', 'log1p'])
@pytest.mark.parametrize('ed_transform', ['linear', 'logit', 'fourth_root'])
def test_working_space_roundtrip_preserves_native_units(count_transform, ed_transform):
    model = B1([0, 3], {'NC': 1000000}, ['NC'], width=8, count_transform=count_transform,
               ed_transform=ed_transform, input_scale=[[2.]] * 6)
    x = torch.full((1, 12, 6, 1), .02);x[:, :, :3] = 50.
    _, anchors, _ = model.encode(x, torch.ones_like(x, dtype=torch.bool), torch.zeros(1, 3))
    native = model.working_to_native(anchors)
    torch.testing.assert_close(native, x[:, -2:, [0, 3]], rtol=1e-5, atol=1e-6)


def test_grouped_forecasts_condition_only_on_own_target_corrections():
    model = B1([0, 3], {'NC': 1000000}, ['NC'], width=8, head_sharing='shared')
    x = torch.rand(1, 12, 6, 1)
    h, _, _ = model.encode(x, torch.ones_like(x, dtype=torch.bool), torch.zeros(1, 3))
    recent = torch.randn(3, 1, 2, 2, 1, requires_grad=True)
    future = model.forecast_from_recent(h, recent, torch.randn(3, 1, 16))
    future[1, :, :, 0].sum().backward()
    assert recent.grad[1, :, :, 0].abs().sum() > 0
    assert recent.grad[:, :, :, 1].abs().sum() == 0
    assert recent.grad[[0, 2]].abs().sum() == 0


def test_b1_rank_keeps_nowcasting_separate_and_normalizes_units():
    import pandas as pd
    from tapestry.models.b1_experiment import ranking_tables, totals
    rows = []
    for config_id, tasks in [('direct', ['forecast']), ('two', ['forecast', 'nowcast'])]:
        for task in tasks:
            # Equal normalized errors despite different native units.
            for target, scale, weight in [('admissions', 100., 2 / 3), ('ed', .01, 1 / 3)]:
                error = 2. if task == 'forecast' else 5.
                rows.append(dict(config_id=config_id, scenario_string=config_id, seed=42, stress='natural', task=task,
                    issuance_date='2025-08-06', target_date='2025-08-09' if task == 'forecast' else '2025-08-02',
                    target=target, location='US', horizon=0 if task == 'forecast' else -1, season='2025-2026',
                    observed=scale, loss_scale=scale, objective_weight=weight, crps=error * scale,
                    wis=error * scale, coverage_50=1., coverage_95=1.))
    frame = pd.DataFrame(rows)
    runs, target, digests = totals(frame)
    _, ranks, _ = ranking_tables(runs, target, [digests])
    np.testing.assert_allclose(ranks.loc[ranks.task == 'forecast', 'score_mean'], 2.)
    np.testing.assert_allclose(ranks.loc[ranks.task == 'nowcast', 'score_mean'], 5.)
    assert ranks.score_sd.isna().all()  # One seed does not establish zero variance.
    # A run scoring different truth/scales is rejected by its per-task digest.
    changed = frame.copy()
    changed.loc[changed.config_id == 'direct', 'loss_scale'] *= 2
    other_runs, other_target, other_digests = totals(changed)
    with pytest.raises(ValueError, match='different cells, truth, scales or weights'):
        ranking_tables(pd.concat([runs, other_runs]), pd.concat([target, other_target]),
                       [digests, other_digests])
    # Dropping a cell breaks the normalization that each task's weights sum to one.
    short_runs, short_target, short_digests = totals(frame.drop(index=0))
    with pytest.raises(ValueError, match='weights must sum to one'):
        ranking_tables(short_runs, short_target, [short_digests])


def test_b1_hub_export_maps_only_future_weeks_and_keeps_ed_proportions(tmp_path):
    from tapestry.models.b1_hubs import export_forecasts
    from tapestry.models.quantiles import LEVELS
    q = np.ones((len(LEVELS), 1, 6, 6, 2))
    q[:, :, :, 3:] = .02
    dates = np.array([['2025-07-26', '2025-08-02', '2025-08-09', '2025-08-16', '2025-08-23', '2025-08-30']])
    np.savez(tmp_path / 'forecasts-demo-s42-natural.npz', quantiles=q, horizons=np.arange(-2, 4),
             quantile_levels=LEVELS, target_dates=dates, issuance_dates=['2025-08-06'], locations=['US', 'NC'])
    frames = export_forecasts(tmp_path, dict(run_id='demo', seed=42),
                             dict(evaluation_start='2025-08-06', evaluation_end='2025-08-23'))
    assert len(frames) == 6
    for (_, target), frame in frames.items():
        assert set(frame.reference_date) == {'2025-08-09'}
        assert set(frame.horizon) == {0, 1, 2}
        assert set(frame.location) == {'US', '37'}
        np.testing.assert_allclose(frame['q0.5'], .02 if 'prop' in target else 1.)


def test_b1_season_folds_exclude_held_out_and_hidden_weeks_from_fitting():
    """B0's leakage rule: a held-out or hidden week never informs a fold's fit.

    Checked on inputs as well as labels, because B1 conditions on real Wednesday
    vintages: zeroing a context week is what mirrors B0 zeroing its panel.
    """
    from datetime import date
    from tapestry.model_data.finalized import season
    from tapestry.model_data.wednesday import WednesdayDataset
    from tapestry.models.b1_seasons import fold, hidden_weeks
    from tapestry.models.season_cv import SEASONS

    ds = WednesdayDataset.load('data/processed/build_b1_wednesday.npz')
    for held in SEASONS:
        fitting, validation, evaluation, _ = fold(ds, held)
        hidden = hidden_weeks(ds, held)
        for episode in fitting:
            for h, day in enumerate(episode['target_dates']):
                if episode['Y'][h, :, 1].any():
                    assert season(date.fromisoformat(str(day))) != held
                    assert str(day) not in hidden
            for j, day in enumerate(episode['context_dates']):
                if str(day) in hidden or season(date.fromisoformat(str(day))) == held:
                    assert not episode['X'][j, :, 1].any(), 'hidden/held-out week visible as a fitting input'
        # Validation scores only hidden weeks; evaluation only the held-out season.
        for episode in validation:
            for h, day in enumerate(episode['target_dates']):
                if episode['Y'][h, :, 1].any():
                    assert str(day) in hidden
        for episode in evaluation:
            for h, day in enumerate(episode['target_dates']):
                if episode['Y'][h, :, 1].any():
                    assert season(date.fromisoformat(str(day))) == held


def test_b1_season_fold_hub_export_keeps_only_the_held_out_season(tmp_path):
    """A fold's 6-week window spans seasons; only the held-out one is out-of-sample."""
    from tapestry.models.b1_hubs import export_forecasts
    from tapestry.models.quantiles import LEVELS

    seasons = ('2023-2024', '2024-2025')
    # Each fold's last origin reaches into the next season, as real folds do.
    spans = {'2023-2024': ('2024-07-27', '2024-08-03'), '2024-2025': ('2025-07-26', '2025-08-02')}
    run_id, seed = 'probe-run', 42
    for held, (inside, outside) in spans.items():
        folder = tmp_path / f'eval_{held}'
        folder.mkdir()
        np.savez_compressed(folder / f'forecasts-{run_id}-s{seed}.npz'.replace('.npz', '-natural.npz'),
            quantiles=np.ones((len(LEVELS), 1, 4, 6, 1)), quantile_levels=LEVELS,
            truth=np.ones((1, 4, 6, 1)), mask=np.ones((1, 4, 6, 1), bool),
            target_dates=np.array([[inside, inside, outside, outside]]),
            issuance_dates=np.array(['2024-07-24']), locations=np.array(['US']),
            channels=np.array(['nhsn_flu_admissions'] * 6), horizons=np.arange(4))
    metadata = dict(run_id=run_id, seed=seed, protocol='season_cv', seasons=list(seasons))
    frames = export_forecasts(tmp_path, metadata, dict(evaluation_start=None, evaluation_end=None))
    assert frames, 'expected exported Hub frames'
    # Nothing from a season the fold trained on, and no fold collides with another.
    assert {label for label, _ in frames} <= set(seasons)
    for (label, _), frame in frames.items():
        assert set(frame.target_end_date) == {spans[label][0]}
