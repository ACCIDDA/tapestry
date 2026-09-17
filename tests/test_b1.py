"""B1 scientific invariants: vintage leakage, units, masks, weights and pairing."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tapestry.model_data import wednesday as wd
from tapestry.models.b1 import B1, draw_dropout
from tapestry.models.b1_run import task_weights, unique_truth


def test_final_history_and_recent_fallback_preserve_real_wednesday_reports():
    archive = wd.VintageArchive()
    context, targets = wd.output_dates('2023-11-22')
    for day in set(context + targets):
        archive.add('delphi_nhsn', '2023-12-20', day, 0, 'NC', 100)
    archive.add('delphi_nhsn', '2023-11-22', context[-2], 0, 'NC', 30)
    archive.add('delphi_nhsn', '2023-11-22', context[-3], 0, 'NC', 20)
    ds = wd.build_wednesday(start='2023-11-22', end='2023-11-22', truth_cutoff='2023-12-20',
                            locations=('NC',), archive=archive)
    a = ds.arrays
    np.testing.assert_array_equal(a['X_values'][0, -3:, 0, 0], [100, 30, 100])
    np.testing.assert_array_equal(a['X_final'][0, -3:, 0, 0], [True, False, True])
    assert not a['X_available'][:, :, 1:].any()
    assert not a['X_final'][:, :, 1:].any()
    np.testing.assert_array_equal(a['Y_recent'][0, :, 0, 0], [100, 100])
    # No eligible vintage at all still produces a supervised episode.
    earlier = wd.build_wednesday(start='2023-11-15', end='2023-11-15', truth_cutoff='2023-12-20',
                                 locations=('NC',), archive=archive)
    assert len(list(earlier.episodes())) == 1
    assert earlier.arrays['X_final'][0, -2:, 0, 0].all()


@pytest.mark.parametrize('ed_transform', ['linear', 'logit', 'fourth_root'])
def test_known_finals_bypass_nowcast_and_hidden_finals_do_not_leak(ed_transform):
    model = B1([0, 3], {'NC': 1000000}, ['NC'], width=8, ed_transform=ed_transform)
    x = torch.ones(1, 12, 6, 1) * .2
    x[:, :, :3] = 100
    x[:, -2, [0, 3]] = 0
    x[:, -1, 3] = 1
    a = torch.ones_like(x, dtype=torch.bool)
    known = torch.ones_like(a)
    cal = torch.zeros(1, 3)
    z = torch.randn(4, 1, 16)
    zf = torch.randn_like(z)
    result = model(x, a, cal, known_final=known, z_recent=z, z_future=zf)
    expected = x[:, -2:, [0, 3]][None].expand(4, -1, -1, -1, -1)
    torch.testing.assert_close(result[:, :, :2], expected, rtol=0, atol=0)
    torch.testing.assert_close(result, model(x, a, cal, known_final=known, z_recent=z + 10, z_future=zf))
    assert result.isfinite().all()
    d = torch.zeros_like(a); d[:, -2:] = True
    altered = x.clone(); altered[:, -2:] = float('nan')
    masked = model(x, a, cal, known_final=known, dropout=d, z_recent=z, z_future=zf)
    no_flags = known.clone(); no_flags[:, -2:] = False
    torch.testing.assert_close(masked, model(altered, a, cal, known_final=no_flags, dropout=d, z_recent=z, z_future=zf))
    masked[:, :, 2:].sum().backward()
    assert sum(p.grad.abs().sum() for p in model.recent.parameters() if p.grad is not None) > 0


def test_known_final_supervision_tracks_dropout_and_fold_masks():
    from tapestry.models.b1_run import supervision_mask
    from tapestry.models.b1_seasons import _mask_context
    context, dates = wd.output_dates('2025-01-08')
    x = np.ones((12, 6, 3, 2), np.float32)
    e = dict(X=x, Y=np.ones((6, 6, 2, 2), np.float32), context_dates=context,
             target_dates=dates, locations=('NC', 'US'))
    assert not supervision_mask(e)[:2].any()
    w = task_weights([e], 0)
    assert not w[:, :2].any()
    assert w[:, 2:].sum() == pytest.approx(.5)
    d = np.zeros((1, 12, 6, 2), bool); d[:, -2:] = True
    w = task_weights([e], 0, dropout=d)
    assert w[:, :2].sum() == pytest.approx(.5)
    assert w[:, 2:].sum() == pytest.approx(.5)
    np.testing.assert_allclose(w.sum((0, 1, 2)), [.8, .2], rtol=1e-6)
    blocked = _mask_context(e, set(context[-2:]))
    assert not blocked[-2:].any()
    assert supervision_mask({**e, 'X': blocked})[:2].all()


def test_scoring_exports_exclude_visible_finals_and_restore_hidden_ones(tmp_path, monkeypatch):
    from tapestry.models import b1_report, b1_run
    from tapestry.models.b1 import MASK_SCENARIOS
    context, dates = wd.output_dates('2025-01-08')
    e = dict(X=np.ones((12, 6, 3, 1), np.float32), Y=np.ones((6, 6, 2, 1), np.float32),
             context_dates=context, target_dates=dates, locations=('US',), issuance_date='2025-01-08')
    def sample(models, episodes, *, scenario, **kwargs):
        d = np.zeros((1, 12, 6, 1), bool)
        if scenario != 'natural':
            d[:, -2:] = True
        return np.ones((4, 1, 6, 6, 1)), d
    monkeypatch.setattr(b1_run, 'sample', sample)
    monkeypatch.setattr(b1_report, 'path_graph', lambda *args: None)
    model = SimpleNamespace(config={'direct': False}, scale=torch.ones(6, 1))
    ds = SimpleNamespace(locations=('US',), metadata={'history_mode': 'final_history_recent_final_fallback'})
    rows = b1_report.evaluate([model], [e], ds, SimpleNamespace(evaluation_members=4, device='cpu'), 'demo', 42, tmp_path)
    assert not any(r['task'] == 'nowcast' and r['stress'] == 'natural' for r in rows)
    for stress in MASK_SCENARIOS:
        with np.load(tmp_path / f'forecasts-demo-s42-{stress}.npz') as data:
            assert data['mask'][:, 2:].all()
            assert data['mask'][:, :2].all() == (stress != 'natural')
        if stress != 'natural':
            assert sum(r['objective_weight'] for r in rows if r['task'] == 'nowcast' and r['stress'] == stress) == pytest.approx(1)


@pytest.mark.parametrize('noise,us_error', [('global', 'none'), ('local', 'shared_factor')])
def test_direct_b1_is_b0_on_identical_masked_inputs(noise, us_error):
    """The input-vintage control must not change dynamics, anchors or decoding."""
    from tapestry.models.b0 import B0
    locations, population = ['NC', 'US'], {'NC': 10000000., 'US': 330000000.}
    options = dict(lookback=12, width=8, latent=4, count_transform='fourth_root',
        ed_transform='logit', geography=True, dynamics=True, noise=noise, us_error=us_error)
    torch.manual_seed(17)
    original = B0(populations=population, location_ids=locations, **options)
    torch.manual_seed(17)
    direct = B1([0, 3], population, locations, direct=True, **options)
    values = torch.rand(2, 12, 6, 2)
    available = torch.rand_like(values) > .2
    available[:, :, 0, 0] = False  # no focal history: the B0 prior must be preserved
    dropout = torch.rand_like(values) > .8
    visible = available & ~dropout
    values[~visible] = float('nan')  # masked values cannot leak through either path
    calendar = torch.zeros(2, 3)
    fixed = direct.draw_noise(3, 2, torch.Generator().manual_seed(42))
    expected = original(torch.stack((values, visible.float()), dim=3), calendar,
        locations=locations, z=fixed['z'], local_z=fixed['local'], national_z=fixed['national'])[:, :, :, [0, 3]]
    actual = direct(values, available, calendar, dropout=dropout,
        z_future=fixed['z'], local_future=fixed['local'], national_future=fixed['national'])
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_unchanged_versions_persist_until_revision_or_retraction():
    archive = wd.VintageArchive()
    day = '2023-11-04'
    for source, channel in [('delphi_nhsn', 0), ('hub_covid_current:git', 1), ('hub_rsv_current', 2)]:
        archive.add(source, '2023-11-07', day, channel, 'NC', 10)
        archive.add(source, '2023-11-30', day, channel, 'NC', None)
    for issuance in ('2023-11-08', '2023-11-15', '2023-11-22', '2023-11-29'):
        x, a, _, _ = archive.panel((day,), ('NC',), archive.resolve(issuance))
        np.testing.assert_array_equal(x[0, :3, 0], [10, 10, 10])
        assert a[0, :3, 0].all()
    assert not archive.panel((day,), ('NC',), archive.resolve('2023-11-30'))[1].any()
    # Missing a revision on Wednesday must never invoke recent-final filling.
    ds = wd.build_wednesday(start='2023-11-08', end='2023-11-15', truth_cutoff='2023-11-29',
                            locations=('NC',), archive=archive)
    for i, days in enumerate(ds.arrays['context_dates']):
        t = list(days).index(day)
        assert ds.arrays['X_available'][i, t, :3, 0].all()
        assert not ds.arrays['X_final'][i, t, :3, 0].any()


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
        def __call__(self, x, available, cal, members, dropout, known_final=None):
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


def test_nowcast_baseline_is_persistence_of_the_visible_value():
    """The nowcast denominator must be the last value visible on Wednesday.

    If it silently used the reference final instead, the denominator would be
    zero error and every model would look infinitely bad; if it used a later
    revision it would leak truth into the baseline. Both failures are silent in
    the ratio, so the baseline itself is pinned here.
    """
    from tapestry.models.season_cv import persistence
    x = np.zeros((4, 6, 2, 1))          # week, channel, value/mask, location
    x[:, :, 1] = 1                      # everything visible by default
    x[:, 0, 0, 0] = [10., 20., 30., 40.]
    # Channel 1's two most recent weeks are missing, so it falls back further.
    x[:, 1, 0, 0] = [7., 8., 9., 99.]
    x[2:, 1, 1, 0] = 0
    x[:, 2, 1, 0] = 0                   # channel 2 has no visible history at all
    values, available = persistence(x)
    assert values[0, 0] == 40. and available[0, 0]      # latest visible week
    assert values[1, 0] == 8. and available[1, 0]       # latest STILL-visible week
    assert not available[2, 0]                          # no baseline exists


def test_nowcast_scores_against_persistence_and_excludes_cells_without_history():
    """A deterministic baseline's WIS is its absolute error, and nothing else."""
    from tapestry.evaluation.totals import quantile_scores
    from tapestry.models.quantiles import LEVELS
    truth = np.array([100., 50.])
    baseline = np.array([90., 50.])
    flat = np.repeat(baseline[:, None], len(LEVELS), axis=1)
    scores = quantile_scores(flat, truth)
    np.testing.assert_allclose(scores.wis, np.abs(truth - baseline))
    np.testing.assert_allclose(scores.dispersion, 0.)
    # A point forecast covers an interval only when it is exactly right.
    np.testing.assert_allclose(scores.covered_95, [0., 1.])


def test_nowcast_relative_support_does_not_drop_unavailable_targets_silently(tmp_path):
    from tapestry.evaluation.nowcast import nowcast_cells, CHANNEL_TARGETS
    from tapestry.models.provenance import SEASONS
    from tapestry.models.quantiles import LEVELS
    for held in SEASONS:
        folder = tmp_path / f'eval_{held}'
        folder.mkdir()
        truth = np.full((1, 2, 6, 1), 5.)
        valid = np.zeros_like(truth, dtype=bool)
        valid[:, :, :2] = True
        baseline_mask = np.zeros((1, 6, 1), dtype=bool)
        baseline_mask[:, 0] = True
        np.savez(folder / 'forecasts-demo-s42-natural.npz', horizons=[-2, -1],
            quantiles=np.repeat(truth[None], len(LEVELS), axis=0), quantile_levels=LEVELS,
            truth=truth, mask=valid, locations=['US'],
            target_dates=[[f'{held[5:]}-01-06', f'{held[5:]}-01-13']],
            baseline=np.full((1, 6, 1), 4.), baseline_mask=baseline_mask)
    totals, audit = nowcast_cells(tmp_path, dict(run_id='demo', seed=42))
    assert audit['scored_cells'] == audit['excluded_no_history'] == 6
    assert totals.n.sum() == 6
    assert totals.model_wis.sum() == 0
    assert totals.ensemble_wis.sum() == 6
    covid = [r for r in audit['by_target_season'] if r['target'] == CHANNEL_TARGETS[1]]
    assert len(covid) == 3
    assert all(r['label_cells'] == r['excluded_no_history'] == 2 and r['scored_cells'] == 0 for r in covid)
    # Supplying one recent final per season must not create a perfect nowcast score,
    # even if an external caller left its label mask true in the export.
    for held in SEASONS:
        path = tmp_path / f'eval_{held}' / 'forecasts-demo-s42-natural.npz'
        with np.load(path) as data:
            arrays = dict(data)
        final = np.zeros((1, 12, 6, 1), bool); final[:, -1, 0] = True
        np.savez(path, **arrays, X_final=final)
    totals, audit = nowcast_cells(tmp_path, dict(run_id='demo', seed=42))
    assert audit['excluded_supplied_final'] == audit['scored_cells'] == 3
    assert audit['excluded_no_history'] == 6
    assert totals.n.sum() == 3


def test_b1_custom_b0_calendar_controls_views_and_hidden_weeks(tmp_path):
    """A custom B0 start must move B1's calendar and validation pattern together."""
    from datetime import date, timedelta
    from tapestry.model_data.finalized import FinalizedDataset
    from tapestry.models.b1_seasons import hidden_weeks
    from tapestry.models.season_cv import validation_split, SEASONS
    first, last = date(2023, 9, 30), date(2026, 8, 1)
    days = tuple((first + timedelta(weeks=i)).isoformat() for i in range((last-first).days // 7 + 1))
    b0 = FinalizedDataset(np.ones((len(days), 6, 2, 1), np.float32), days, ('NC',), {})
    path = tmp_path / 'b0.npz'
    b0.save(path)
    archive = wd.VintageArchive()
    for day in ('2023-09-02', *days):
        release = (date.fromisoformat(day) + timedelta(days=4)).isoformat()
        archive.add('delphi_nhsn', release, day, 0, 'NC', 10)
    raw = wd.build_wednesday(start='2023-09-06', end='2026-08-05', truth_cutoff='2026-08-05',
                             locations=('NC',), archive=archive, calendar_dataset=path)
    output = tmp_path / 'b1.npz'
    raw.save(output)
    view = wd.WednesdayDataset.load(output)
    assert view.calendar_weeks == days
    assert view.metadata['calendar_start'] == days[0]
    assert wd.WednesdayDataset.load(output, archive=True).arrays['issuance_dates'][0] == '2023-09-06'
    a = view.arrays
    outside = ~np.isin(a['context_dates'], days)
    assert not a['X_available'][outside].any()
    assert not a['X_values'][outside].any()
    labels = np.concatenate((a['Y_recent_valid'], a['Y_future_valid']), axis=1)
    assert not labels[~np.isin(a['target_dates'], days)].any()
    # Archive data outside the model calendar was not overwritten by the view.
    assert raw.arrays['X_available'][raw.arrays['context_dates'] == '2023-09-02'].any()
    for held in SEASONS:
        *_, info = validation_split(b0, held, lookback=12)
        assert hidden_weeks(view, held) == set(info['validation_weeks'])


def test_b1_hub_export_maps_only_future_weeks_and_keeps_ed_proportions(tmp_path):
    """Recent offsets are nowcasts and must never be relabelled as Hub forecasts.

    A Wednesday issuance maps to the FOLLOWING Saturday at horizon 0. Shifting
    that by a week, or letting offsets -2/-1 through, would score nowcasts
    against the ensemble's forecasts and silently flatter the model.
    """
    from tapestry.evaluation.hubs import export_b1
    from tapestry.models.quantiles import LEVELS
    q = np.ones((len(LEVELS), 1, 6, 6, 2))
    q[:, :, :, 3:] = .02
    dates = np.array([['2025-11-08', '2025-11-15', '2025-11-22', '2025-11-29', '2025-12-06', '2025-12-13']])
    for held in ('2023-2024', '2024-2025', '2025-2026'):
        folder = tmp_path / f'eval_{held}'
        folder.mkdir()
        # Only the 2025-2026 fold has target dates in its own held-out season.
        np.savez(folder / 'forecasts-demo-s42-natural.npz', quantiles=q, horizons=np.arange(-2, 4),
                 quantile_levels=LEVELS, target_dates=dates, issuance_dates=['2025-11-19'],
                 locations=['US', 'NC'])
    frames = export_b1(tmp_path, dict(run_id='demo', seed=42))
    for (held, target), frame in frames.items():
        if held != '2025-2026':
            assert frame.empty  # a trained-on season leaking through the output window
            continue
        assert set(frame.reference_date) == {'2025-11-22'}
        assert set(frame.horizon) == {0, 1, 2, 3}
        assert set(frame.location) == {'US', '37'}
        np.testing.assert_allclose(frame['q0.5'], .02 if 'prop' in target else 1.)


def test_b1_season_folds_exclude_held_out_and_hidden_weeks_from_fitting():
    """B0's leakage rule: a held-out or hidden week never informs a fold's fit.

    Checked on inputs, final flags and labels using a fully observed synthetic
    calendar, so this leakage check also runs without local research datasets.
    """
    from datetime import date, timedelta
    from tapestry.model_data.finalized import season
    from tapestry.models.b1_seasons import fold, hidden_weeks
    from tapestry.models.season_cv import SEASONS

    archive = wd.VintageArchive()
    day = date(2023, 8, 5)
    while day <= date(2026, 9, 26):
        for c in range(6):
            archive.add('delphi_nhsn' if c < 3 else 'delphi_nssp',
                        '2026-09-30', day.isoformat(), c, 'NC', 100 if c < 3 else .02)
        day += timedelta(weeks=1)
    ds = wd.build_wednesday(start='2023-08-09', end='2026-09-16', truth_cutoff='2026-09-30',
                            locations=('NC',), archive=archive)
    for held in SEASONS:
        fitting, validation, _, evaluation, _ = fold(ds, held)
        hidden = hidden_weeks(ds, held)
        for episode in fitting:
            for h, day in enumerate(episode['target_dates']):
                if episode['Y'][h, :, 1].any():
                    assert season(date.fromisoformat(str(day))) != held
                    assert str(day) not in hidden
            for j, day in enumerate(episode['context_dates']):
                if str(day) in hidden or season(date.fromisoformat(str(day))) == held:
                    assert not episode['X'][j, :, 1].any(), 'hidden/held-out week visible as a fitting input'
                    assert not episode['X'][j, :, 2].any(), 'hidden/held-out final flag visible as a fitting input'
        # Validation scores only hidden weeks; evaluation only the held-out season.
        for episode in validation:
            for h, day in enumerate(episode['target_dates']):
                if episode['Y'][h, :, 1].any():
                    assert str(day) in hidden
        for episode in evaluation:
            for h, day in enumerate(episode['target_dates']):
                if episode['Y'][h, :, 1].any():
                    assert season(date.fromisoformat(str(day))) == held


def _synthetic_fold_dataset():
    """Fully observed three-season calendar; no local research dataset needed."""
    from datetime import date, timedelta
    archive = wd.VintageArchive()
    day = date(2023, 8, 5)
    while day <= date(2026, 9, 26):
        for c in range(6):
            archive.add('delphi_nhsn' if c < 3 else 'delphi_nssp',
                        '2026-09-30', day.isoformat(), c, 'NC', 100 if c < 3 else .02)
        day += timedelta(weeks=1)
    return wd.build_wednesday(start='2023-08-09', end='2026-09-16', truth_cutoff='2026-09-30',
                              locations=('NC',), archive=archive)


def test_b1_refit_partition_restores_hidden_weeks_that_inner_fitting_hides():
    """The refit sees every training week; inner fitting and its scales do not.

    This is the partition half of B0's select-then-refit procedure: hidden weeks
    are withheld only to choose an epoch count, then restored for the final fit.
    """
    from datetime import date
    from tapestry.model_data.finalized import season
    from tapestry.models.b1_seasons import fold, hidden_weeks
    from tapestry.models.season_cv import SEASONS

    ds = _synthetic_fold_dataset()
    for held in SEASONS:
        fitting, validation, refit, _, info = fold(ds, held)
        hidden = hidden_weeks(ds, held)
        assert hidden, 'the fixture must exercise hidden weeks'
        # Inner fitting hides them; the refit supervises them again.
        inner_labelled = {str(d) for e in fitting for h, d in enumerate(e['target_dates'])
                          if e['Y'][h, :, 1].any()}
        refit_labelled = {str(d) for e in refit for h, d in enumerate(e['target_dates'])
                          if e['Y'][h, :, 1].any()}
        assert not (inner_labelled & hidden), 'hidden week supervised during inner fitting'
        assert hidden <= refit_labelled, 'refit must restore hidden-week supervision'
        # Neither ever supervises or conditions on the held-out season.
        for name, episodes in (('refit', refit), ('fitting', fitting)):
            for episode in episodes:
                for h, day in enumerate(episode['target_dates']):
                    if episode['Y'][h, :, 1].any():
                        assert season(date.fromisoformat(str(day))) != held, f'{name} leaks held-out label'
                for j, day in enumerate(episode['context_dates']):
                    if season(date.fromisoformat(str(day))) == held:
                        assert not episode['X'][j, :, 1].any(), f'{name} leaks held-out context'
                        assert not episode['X'][j, :, 2].any(), f'{name} leaks held-out final flag'
        # B0's validation may condition on earlier hidden weeks while scoring only them.
        visible = {str(d) for e in validation for j, d in enumerate(e['context_dates'])
                   if e['X'][j, :, 1].any()}
        assert visible & hidden, 'validation must be able to condition on earlier hidden weeks'
        assert info['refit_episodes'] >= info['fitting_episodes']


def test_b1_fold_origins_follow_calendar_membership_of_the_context_end():
    """B0 selects origins by calendar membership, not by remaining labels."""
    from datetime import date
    from tapestry.model_data.finalized import season
    from tapestry.models.b1_seasons import fold, hidden_weeks, season_weeks
    from tapestry.models.season_cv import SEASONS

    ds = _synthetic_fold_dataset()
    weeks = set(season_weeks(ds))
    for held in SEASONS:
        fitting, validation, refit, evaluation, _ = fold(ds, held)
        hidden = hidden_weeks(ds, held)
        training = {d for d in weeks if season(date.fromisoformat(d)) != held}
        held_weeks = {d for d in weeks if season(date.fromisoformat(d)) == held}
        for episode in fitting:
            assert str(episode['context_dates'][-1]) in training - hidden
        for episode in refit:
            assert str(episode['context_dates'][-1]) in training
        for episode in validation:
            assert str(episode['context_dates'][-1]) in training
            # B0 keeps only origins that can actually reach a hidden target.
            assert hidden & {str(d) for d in episode['target_dates']}
        for episode in evaluation:
            assert str(episode['context_dates'][-1]) in held_weeks


def test_b1_perturbing_held_out_values_cannot_change_fitting_or_refit_inputs():
    """Held-out observations may not reach inner or refit inputs, labels or scales."""
    import numpy as np
    from datetime import date
    from tapestry.model_data.finalized import season
    from tapestry.model_data.wednesday import WednesdayDataset
    from tapestry.models.b1_seasons import fold
    from tapestry.models.season_cv import SEASONS

    ds = _synthetic_fold_dataset()
    held = SEASONS[-1]
    arrays = {k: v.copy() for k, v in ds.arrays.items()}
    target_dates = arrays['target_dates']
    for i in range(target_dates.shape[0]):
        for h, day in enumerate(target_dates[i]):
            if season(date.fromisoformat(str(day))) == held:
                if h < 2:
                    arrays['Y_recent'][i, h] *= 7.5
                else:
                    arrays['Y_future'][i, h - 2] *= 7.5
    perturbed = WednesdayDataset(arrays, dict(ds.metadata))
    for name, a, b in zip(('fitting', 'validation', 'refit'),
                          fold(ds, held)[:3], fold(perturbed, held)[:3]):
        assert len(a) == len(b), f'{name} episode count changed'
        for ea, eb in zip(a, b):
            assert np.array_equal(ea['X'], eb['X']), f'{name} inputs saw held-out values'
            assert np.array_equal(ea['Y'], eb['Y']), f'{name} labels saw held-out values'


def test_b1_season_fold_hub_export_keeps_only_the_held_out_season(tmp_path):
    """A fold's 6-week window spans seasons; only the held-out one is out-of-sample.

    Without the filter a fold would contribute forecasts for weeks it trained on,
    and two folds would both claim the same target date, inflating the score with
    in-sample predictions.
    """
    from tapestry.evaluation.hubs import export_b1
    from tapestry.models.quantiles import LEVELS

    # Each fold's last origin reaches into the next season, as real folds do.
    # (last week inside the held-out season, first week of the next one)
    spans = {'2023-2024': ('2024-07-27', '2024-08-03'), '2024-2025': ('2025-07-26', '2025-08-02'),
             '2025-2026': ('2026-08-01', '2026-08-08')}
    run_id, seed = 'probe-run', 42
    for held, (inside, outside) in spans.items():
        folder = tmp_path / f'eval_{held}'
        folder.mkdir()
        np.savez_compressed(folder / f'forecasts-{run_id}-s{seed}-natural.npz',
            quantiles=np.ones((len(LEVELS), 1, 4, 6, 1)), quantile_levels=LEVELS,
            truth=np.ones((1, 4, 6, 1)), mask=np.ones((1, 4, 6, 1), bool),
            target_dates=np.array([[inside, inside, outside, outside]]),
            issuance_dates=np.array(['2024-07-24']), locations=np.array(['US']),
            channels=np.array(['nhsn_flu_admissions'] * 6), horizons=np.arange(4))
    frames = export_b1(tmp_path, dict(run_id=run_id, seed=seed))
    assert frames, 'expected exported Hub frames'
    # Nothing from a season the fold trained on, and no fold collides with another.
    assert {label for label, _ in frames} <= set(spans)
    for (label, _), frame in frames.items():
        assert set(frame.target_end_date) == {spans[label][0]}


def test_b1_refit_uses_a_fresh_model_and_exactly_the_selected_epochs(monkeypatch, tmp_path):
    """B0's procedure: select an epoch count, then refit a freshly seeded model.

    Spies on `fit_component` so the contract is checked without GPU training: the
    refit call must receive the selection's chosen epoch count, the refit partition
    and its own recomputed options, and the exported model must be the refit's.
    """
    import argparse
    import numpy as np
    import torch
    from tapestry.models import b1_run
    from tapestry.models.season_cv import SEASONS

    ds = _synthetic_fold_dataset()
    dataset = tmp_path / 'fixture.npz'
    ds.save(dataset)
    population = tmp_path / 'pop.csv'
    population.write_text('location,population\nNC,10000000\n')

    calls = []
    real_fit = b1_run.fit_component

    class Fake:
        config = dict(target=(0,), direct=True)
        def state_dict(self):
            return {'w': torch.zeros(1)}

    def spy(train, validation, targets, component, options, args, scenario, epochs=None):
        calls.append(dict(phase='select' if validation is not None else 'refit',
                          episodes=len(train), epochs=epochs, options=id(options),
                          scale=np.asarray(options['scale']).copy()))
        record = dict(targets=targets, best_epoch=3, selected_epoch=3,
                      phase='select' if validation is not None else 'refit',
                      epochs=scenario.epochs if epochs is None else epochs,
                      fitting_episodes=len(train), validation_episodes=0, history=[])
        audit = dict(dropout_packed=np.zeros((1, 1), np.uint8), dropout_shape=np.array([1, 1]),
                     train_issuance_dates=[e['issuance_date'] for e in train])
        return Fake(), record, audit

    monkeypatch.setattr(b1_run, 'fit_component', spy)
    monkeypatch.setattr(b1_run, 'load_models', lambda *a, **k: ([None], None))
    monkeypatch.setattr(b1_run, 'GROUPS', {'target': [(0,)]})
    import tapestry.models.b1_report as b1_report
    monkeypatch.setattr(b1_report, 'evaluate', lambda *a, **k: None)

    args = argparse.Namespace(dataset=str(dataset), population_file=str(population),
        held_out_season=SEASONS[-1], seed=42, device='cpu', retrospective=True,
        output=str(tmp_path / 'out'), eval_members=2, scenario=None, min_availability=0.)
    from tapestry.models.b1_scenarios import B1Scenario
    scenario = B1Scenario.from_string(
        'b1:v2:h12:tr_4rt:ed_logit:geo1:dyn1:lw_obj:enc_mlp:sp_none:hd_sh:dec_leg:'
        'nz_glob:us_none:z16:w64:ep100:pat30:bs8:m128:lr0.001:hs_sh:cal1:id0:'
        'fit_targ:vm256:wd0.0:pipe_direct:mask0.0:mr0.5:mg0.3:mo0.2')
    b1_run.train(args, scenario=scenario)

    assert [c['phase'] for c in calls] == ['select', 'refit'], 'expected one selection then one refit'
    select, refit = calls
    assert refit['epochs'] == 3, 'refit must run exactly the selected epoch count'
    assert refit['episodes'] > select['episodes'], 'refit must see the restored hidden weeks'
    assert refit['options'] != select['options'], 'refit must recompute its own options'
    # The constant-valued fixture gives both partitions the same Q95, so equality here
    # is expected; the separate options object above is what proves recomputation.
    assert np.array_equal(refit['scale'], select['scale'])

    import json
    manifest = json.loads((tmp_path / 'out' / 'manifest.json').read_text())
    assert manifest['protocol'] == 'season_cv_refit_v1'
    assert manifest['selected_epochs'] == [3] and manifest['refit_epochs'] == [3]


def test_b1_refit_loss_scales_use_restored_hidden_weeks():
    """Hidden weeks are excluded from inner scales but included in the refit's.

    Uses a calendar whose hidden weeks carry a distinctly larger level, so a Q95
    computed with them differs from one computed without.
    """
    from datetime import date, timedelta
    import numpy as np
    from tapestry.models.b1_run import unique_truth, loss_scales
    from tapestry.models.b1_seasons import fold, hidden_weeks
    from tapestry.models.season_cv import SEASONS

    held = SEASONS[-1]
    hidden = hidden_weeks(_synthetic_fold_dataset(), held)
    archive = wd.VintageArchive()
    day = date(2023, 8, 5)
    while day <= date(2026, 9, 26):
        spike = day.isoformat() in hidden
        for c in range(6):
            archive.add('delphi_nhsn' if c < 3 else 'delphi_nssp', '2026-09-30', day.isoformat(),
                        c, 'NC', (1000 if spike else 100) if c < 3 else (.4 if spike else .02))
        day += timedelta(weeks=1)
    ds = wd.build_wednesday(start='2023-08-09', end='2026-09-16', truth_cutoff='2026-09-30',
                            locations=('NC',), archive=archive)
    fitting, _, refit, _, _ = fold(ds, held)
    inner_scale = loss_scales(unique_truth(fitting))
    refit_scale = loss_scales(unique_truth(refit))
    assert not np.array_equal(inner_scale, refit_scale), 'refit scales must see restored hidden weeks'
    assert (np.asarray(refit_scale) > np.asarray(inner_scale)).any()


@pytest.mark.skipif(not __import__('pathlib').Path('data/processed/build_b_finalized.npz').exists()
                    or not __import__('pathlib').Path('data/processed/build_b1_wednesday_calendar.npz').exists(),
                    reason='requires the local B0 and B1 datasets')
def test_b1_direct_fold_origins_match_b0_exactly():
    """The direct task's four partitions must select B0's own origins, fold by fold."""
    from tapestry.model_data.finalized import FinalizedDataset
    from tapestry.model_data.wednesday import WednesdayDataset
    from tapestry.models.b1_seasons import fold
    from tapestry.models.season_cv import SEASONS, fold_data, validation_split

    b1ds = WednesdayDataset.load('data/processed/build_b1_wednesday_calendar.npz')
    b0ds = FinalizedDataset.load('data/processed/build_b_finalized.npz')
    ends = lambda eps: {str(e['context_dates'][-1]) for e in eps}
    for held in SEASONS:
        fitting, validation, refit, evaluation, _ = fold(b1ds, held)
        b0_train, b0_eval, _ = fold_data(b0ds, held, 12)
        b0_inner, b0_val, _, _ = validation_split(b0ds, held, 12, 4)
        assert ends(refit) == ends(b0_train), f'{held}: refit origins differ from B0 training'
        assert ends(fitting) == ends(b0_inner), f'{held}: inner origins differ from B0'
        assert ends(validation) == ends(b0_val), f'{held}: validation origins differ from B0'
        assert ends(evaluation) == ends(b0_eval), f'{held}: evaluation origins differ from B0'
