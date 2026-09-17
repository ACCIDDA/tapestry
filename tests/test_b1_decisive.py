"""Scientific boundaries of the flagged direct and parallel auxiliary candidates."""
import numpy as np
import torch

from tapestry.models.b1 import B1
from tapestry.models.b1_scenarios import b0_top4, B1Scenario


def test_parallel_forecast_never_depends_on_recent_head_or_hidden_final_flags():
    torch.manual_seed(42)
    model = B1(list(range(6)), {'NC': 1e6}, ['NC'], width=8,
               supplied_final=True, parallel_recent=True, count_transform='fourth_root', ed_transform='logit')
    x = torch.full((2, 12, 6, 1), .2)
    x[:, :, :3] = 100
    available = torch.ones_like(x, dtype=torch.bool)
    final = available.clone()
    dropout = torch.zeros_like(available)
    dropout[:, -2] = True
    z = torch.randn(4, 2, 16)
    cal = torch.zeros(2, 3)
    result = model(x, available, cal, dropout=dropout, known_final=final, z_future=z)
    torch.testing.assert_close(result[:, :, 1], x[:, -1][None].expand(4, -1, -1, -1), rtol=0, atol=0)
    changed = x.clone(); changed[:, -2] = float('nan')
    final[:, -2] = False
    torch.testing.assert_close(result, model(changed, available, cal, dropout=dropout, known_final=final, z_future=z))
    result[:, :, 2:].sum().backward()
    assert all((p.grad is None or not p.grad.any()) for p in model.direct_model.recent_heads.parameters())
    assert model.direct_model.context[0].weight.grad.abs().sum() > 0
    with torch.no_grad():
        for p in model.direct_model.recent_heads.parameters():
            p.add_(3)
    after = model(x, available, cal, dropout=dropout, known_final=final, z_future=z)
    torch.testing.assert_close(result[:, :, 2:], after[:, :, 2:], rtol=0, atol=0)
    assert not torch.allclose(result[:, :, 0], after[:, :, 0])


def test_candidate_a_identity_and_seeded_predictions_unchanged():
    # Existing identity is the key to preserving completed attempts.
    a = b0_top4('direct', .5)[0]
    assert a.run_id == 'mlp-target-direct-mask0.5-a18eb85fd2dd'
    assert B1Scenario.from_string(a.scenario_string) == a
    for pipeline in ('direct_finalflag', 'joint_aux025'):
        b = b0_top4(pipeline, .5)[0]
        assert B1Scenario.from_string(b.scenario_string) == b
        assert b.run_id != a.run_id


def test_auxiliary_weight_and_forecast_only_selection_preserve_absent_task():
    from tapestry.model_data.wednesday import output_dates
    from tapestry.models.b1_run import objective_weights
    context, dates = output_dates('2025-01-08')
    episode = dict(X=np.ones((12, 6, 3, 2), np.float32),
                   Y=np.ones((6, 6, 2, 2), np.float32),
                   context_dates=context, target_dates=dates, locations=('NC', 'US'))
    scenario = b0_top4('joint_aux025', .5)[0]
    visible = objective_weights([episode], [0], scenario)
    assert not visible[:, :2].any()
    np.testing.assert_allclose(visible[:, 2:].sum(), 1.)
    d = np.zeros((1, 12, 6, 2), bool); d[:, -2:] = True
    train = objective_weights([episode], [0], scenario, d)
    validation = objective_weights([episode], [0], scenario, d, validation=True)
    np.testing.assert_allclose(train[:, :2].sum(), .25)
    np.testing.assert_allclose(train[:, 2:].sum(), 1.)
    np.testing.assert_array_equal(validation[:, 2:], train[:, 2:])
    assert not validation[:, :2].any()
    np.testing.assert_allclose(train.sum((0, 1, 2)), [1., .25], rtol=1e-6)


def test_temporal_blocks_preserve_scientific_ratios_and_pairing():
    import pandas as pd
    from tapestry.evaluation.decisive import temporal_intervals
    from tapestry.evaluation.totals import METRICS
    rows = []
    for i, day in enumerate(pd.date_range('2025-01-04', periods=16, freq='7D')):
        for loc, scale in [('US', 1000), ('37', 10)]:
            for h in range(4):
                rows.append(dict(season='2024-2025', target='wk inc flu hosp',
                    reference_date=day.date().isoformat(), target_end_date=(day + pd.Timedelta(weeks=h)).date().isoformat(),
                    horizon=h, location=loc, geography='US' if loc == 'US' else 'states_dc',
                    **{f'{who}_{metric}': scale * (i+1) for who in ('model', 'ensemble') for metric in METRICS}))
    a = pd.DataFrame(rows)
    b = a.copy(); b['model_wis'] *= .9
    intervals = temporal_intervals({'A': a, 'B': b}, 8, repetitions=100)
    np.testing.assert_allclose([intervals[0]['relative_low'], intervals[0]['relative_high']], [-.1, -.1])
