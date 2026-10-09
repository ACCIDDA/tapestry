"""Scientific guards: flu ED receives loss, excluded pathogen histories cannot leak."""
import numpy as np
import torch
from tapestry.model.network import Model, IndependentBundle
from tapestry.model.series import SeriesModel
from tapestry.model.objective import LOSS_WEIGHTS
from tapestry.model.revision_tree import TrajectoryNowcaster


def test_ed_correction_preserves_proportions_and_other_channels():
    values=np.ones((8,6,4),np.float32);values[:,3:]=.02
    e=dict(values=values,available=np.ones_like(values,dtype=bool),
           known_final=np.zeros_like(values,dtype=bool),
           context_dates=tuple(f'2025-01-{day:02d}' for day in range(1,9)))
    truth=dict(e,values=values.copy());truth['values'][:,3]=.04
    model=TrajectoryNowcaster(2,10,1,'phase');model.channels=[3];model.pathogen_inputs='flu'
    model.fit([(e,truth)]*30)
    changed=model.apply(e)
    assert np.array_equal(changed['values'][:,:3],values[:,:3])
    assert np.array_equal(changed['values'][:-2,3],values[:-2,3])
    assert np.all(changed['values'][-2:,3]>.02)
    assert np.all(changed['values'][:,3:]<=1)


def test_two_season_extension_is_nested():
    from tapestry.dataset.cv import training_seasons
    from tapestry.model.scenario import Scenario
    assert training_seasons(Scenario(training_window='last2'),'2025-2026')==('2023-2024','2024-2025')
    assert training_seasons(Scenario(training_window='last2'),'2024-2025')==('2023-2024','2025-2026')


def test_operational_episode_keeps_unknown_future_labels_masked():
    from tapestry.dataset.episodes import episodes
    panel = dict(dates=np.array(['2026-09-19', '2026-09-26', '2026-10-03']),
                 locations=np.array(['US']), issuance_dates=np.array(['2026-10-07']),
                 targets=np.ones((3, 1, 6), np.float32),
                 asof_targets=np.ones((1, 3, 1, 6), np.float32))
    assert episodes(panel, 2, 'reported') == []
    episode, = episodes(panel, 2, 'reported', require_labels=False)
    assert episode['context_dates'][-1] == '2026-10-03'
    assert episode['target_dates'] == ('2026-10-10', '2026-10-17', '2026-10-24', '2026-10-31')
    assert not episode['target_available'].any()
    assert not episode['Y'][:, :, 1].any()


def test_flu_scope_excludes_both_values_and_presence():
    torch.manual_seed(42)
    x=torch.rand(2,8,6,3);mask=torch.ones_like(x,dtype=torch.bool);cal=torch.zeros(2,3)
    other=x.clone();other[:,:, [1,2,4,5]]=10000
    absent=mask.clone();absent[:,:,[1,2,4,5]]=False
    for model in [Model(lookback=8,width=16,decoder='quantile',pathogen_inputs='flu'),SeriesModel(lookback=8,width=16,pathogen_inputs='flu')]:
        a=model(values=x,available=mask,calendar=cal)
        b=model(values=other,available=absent,calendar=cal)
        torch.testing.assert_close(a[:,:,:, [0,3]],b[:,:,:, [0,3]],rtol=0,atol=0)


def test_flu_loss_and_bundle_preserve_ed():
    assert LOSS_WEIGHTS['flu_hosp_ed']==[1,0,0,.5,0,0]
    model=IndependentBundle([SeriesModel(lookback=8,width=16),SeriesModel(lookback=8,width=16)],[[0],[3]])
    y=model(values=torch.rand(1,8,6,2),available=torch.ones(1,8,6,2,dtype=torch.bool),calendar=torch.zeros(1,3))
    assert y[:,:,:,3].sum()>0
    assert y[:,:,:, [1,2,4,5]].sum()==0


def test_nowcaster_scope_excludes_other_pathogen_histories():
    m=TrajectoryNowcaster(4,10.,1.,'phase');m.pathogen_inputs='flu'
    e=dict(values=np.ones((8,6,3)),available=np.ones((8,6,3),bool),context_dates=np.array(['2024-10-05']*8))
    a=m.design(e,0)[0]
    e['values'][:,[1,2,4,5]]=10000;e['available'][:,[1,2,4,5]]=False
    np.testing.assert_array_equal(a,m.design(e,0)[0])


def test_specialist_excludes_other_signal_values_and_masks():
    for scope, channel in [('flu_hosp', 0), ('flu_ed', 3)]:
        x=torch.rand(1,8,6,3);mask=torch.ones_like(x,dtype=torch.bool)
        excluded=[c for c in range(6) if c!=channel]
        other=x.clone();other[:,:,excluded]=10000
        absent=mask.clone();absent[:,:,excluded]=False
        m=Model(lookback=8,width=16,decoder='quantile',pathogen_inputs=scope)
        torch.testing.assert_close(m(values=x,available=mask,calendar=torch.zeros(1,3)),
                                   m(values=other,available=absent,calendar=torch.zeros(1,3)),rtol=0,atol=0)
        tree=TrajectoryNowcaster(2,10.,1.,'phase');tree.pathogen_inputs=scope
        e=dict(values=x[0].numpy(),available=mask[0].numpy(),context_dates=np.array(['2024-10-05']*8))
        changed=dict(e,values=other[0].numpy(),available=absent[0].numpy())
        np.testing.assert_array_equal(tree.design(e,channel)[0],tree.design(changed,channel)[0])
    assert LOSS_WEIGHTS['flu_only']==[1,0,0,0,0,0]
    assert LOSS_WEIGHTS['flu_ed']==[0,0,0,1,0,0]


def test_flu_covariate_aliases_do_not_add_other_pathogens():
    from tapestry.dataset.build import covariate_names_for
    assert covariate_names_for('ww_flu')==('nwss_flu_wval_like',)
    assert covariate_names_for('outpatient_flu')==('outpatient_flu',)
    assert covariate_names_for('kinsa+ww_flu')==('nwss_flu_wval_like','kinsa_ili')
