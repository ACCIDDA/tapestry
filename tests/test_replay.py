"""Input alignment and label leakage checks for fixed-checkpoint replay."""
import numpy as np
from tapestry.experiment.replay import replay_episode


def example():
    dates=np.arange(np.datetime64('2025-01-04'),np.datetime64('2025-03-23'),7).astype(str)
    t=len(dates)
    panel=dict(dates=dates,issuance_dates=np.array(['2025-03-26']),locations=np.array(['AA','US']),
        asof_targets=np.full((1,t,2,1),3.,np.float32),
        asof_covariates=np.full((1,t,2,1),5.,np.float32),
        asof_covariates_national=np.full((1,t,1),7.,np.float32),
        covariate_names=np.array(['state_source']),covariate_national_names=np.array(['national_source']))
    panel['asof_targets'][0,-1,0,0]=np.nan
    episode=dict(context_dates=tuple(dates),values=np.full((t,1,2),999.,np.float32),
        available=np.ones((t,1,2),bool),known_final=np.ones((t,1,2),bool),
        target_values=np.full((4,1,2),888.),target_available=np.ones((4,1,2),bool),
        covariates=np.full((t,2,2,2),999.))
    return panel,episode


def test_vintage_replay_never_fills_from_final_and_preserves_labels():
    panel,episode=example()
    out=replay_episode(episode,panel,('state_source','national_source'),'vintage')
    assert out['values'][0,0,0]==3
    assert out['values'][-1,0,0]==0 and not out['available'][-1,0,0]
    assert not out['known_final'].any()
    assert np.all(out['covariates'][:,0,0]==5)
    assert np.all(out['covariates'][:,1,0]==7)
    assert out['target_values'] is episode['target_values']
    assert out['target_available'] is episode['target_available']
    assert np.all(episode['values']==999)


def test_nowcast_replay_only_replaces_eight_aligned_weeks():
    panel,episode=example()
    correction=np.broadcast_to(np.arange(8,dtype=np.float32)[None,:,None,None]+20,(1,8,2,1)).copy()
    correction[0,0,0,0]=np.nan
    out=replay_episode(episode,panel,('state_source','national_source'),'nowcast',correction)
    np.testing.assert_array_equal(out['values'][-8:,0,1],np.arange(27,19,-1))
    assert np.all(out['values'][:-8]==3)
    assert not out['available'][-1,0,0] and out['values'][-1,0,0]==0
    assert not out['known_final'].any()
    assert out['target_values'] is episode['target_values']
    assert replay_episode(episode,panel,(),'finalized') is episode


def test_flag_controls_do_not_change_values_masks_or_labels():
    panel,episode=example()
    off=replay_episode(episode,panel,(),'finalized',flags='off')
    assert not off['known_final'].any()
    for key in ('values','available','target_values','target_available'):
        assert off[key] is episode[key]
    original=replay_episode(episode,panel,(),'vintage')
    compatible=replay_episode(episode,panel,(),'vintage',flags='available')
    for key in ('values','available','target_values','target_available'):
        np.testing.assert_array_equal(original[key],compatible[key])
    np.testing.assert_array_equal(compatible['known_final'],compatible['available'])
    assert not original['known_final'].any()


def test_scheduled_proxies_preserve_observed_reports_and_native_gaps():
    panel,episode=example()
    panel['targets']=np.full((len(panel['dates']),2,1),11.)
    raw=replay_episode(episode,panel,('state_source','national_source'),'vintage',flags='available',scheduled=True)
    assert raw['values'][-1,0,0]==11 and raw['values'][-1,0,1]==3
    assert raw['covariates'] is episode['covariates']
    correction=np.full((1,8,2,1),4.)
    corrected=replay_episode(episode,panel,(),'nowcast',correction,flags='available',scheduled=True)
    assert corrected['values'][-1,0,0]==11 and corrected['values'][-1,0,1]==4
    panel['targets'][-1,0,0]=np.nan
    missing=replay_episode(episode,panel,(),'nowcast',correction,scheduled=True)
    assert not missing['available'][-1,0,0]
    assert missing['target_values'] is episode['target_values']
