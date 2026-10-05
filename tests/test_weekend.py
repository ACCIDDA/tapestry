"""Scientific invariants of the weekend revision comparison."""
from dataclasses import replace
import numpy as np
from tapestry.model.scenario import Scenario
from tapestry.experiment.training import objective_weights
from tapestry.model.revision_regression import RevisionRegression


def episode():
    values=np.arange(12*6*2,dtype=np.float32).reshape(12,6,2)+1
    values[:,3:]/=1000
    available=np.ones_like(values,dtype=bool)
    dates=tuple(str(d) for d in np.arange(np.datetime64('2025-10-04'),np.datetime64('2025-12-21'),np.timedelta64(7,'D')))
    y=np.ones((8,6,2),np.float32)
    mask=np.ones_like(y,dtype=bool)
    return dict(values=values,available=available,known_final=available.copy(),
        context_dates=dates,locations=('NC','US'),target_values=y,target_available=mask,
        target_dates=('2025-11-29','2025-12-06','2025-12-13','2025-12-20','2025-12-27','2026-01-03','2026-01-10','2026-01-17'),
        Y=np.stack((y,mask),axis=2))


def test_joint_weights_keep_forecast_objective():
    s=Scenario(weekend_family='joint',input_mode='scheduled_final',reporting_missingness=False,nowcast_weeks=4,joint_weight=.25)
    w=objective_weights([episode()],s,list(range(6)))
    np.testing.assert_allclose(w[:,:4].sum(),.25,rtol=1e-6)
    np.testing.assert_allclose(w[:,4:].sum(),1.,rtol=1e-6)


def test_nowcast_features_do_not_read_final_labels():
    e=episode();model=RevisionRegression()
    x,floor=model.design(e,0)
    changed=dict(e,target_values=e['target_values']*100,Y=e['Y']*100)
    x2,floor2=model.design(changed,0)
    np.testing.assert_array_equal(x,x2)
    np.testing.assert_array_equal(floor,floor2)


def test_zero_correction_preserves_inputs_and_labels():
    e=episode();model=RevisionRegression(strength=0.)
    model.coefficients=[np.ones(model.design(e,k)[0].shape[-1]) for k in range(6)]
    got=model.apply(e)
    np.testing.assert_allclose(got['values'],e['values'],rtol=1e-6)
    np.testing.assert_array_equal(got['target_values'],e['target_values'])
    np.testing.assert_array_equal(got['available'],e['available'])


def test_joint_weight_changes_actual_optimizer_loss():
    # One episode / one optimizer step: identical initialization and draws mean
    # the initial loss must change when the auxiliary objective weight changes.
    # This catches defining a correct helper but never using it in the fitter.
    from tapestry.experiment.training import fit_component, model_options
    e=episode()
    base=Scenario(weekend_family='joint',input_mode='scheduled_final',reporting_missingness=False,
                  nowcast_weeks=4,joint_weight=.25,epochs=1,batch_size=1,members=4,
                  width=8,latent=2,geography=False)
    losses=[]
    for weight in (.25,1.):
        s=replace(base,joint_weight=weight)
        options=model_options([e],s,{'NC':10_000_000.,'US':300_000_000.})
        _,record=fit_component([e],None,list(range(6)),0,options,s,42,'cpu')
        losses.append(record['history'][0]['loss'])
    assert losses[1] > losses[0] + 1e-6
