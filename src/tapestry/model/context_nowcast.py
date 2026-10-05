"""Online, trajectory-aware residual calibration of causal seasonal nowcasts."""
import numpy as np
from .finalization import reporting_context



def revision_momentum(asof, wi, ti, li, age, integer=False):
    """Changes known at issuance, for prior event weeks and the event itself.

    Each row compares the same event at w and w-1. Prior weeks are relative to
    the latest event at the row's issuance, so old ages share reporting context.
    """
    events=np.column_stack((ti+age-1,ti+age-2,ti+age-3,ti))
    a=asof[wi[:,None],events.clip(0),li[:,None]].astype(float)
    b=asof[(wi-1).clip(0)[:,None],events.clip(0),li[:,None]].astype(float)
    offset=float(integer)
    valid=(wi[:,None]>0)&(events>=0)&np.isfinite(a)&np.isfinite(b)&(a+offset>0)&(b+offset>0)
    ratio=np.divide(a+offset,b+offset,out=np.ones_like(a),where=valid)
    changes=np.where(valid,np.clip(np.log(ratio),-1,1),0.)
    return np.column_stack((changes,~valid))


def causal_gate(panel, asof, rows, baseline, candidate, integer=False):
    """Choose residual strength using only earlier prequential predictions.

    Require eight distinct mature issuance paths in the previous 38 weeks
    (approximately 26 weeks plus the twelve-week maturation lag). Each target's
    states share a choice with equal location weights; US chooses independently.
    Historical inputs must have a complete twelve-week visible history.
    """
    issues=panel['issuance_dates'].astype('datetime64[D]')
    dates=panel['dates'].astype('datetime64[D]')
    wi=np.searchsorted(issues,rows['issuance'].astype('datetime64[D]'))
    ti=np.searchsorted(dates,rows['boundary'].astype('datetime64[D]'))
    li=rows['location'];age=rows['age'];nl=asof.shape[2]
    due=dates[ti]+np.timedelta64(4+7*(rows['lag']+12),'D')
    mw=np.searchsorted(issues,due)
    truth=asof[mw.clip(0,len(issues)-1),ti,li].astype(float)
    truth[(mw>=len(issues))|(issues[mw.clip(0,len(issues)-1)]!=due)]=np.nan
    lookup=np.full((len(issues),nl,4),-1,int)
    use=age<4
    lookup[wi[use],li[use],age[use]]=np.flatnonzero(use)
    paths=lookup.reshape(-1,4)
    paths=paths[(paths>=0).all(1)]
    valid=np.isfinite(truth[paths]).all(1)&np.isfinite(rows['baseline_history'][paths[:,0]]).all(1)
    paths=paths[valid]
    out=baseline.copy();alphas=np.zeros(len(candidate))
    def resolution(values):
        return np.rint(np.maximum(values,0)) if integer else np.rint(np.clip(values,0,1)*10000)/10000
    for national in (False,True):
        group=(rows['locations'][li]=='US')==national
        gp=paths[group[paths[:,0]]]
        for current in np.unique(wi[group]):
            apply=group&(wi==current)
            past=gp[(mw[gp]<=current).all(1)&(wi[gp[:,0]]<current)&(wi[gp[:,0]]>=current-38)]
            if len(np.unique(wi[past[:,0]]))<8:continue
            location=li[past[:,0]]
            scales=np.ones(len(past))
            for loc in np.unique(location):
                mask=location==loc
                scales[mask]=max(float(np.quantile(truth[past[mask]],.95)),1. if integer else 1e-8)
            def utility(pred):
                delta=resolution(pred)-truth[past]
                errors=(abs(delta[:,0])+abs(delta.mean(1))+abs(delta[:,:2].mean(1)-delta[:,2:].mean(1)))/(3*scales)
                return np.mean([errors[location==loc].mean() for loc in np.unique(location)])
            grid=(0.,.25,.5,1.)
            losses=[utility(baseline[past]+alpha*(candidate[past]-baseline[past])) for alpha in grid]
            best=int(np.argmin(losses))
            # Require 1% historical utility improvement, with identity winning ties.
            alpha=grid[best] if losses[best]<.99*losses[0] else 0.
            out[apply]=baseline[apply]+alpha*(candidate[apply]-baseline[apply])
            alphas[apply]=alpha
    return out,alphas

def residual_predictions(panel, asof, rows, seasonal, integer=False, penalty=10., growth_weight=1., residual_halflife=26., features_mode="basic"):
    """Fit to past causal predictions, supervised only by actually due delay-12 reports.

    A small pooled multiplicative correction has age, growth, holiday and local
    effects. Three IRLS steps approximate absolute point/level/growth loss with
    ridge shrinkage to the seasonal control. States share slopes; US fits alone.
    Refit every four scheduled issuances; all labels are mature at that refit.
    """
    dates=panel['dates'].astype('datetime64[D]')
    issues=panel['issuance_dates'].astype('datetime64[D]')
    wi=np.searchsorted(issues,rows['issuance'].astype('datetime64[D]'))
    ti=np.searchsorted(dates,rows['boundary'].astype('datetime64[D]'))
    li=rows['location'];age=rows['age'];n_age=int(age.max())+1
    due=dates[ti]+np.timedelta64(4+7*(rows['lag']+12),'D')
    mw=np.searchsorted(issues,due)
    mature=asof[mw.clip(0,len(issues)-1),ti,li].astype(float)
    mature[(mw>=len(issues)) | (issues[mw.clip(0,len(issues)-1)]!=due)]=np.nan
    growth,flags=reporting_context(asof,issues,dates,integer)
    context=growth[wi,ti,li]
    # Unknown growth gets an explicit indicator rather than asserting flat growth.
    calendar=flags(issues[wi]).astype(float)
    common=np.column_stack((np.eye(n_age)[age],np.nan_to_num(context),
                            ~np.isfinite(context),calendar))
    if features_mode in ('momentum', 'age_momentum'):
        common=np.column_stack((common,revision_momentum(asof,wi,ti,li,age,integer)))
    prediction=seasonal.copy()
    visible=np.isfinite(rows['baseline_history'][:,-1])
    for national in (False,True):
        group=(rows['locations'][li]=='US')==national
        locations=np.unique(li[group])
        if not len(locations):continue
        # Redundant local effects are regularized; no unpenalized intercept.
        local=(li[:,None]==locations[None]).astype(float) if not national else np.empty((len(li),0))
        if features_mode in ('age', 'age_momentum'):
            # Reporting effects can change the trajectory: their correction must
            # vary with report age, not shift all eight weeks by the same factor.
            # Two smooth bases permit a recent level and slope effect, while
            # shrinking context/local effects toward zero as reports mature.
            contextual=np.column_stack((common[:,n_age:],local))
            decay=2.**(-age/2.)
            features=np.column_stack((common[:,:n_age],contextual*decay[:,None],
                                      contextual*(age*decay)[:,None]))
        else:
            features=np.column_stack((common,local))
        beta=None;last_refit=-999;scale=None
        for current in np.unique(wi[group]):
            apply=group & (wi==current) & visible
            if not apply.any():continue
            if current-last_refit>=4:
                train=group & visible & np.isfinite(mature) & (mw<=current) & (wi<current) & (wi>=current-104)
                # Only complete four-week historical paths train the primary utility.
                index=np.flatnonzero(train)
                if len(np.unique(wi[index]))<12:continue
                scale=np.ones(asof.shape[2])
                for l in locations:
                    vals=mature[train & (li==l)]
                    scale[l]=max(float(np.quantile(vals,.95)),1. if integer else 1e-8) if len(vals) else 1.
                x=features[index]*(seasonal[index]/scale[li[index]])[:,None]
                y=(mature[index]-seasonal[index])/scale[li[index]]
                weight=2.**(-(current-wi[index])/residual_halflife)
                # Identical total weight per location despite missing observations.
                totals=np.bincount(li[index],weights=weight,minlength=len(scale))
                weight/=np.maximum(totals[li[index]],1e-12)
                weight/=weight.mean()
                lookup=np.full((len(issues),len(scale),n_age),-1,int)
                lookup[wi[index],li[index],age[index]]=np.arange(len(index))
                paths=lookup[:,:,:4].reshape(-1,4)
                paths=paths[(paths>=0).all(1)]
                if not len(paths):continue
                pw=weight[paths].mean(1)
                xx=[x]
                yy=[y];ww=[weight*.125] # supplementary point accuracy at all ages
                xx.extend([x[paths[:,0]],x[paths].mean(1),x[paths[:,:2]].mean(1)-x[paths[:,2:]].mean(1)])
                yy.extend([y[paths[:,0]],y[paths].mean(1),y[paths[:,:2]].mean(1)-y[paths[:,2:]].mean(1)])
                ww.extend([pw,pw,pw*growth_weight])
                design=np.concatenate(xx);target=np.concatenate(yy);weights=np.concatenate(ww)*4/(3+growth_weight)
                beta=np.zeros(features.shape[1])
                # Fixed 0.01 normalized residual floor avoids singular near-exact fits.
                for _ in range(3):
                    robust=weights/np.maximum(abs(target-design@beta),.01)
                    lhs=design.T@(robust[:,None]*design)+penalty*np.eye(design.shape[1])
                    beta=np.linalg.solve(lhs,design.T@(robust*target))
                last_refit=current
            if beta is not None:
                # Explicit conservative bound on this additional calibration.
                adjustment=np.clip(features[apply]@beta,-.5,.5)
                prediction[apply]=seasonal[apply]*(1+adjustment)
    return prediction


def context_predictions(panel, asof, rows, anchor, integer=False, penalty=10.,
                        halflife=8., prior_weeks=4., statistic='median', common_factors=None,
                        growth_weight=1., residual_halflife=26., features_mode="basic", gate="none", diagnostics=None, strength=1.):
    """Construct training histories independently of reference-label support/CV masks."""
    from tapestry.dataset.finalization import boundary_rows
    from .finalization import seasonal_predictions
    # boundary_rows uses the name only to select its scheduled lag.
    name='ilinet_ili' if rows['lag'] else 'nhsn_flu_admissions'
    history=boundary_rows(panel,name,np.zeros(asof.shape[1:]),asof,rows['locations'],12,int(rows['age'].max())+1)
    keep=history['issuance']<=max(rows['issuance'])
    history={k:v[keep] if isinstance(v,np.ndarray) and k!='locations' else v for k,v in history.items()}
    shared=np.ones(len(history['age']))
    base,support,factors=seasonal_predictions(panel,asof,history,np.zeros(len(history['age'])),
        mode='adaptive_chain',integer=integer,halflife=halflife,prior_weeks=prior_weeks,statistic=statistic,common_factors=shared)
    predicted=(base.copy() if gate=='admissions' and not integer else
        residual_predictions(panel,asof,history,base,integer,penalty,growth_weight,residual_halflife,features_mode))
    alpha=np.ones(len(base)) if gate!='admissions' or integer else np.zeros(len(base))
    if gate=='causal':
        predicted,alpha=causal_gate(panel,asof,history,base,predicted,integer)
    if strength!=1.:
        predicted=base+strength*(predicted-base)
    alpha*=strength
    def keys(r):
        w=np.searchsorted(panel['issuance_dates'].astype(str),r['issuance'])
        return (w*(int(history['age'].max())+1)+r['age'])*len(rows['locations'])+r['location']
    order=np.argsort(keys(history));index=order[np.searchsorted(keys(history)[order],keys(rows))]
    if common_factors is not None:
        common_factors[:]=shared[index]
    if diagnostics is not None:
        diagnostics['alpha']=alpha[index]
    result=np.where(np.isfinite(rows['baseline_history'][:,-1]),predicted[index],anchor)
    return result,support[index],factors[index]
