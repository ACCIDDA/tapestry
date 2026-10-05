"""Causal bootstrap of intact eight-week revision-error histories."""
import warnings
import numpy as np


def empirical_crps(samples, truth):
    """Exact CRPS of the supplied empirical distribution, member axis first."""
    m=len(samples)
    rank=(2*np.arange(1,m+1)-m-1).reshape((m,)+(1,)*(samples.ndim-1))
    return np.mean(abs(samples-truth),axis=0)-(np.sort(samples,axis=0)*rank).sum(0)/m**2


class RevisionUncertainty:
    """Past prequential errors, supervised by actual archived delay-12 reports.

    One donor supplies all weeks, locations and targets. Missing donor cells have
    zero perturbation; cells with fewer than twelve donors remain deterministic.
    The empirical median log error is removed to preserve the point estimator.
    """
    def __init__(self,panel,predictions):
        self.panel=panel
        self.predictions=predictions
        self.issues=panel['issuance_dates'].astype('datetime64[D]')
        dates=panel['dates'].astype('datetime64[D]')
        self.event=np.searchsorted(dates,self.issues-np.timedelta64(4,'D'))[:,None]-np.arange(8)
        due=dates[self.event.clip(0)]+np.timedelta64(88,'D') # 4 + 12*7
        self.maturity=np.searchsorted(self.issues,due)
        valid=(self.event>=0)&(self.maturity<len(self.issues))
        valid &= self.issues[self.maturity.clip(0,len(self.issues)-1)]==due
        self.truth=panel['asof_targets'][self.maturity.clip(0,len(self.issues)-1),self.event.clip(0)].copy()
        self.observed=np.isfinite(panel['asof_targets'][np.arange(len(self.issues))[:,None],self.event.clip(0)])
        self.support=valid[:,:,None,None]&self.observed&np.isfinite(self.truth)&np.isfinite(predictions)
        self.offset=np.array([1. if str(n).startswith('nhsn_') else .0001 for n in panel['target_names']])
        self.errors=np.where(self.support,np.clip(np.log((self.truth+self.offset)/(predictions+self.offset)),
                                                  -np.log(4),np.log(4)),np.nan)

    def draw(self,issue,center,members=256,strength=1.,seed=42):
        """Draw native-unit chronological histories; never perturb proxy-filled cells."""
        current=int(np.searchsorted(self.issues,np.datetime64(issue)))
        donor=np.flatnonzero((self.maturity.max(1)<=current)&
            (np.arange(len(self.issues))<current)&(np.arange(len(self.issues))>=current-104)&
            self.support.any(axis=(1,2,3)))
        histories=np.broadcast_to(center,(members,*center.shape)).copy()
        floor=np.broadcast_to(self.offset,(center.shape[2],center.shape[1])).T
        # The fallback must be independent of the candidate prediction so that
        # normalized proper scores remain comparable between point centers.
        ti=self.event[current,0]-np.arange(center.shape[0]-1,-1,-1)
        raw=self.panel['asof_targets'][current,ti.clip(0)].copy()
        raw[ti<0]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            raw_scale=np.nanquantile(raw,.95,axis=0).T
        scale=np.maximum(np.nan_to_num(raw_scale),floor)
        metadata=dict(donors=len(donor),supported_fraction=0.,latest_mature_issuance=None)
        if len(donor)<12:return histories,scale,metadata
        error=self.errors[donor]
        supported=np.isfinite(error).sum(0)>=12
        # One common donor weight preserves joint patterns; recency is known at issuance.
        weight=2.**(-(current-donor)/26.);weight/=weight.sum()
        order=np.argsort(np.where(np.isfinite(error),error,np.inf),axis=0)
        sorted_error=np.take_along_axis(error,order,axis=0)
        cell_weight=np.where(np.isfinite(error),weight[:,None,None,None],0.)
        cumulative=np.cumsum(np.take_along_axis(cell_weight,order,axis=0),axis=0)
        median_index=(cumulative<.5*cumulative[-1]).sum(0).clip(0,len(donor)-1)
        median=np.take_along_axis(sorted_error,median_index[None],axis=0)[0]
        rng=np.random.default_rng(seed+current)
        chosen=rng.choice(len(donor),size=members,p=weight)
        noise=np.where(supported,np.nan_to_num(error[chosen]-median),0.)
        noise=np.moveaxis(noise[:,::-1],-1,-2) # member, chronological week, target, location
        visible=np.moveaxis(self.observed[current,::-1],-1,-2)
        values=center[-8:]
        perturbed=np.maximum(0,(values+self.offset[:,None])*np.exp(strength*noise)-self.offset[:,None])
        for k,name in enumerate(self.panel['target_names']):
            perturbed[:,:,k]=np.rint(perturbed[:,:,k]) if str(name).startswith('nhsn_') else np.clip(perturbed[:,:,k],0,1)
        histories[:,-8:]=np.where(visible,perturbed,values)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            q=np.nanquantile(np.where(self.support[donor],self.truth[donor],np.nan),.95,axis=(0,1)).T
        scale=np.where(np.isfinite(q),np.maximum(np.nan_to_num(q),floor),scale)
        metadata.update(supported_fraction=float(supported.mean()),
            latest_mature_issuance=str(self.issues[self.maturity[donor].max()]))
        return histories.astype(np.float32),scale,metadata


def trajectory_distribution_scores(draws,truth,center,scale):
    """Proper scores and 80% coverage of point, four-week level and signed change."""
    def functionals(a):
        return (a[...,-1,:,:],a[...,-4:,:,:].mean(-3),
                a[...,-2:,:,:].mean(-3)-a[...,-4:-2,:,:].mean(-3))
    result={}
    for name,x,y,baseline in zip(('point','level','growth'),functionals(draws),functionals(truth),functionals(center)):
        result[name+'_crps']=empirical_crps(x,y)/scale
        result[name+'_point_error']=abs(baseline-y)/scale
        lo,hi=np.quantile(x,[.1,.9],axis=0)
        result[name+'_coverage80']=np.where(np.isfinite(y),(y>=lo)&(y<=hi),np.nan)
    result['trajectory_crps']=sum(result[n+'_crps'] for n in ('point','level','growth'))/3
    result['trajectory_point_error']=sum(result[n+'_point_error'] for n in ('point','level','growth'))/3
    return result
