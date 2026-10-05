"""Causal, partially pooled chain-ladder completion of weekly cumulative reporting triangles.

Inspired by epinowcast/baselinenowcast's development-factor estimator. This
implementation handles gaps pairwise, preserves signed revisions, and uses
identity pseudo-exposure to stabilize sparse local triangles.
"""
import numpy as np


def reporting_context(asof, issues, dates, integer=False):
    """Four-week growth at each issuance/event, using only that vintage.

    Two two-week means reduce count noise. Zero-rate pairs have no log ratio;
    missing context does not imply flat growth. Calendar flags describe the
    reporting week and its following week, including holiday recovery.
    """
    from datetime import date, timedelta
    recent = (asof + np.roll(asof, 1, axis=1)) / 2
    older = (np.roll(asof, 2, axis=1) + np.roll(asof, 3, axis=1)) / 2
    offset = float(integer)
    good = np.isfinite(recent) & np.isfinite(older) & (recent+offset > 0) & (older+offset > 0)
    ratio = np.divide(recent+offset, older+offset, out=np.ones_like(recent), where=good)
    growth = np.where(good, np.clip(np.log(ratio)/2, -1, 1), np.nan)
    growth[:, :3] = np.nan
    holidays = []
    for year in range(int(str(dates[0])[:4])-1, int(str(issues[-1])[:4])+2):
        nov = date(year, 11, 1)
        holidays.extend([date(year, 1, 1), date(year, 12, 25),
                         nov + timedelta(days=(3-nov.weekday()) % 7 + 21)])
    holidays = np.array(holidays, dtype='datetime64[D]')
    def flags(days):
        # Holiday occurred within the preceding seven days or the week before.
        delta = (days[:, None]-holidays[None]).astype(int)
        return np.stack((((delta >= 0) & (delta <= 6)).any(1),
                         ((delta >= 7) & (delta <= 13)).any(1)), axis=1)
    return growth, flags


def seasonal_predictions(panel, asof, rows, anchor, mode='adaptive', max_delay=12,
                         integer=False, common_factors=None, halflife=8., prior_weeks=4., statistic='mean',
                         growth_bandwidth=.2):
    """Causal seasonal development with exposure-weighted local partial pooling.

    Historical pairs receive a circular eight-week seasonal kernel, a two-year
    window and a one-year half-life. Adaptive modes add a recent eight-week
    kernel (measured since each pair became observable). National curves are
    independent; other locations borrow four effective weeks from an equally
    weighted state curve. No reference labels or held-out calibration enter.
    """
    dates = panel['dates'].astype('datetime64[D]')
    issues = panel['issuance_dates'].astype('datetime64[D]')
    scheduled = dates + np.timedelta64(4 + 7*rows['lag'], 'D')
    conditional = mode.startswith('conditional')
    if conditional:
        growth, holiday_flags = reporting_context(asof, issues, dates, integer)
    triangle = np.full((len(dates), max_delay+1, asof.shape[2]), np.nan)
    for d in range(max_delay+1):
        wanted = scheduled + np.timedelta64(7*d, 'D')
        wi = np.searchsorted(issues, wanted)
        valid = (wi < len(issues)) & (issues[wi.clip(0, len(issues)-1)] == wanted)
        ti = np.flatnonzero(valid)
        triangle[ti, d] = asof[wi[ti], ti]
    result = np.array(anchor, dtype=float, copy=True)
    factors = np.ones(len(result))
    support = np.zeros(len(result), int)
    states = rows['locations'] != 'US'
    offset = float(integer)
    row_issues = rows['issuance'].astype('datetime64[D]')
    row_dates = rows['boundary'].astype('datetime64[D]')
    for issue in np.unique(row_issues):
        for age in np.unique(rows['age']):
            use = (row_issues == issue) & (rows['age'] == age)
            if not use.any():
                continue
            target = row_dates[use][0]
            if conditional:
                current_w = np.searchsorted(issues, issue)
                current_t = np.searchsorted(dates, target)
                current_growth = growth[current_w, current_t]
            development = np.ones(asof.shape[2])
            common_development = 1.
            counts = np.full(asof.shape[2], 104, int)
            pairs = [(d, d+1) for d in range(int(age), max_delay)] if mode.endswith('chain') else [(int(age), max_delay)]
            for start, end in pairs:
                if start >= end:
                    continue
                completed = scheduled + np.timedelta64(7*end, 'D')
                distance = (target-dates).astype(float)/7
                valid_time = (completed <= issue) & (distance > 0) & (distance <= 104)
                phase = np.abs((distance+26.0893) % 52.1786-26.0893)
                weight = np.exp(-.5*(phase/8)**2) * 2.**(-distance/52)
                if mode != 'seasonal':
                    recent = (issue-completed).astype(float)/7
                    weight += 2.**(-np.maximum(recent, 0)/halflife)
                weight *= valid_time
                a, b = triangle[:, start]+offset, triangle[:, end]+offset
                ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b >= offset)
                w = np.where(ok, weight[:, None], 0.)
                if conditional:
                    # Match contexts at the SAME observed delay, not at maturity.
                    # Target context stays at issuance even for later chain steps.
                    historical_issue = scheduled + np.timedelta64(7*int(age), 'D')
                    hw = np.searchsorted(issues, historical_issue).clip(0, len(issues)-1)
                    historical_growth = growth[hw, np.arange(len(dates))]
                    diff = (historical_growth-current_growth[None])/growth_bandwidth
                    similarity = np.where(np.isfinite(diff), .25+.75*np.exp(-.5*diff**2), 1.)
                    target_report_day = np.array([target + np.timedelta64(4+7*(rows['lag']+start), 'D')])
                    historical_report_days = scheduled + np.timedelta64(7*start, 'D')
                    match = (holiday_flags(historical_report_days) == holiday_flags(target_report_day)).all(1)
                    w *= similarity * np.where(match, 1., .25)[:, None]
                exposure = np.where(ok, a, 0.)*w
                mass = exposure.sum(0)
                observed = (ok & valid_time[:, None]).sum(0)
                # Ratio of weighted totals estimates a mean multiplicative revision.
                numerator = (np.where(ok, b, 0.)*w).sum(0)
                local = np.divide(numerator, mass, out=np.ones_like(mass), where=mass > 0)
                if statistic == 'median':
                    ratios=np.divide(b,a,out=np.ones_like(a),where=ok)
                    local=_weighted_median_columns(ratios,exposure)
                eligible = states & (observed >= 4) & (mass > 0)
                aggregate=np.median if statistic=='median' else np.mean
                common = float(aggregate(local[eligible])) if eligible.sum() >= 3 else 1.
                center = np.where(states, common, 1.)
                pseudo = prior_weeks * np.divide(mass, w.sum(0), out=np.zeros_like(mass), where=w.sum(0)>0)
                # US has its own historical data and no fabricated state exposure.
                pseudo[~states] = 0.
                f = np.divide(numerator+pseudo*center, mass+pseudo,
                              out=center.copy(), where=(mass+pseudo)>0)
                if statistic=='median':
                    f=_weighted_median_columns(np.concatenate((ratios,center[None])),
                                               np.concatenate((exposure,pseudo[None])))
                f = np.where(observed >= 4, f, center)
                development *= f
                common_development *= common
                counts = np.minimum(counts, observed)
            li = rows['location'][use]
            report = rows['baseline_history'][use, -1]
            result[use] = np.where(np.isfinite(report),
                np.maximum(0, (report+offset)*development[li]-offset), anchor[use])
            factors[use], support[use] = development[li], counts[li]
            if common_factors is not None:
                common_factors[use] = common_development
    return result, support, factors


def _weighted_median_columns(values, weights):
    order=np.argsort(values,axis=0)
    sorted_values=np.take_along_axis(values,order,axis=0)
    cumulative=np.cumsum(np.take_along_axis(weights,order,axis=0),axis=0)
    index=(cumulative>=cumulative[-1]/2).argmax(0)
    return np.where(cumulative[-1]>0,sorted_values[index,np.arange(values.shape[1])],1.)


def triangle_predictions(panel, asof, rows, anchor, max_delay=12, window=52,
                         prior_weeks=12, integer=False, common_factors=None):
    """Fit a shared delay curve and shrink state curves toward it over 52 weeks.

    Shared factors weight states equally and exclude the native US aggregate.
    Each local L1 fit receives twelve mean-exposure pseudo-weeks at the shared
    factor. With fewer than four local pairs use the common factor, if supported
    by at least three states. US/national-only series use their own history.
    Only report pairs observed by the issuance enter; never reference labels.
    """
    dates = panel['dates'].astype('datetime64[D]')
    issues = panel['issuance_dates'].astype('datetime64[D]')
    scheduled = dates + np.timedelta64(4 + 7*rows['lag'], 'D')
    first = np.searchsorted(issues, scheduled)
    triangle = np.full((len(dates), max_delay+1, asof.shape[2]), np.nan)
    for d in range(max_delay+1):
        w = first+d
        valid = (w<len(issues)) & (first<len(issues))
        valid &= issues[w.clip(0,len(issues)-1)] == scheduled+np.timedelta64(7*d,'D')
        t = np.flatnonzero(valid)
        triangle[t,d] = asof[w[t],t]
    result = np.array(anchor,dtype=float,copy=True)
    support = np.zeros(len(result),int)
    factors = np.ones(len(result))
    row_issues = rows['issuance'].astype('datetime64[D]')
    row_dates = rows['boundary'].astype('datetime64[D]')
    for issue in np.unique(row_issues):
        for age in np.unique(rows['age']):
            use = (row_issues==issue) & (rows['age']==age)
            if not use.any():
                continue
            target = row_dates[use][0]
            development = np.ones(asof.shape[2])
            shared_development = 1.
            counts = np.full(asof.shape[2],window,int)
            for d in range(int(age),max_delay):
                available = (scheduled+np.timedelta64(7*(d+1),'D')<=issue) & (dates!=target)
                ts = np.flatnonzero(available)[-window:]
                a,b = triangle[ts,d],triangle[ts,d+1]
                ok = np.isfinite(a)&np.isfinite(b)&(a>=0)&(b>=0)
                n = ok.sum(0)
                # Local L1 slope: weighted median of paired development ratios.
                # State curves shrink toward the shared reporting pattern below.
                offset = 1. if integer else 0.
                aa,bb = a+offset,b+offset
                valid = ok & (aa>0)
                exposure = np.where(valid,aa,0)
                ratio = np.divide(bb,aa,out=np.ones_like(aa),where=valid)
                # Each state has unit total exposure in the common curve:
                # a large state cannot overwhelm smaller states by population.
                sums=exposure.sum(0)
                state=(rows['locations']!='US') & (n>=4) & (sums>0)
                common=1.
                pooled=state.sum()>=3
                if pooled:
                    shared_weight=np.divide(exposure[:,state],sums[state][None])
                    rr=ratio[:,state].ravel();ww=shared_weight.ravel()
                    order=np.argsort(rr)
                    common=rr[order[np.searchsorted(np.cumsum(ww[order]),ww.sum()/2)]]
                center=np.where(rows['locations']=='US',1.,common)
                pseudo=prior_weeks*sums/np.maximum(valid.sum(0),1)
                # National counts are not another state in the common pool.
                pseudo=np.where(rows['locations']=='US',min(prior_weeks,4)*sums/np.maximum(valid.sum(0),1),pseudo)
                ratio=np.concatenate((ratio,center[None]))
                weight=np.concatenate((exposure,pseudo[None]))
                order=np.argsort(ratio,axis=0)
                sorted_ratio=np.take_along_axis(ratio,order,axis=0)
                cumulative=np.cumsum(np.take_along_axis(weight,order,axis=0),axis=0)
                index=(cumulative>=cumulative[-1]/2).argmax(0)
                f=sorted_ratio[index,np.arange(asof.shape[2])]
                fallback=np.where((rows['locations']!='US') & pooled,common,1.)
                f=np.where((n>=4)&(sums>0),f,fallback)
                shared_development *= common
                development *= f
                counts = np.minimum(counts,n)
            li = rows['location'][use]
            report = rows['baseline_history'][use,-1]
            # CDC's count expectation includes one-count correction; rates do not.
            offset = 1. if integer else 0.
            result[use] = np.where(np.isfinite(report),
                np.maximum(0,(report+offset)*development[li]-offset),anchor[use])
            support[use] = counts[li]
            factors[use] = development[li]
            if common_factors is not None:
                common_factors[use] = shared_development
    return result, support, factors


def choose_shrinkage(truth, anchor, delta, scale, location, upper_bound=None, min_gain=.01, integer=False):
    """Fit strength by equal-location MAE on the supplied calibration rows; ties prefer identity."""
    candidates = (0., .25, .5, 1.)
    losses = []
    for alpha in candidates:
        prediction = np.maximum(0, anchor+alpha*delta)
        if upper_bound is not None:
            prediction = np.minimum(prediction,upper_bound)
        if integer:
            prediction = np.rint(prediction)
        error = np.abs(prediction-truth)/scale
        losses.append(float(np.mean([error[location==l].mean() for l in np.unique(location)])))
    chosen = int(np.argmin(losses))
    # Require a measurable gain; floating-point ties must not activate corrections.
    if losses[chosen] >= losses[0]*(1-min_gain):
        chosen = 0
    return candidates[chosen], losses



def fit_gap(features, truth, anchor, scale, location, issuance, penalty=10.):
    """Regularized linear log-residual model, equal total weight per location."""
    target=np.log1p(truth/scale)-np.log1p(anchor/scale)
    dates=issuance.astype('datetime64[D]')
    weight=2.**(-((dates.max()-dates).astype(float)/7)/26)
    totals=np.bincount(location,weights=weight)
    weight/=totals[location]
    weight/=weight.mean()
    ridge=np.eye(features.shape[1])*penalty
    ridge[0,0]=0.
    return np.linalg.solve(features.T@(features*weight[:,None])+ridge,
                           features.T@(weight*target))


def predict_gap(features, coefficients, anchor, scale):
    log_value=np.log1p(anchor/scale)+features@coefficients
    return np.maximum(0,np.expm1(np.clip(log_value,-20,20)))*scale


def seasonal_gap_predictions(panel, asof, rows, anchor, integer=False, trend=False):
    """Bridge missing reports with visible prior-year seasonal growth.

    One and two years ago receive weights 1 and 1/2. Smooth historical levels
    across three weeks; use log growth so up/down factors combine symmetrically.
    Optional recent log slope shrinks to that seasonal slope over a four-week
    horizon. All source values are taken from the current issuance vintage.
    """
    dates = panel['dates'].astype('datetime64[D]')
    issues = panel['issuance_dates'].astype('datetime64[D]')
    result = np.asarray(anchor, dtype=float).copy()
    offset = float(integer)
    for issue in np.unique(rows['issuance']):
        wi = np.searchsorted(issues, np.datetime64(issue))
        use = np.flatnonzero((rows['issuance']==issue)&~np.isfinite(rows['baseline_history'][:,-1]))
        if not len(use):
            continue
        data = asof[wi]
        for i in use:
            ti = np.searchsorted(dates, np.datetime64(rows['boundary'][i]))
            li = rows['location'][i]
            known = np.flatnonzero(np.isfinite(data[:ti, li]))
            if not len(known):
                continue
            last = known[-1]
            gap = ti-last
            if gap > 12:
                continue
            log_growth, weights = [], []
            for years in (1,2):
                start, end = last-52*years, ti-52*years
                if start < 1:
                    continue
                a, b = data[start-1:start+2,li], data[end-1:end+2,li]
                if not np.isfinite(a).any() or not np.isfinite(b).any():
                    continue
                a,b = np.nanmean(a)+offset, np.nanmean(b)+offset
                if a > 0 and b > 0:
                    log_growth.append(np.log(b/a));weights.append(2.**(1-years))
            seasonal = np.average(log_growth,weights=weights) if weights else 0.
            growth = seasonal
            if trend:
                recent = known[known>=last-5]
                if len(recent)>=3:
                    v=data[recent,li]+offset
                    valid=v>0
                    if valid.sum()>=3:
                        slope=np.polyfit(recent[valid]-last,np.log(v[valid]),1)[0]
                        # Limit extrapolated weekly growth to a doubling/halving.
                        slope=np.clip(slope,-np.log(2),np.log(2))
                        weight=4*(1-np.exp(-gap/4))/gap
                        growth=(1-weight)*seasonal+weight*gap*slope
            result[i]=max(0,(data[last,li]+offset)*np.exp(np.clip(growth,-4,4))-offset)
    return result


def proxy_gap_predictions(panel, asof, rows, anchor, name, integer=False):
    """Transport observed national pathogen growth to each missing local target.

    Current outpatient claims bridge flu/COVID outages. RSV uses current NSSP
    when possible, then its prior-year seasonal growth (avoiding NHSN reporting
    regime changes). If neither is supported, use a damped recent national trend.
    Local levels remain anchored to their last actual report, at most 12 weeks old.
    Longer gaps use the median of up to 104 visible historical weeks; with no
    visible history use zero. Never inherit a frozen-reference fallback.
    """
    dates=panel['dates'].astype('datetime64[D]')
    issues=panel['issuance_dates'].astype('datetime64[D]')
    pathogen=next(v for v in ('flu','covid','rsv') if '_'+v+'_' in name)
    source='nssp_rsv_proportion' if pathogen=='rsv' else 'outpatient_'+pathogen
    if source in panel['target_names']:
        proxy=panel['asof_targets'][...,list(panel['target_names']).index(source)]
    else:
        proxy=panel['asof_covariates'][...,list(panel['covariate_names']).index(source)]
    us=list(panel['locations']).index('US')
    result=np.asarray(anchor,dtype=float).copy()
    offset=float(integer)
    for issue in np.unique(rows['issuance']):
        wi=np.searchsorted(issues,np.datetime64(issue))
        data=asof[wi]
        use=np.flatnonzero((rows['issuance']==issue)&~np.isfinite(rows['baseline_history'][:,-1]))
        for i in use:
            ti=np.searchsorted(dates,np.datetime64(rows['boundary'][i]));li=rows['location'][i]
            known=np.flatnonzero(np.isfinite(data[:ti,li]))
            if not len(known):
                result[i]=0.
                continue
            if ti-known[-1]>12:
                result[i]=np.median(data[known[-104:],li])
                continue
            last=known[-1];gap=ti-last
            growth=None
            # Require the target week's proxy. Do not treat a stale proxy as current.
            if np.isfinite(proxy[wi,ti,us]) and proxy[wi,ti,us]>0:
                # Trailing two-week means reduce claims sampling noise.
                a=proxy[wi,max(0,last-1):last+1,us]
                b=proxy[wi,max(0,ti-1):ti+1,us]
                if np.isfinite(a).any() and np.nanmean(a)>0:
                    growth=np.log(np.nanmean(b)/np.nanmean(a))
            if growth is None and pathogen=='rsv' and last>=53:
                # Borrow the national ED seasonal shape, not historical admission
                # counts near the introduction of mandatory reporting.
                a=proxy[wi,last-53:last-50,us];b=proxy[wi,ti-53:ti-50,us]
                if np.isfinite(a).any() and np.isfinite(b).any() and np.nanmean(a)>0 and np.nanmean(b)>0:
                    growth=np.log(np.nanmean(b)/np.nanmean(a))
            if growth is None:
                recent=np.arange(max(0,last-2),last+1)
                y=data[recent,us]+offset
                valid=np.isfinite(y)&(y>0)
                slope=np.polyfit(recent[valid]-last,np.log(y[valid]),1)[0] if valid.sum()>=2 else 0.
                growth=np.clip(slope,-.35,.35)*4*(1-np.exp(-gap/4))
            result[i]=max(0,(data[last,li]+offset)*np.exp(np.clip(growth,-.35*gap,.35*gap))-offset)
    return result
