"""Wednesday reconstruction rows and causal revision features at T-0/T-1 boundaries."""
import json
from datetime import date

import numpy as np

from .build import context_end, LAG_ONE_COVARIATES
from .cv import season

def signals(panel):
    """National-only covariates occur once, never replicated into 52 training labels."""
    for name_key, value_key, asof_key in (
        ('target_names', 'targets', 'asof_targets'),
        ('covariate_names', 'covariates', 'asof_covariates'),
        ('covariate_national_names', 'covariates_national', 'asof_covariates_national'),
    ):
        for k, name in enumerate(panel[name_key].astype(str)):
            national = value_key == 'covariates_national'
            truth, asof = panel[value_key][..., k], panel[asof_key][..., k]
            yield name, (truth[:, None] if national else truth), (asof[..., None] if national else asof), \
                np.array(['US']) if national else panel['locations'].astype(str)


def boundary_rows(panel, name, truth, asof, locations, lookback, weeks=1):
    if weeks > 1:
        parts = [_age_rows(panel, name, truth, asof, locations, lookback, age) for age in range(weeks)]
        return {key: parts[0][key] if key in ('locations', 'lag') else np.concatenate([p[key] for p in parts])
                for key in parts[0]}
    return _age_rows(panel, name, truth, asof, locations, lookback, 0)


def _age_rows(panel, name, truth, asof, locations, lookback, age):
    dates = panel['dates'].astype('datetime64[D]')
    issuance = panel['issuance_dates'].astype('datetime64[D]')
    lag = int(name in LAG_ONE_COVARIATES)
    ends = np.array([context_end(str(d)) for d in issuance], dtype='datetime64[D]') - np.timedelta64(7 * (lag + age), 'D')
    boundary = np.searchsorted(dates, ends)
    valid = (boundary < len(dates)) & (ends >= dates[0])
    wi = np.flatnonzero(valid)
    bi = boundary[valid]
    if not np.array_equal(dates[bi], ends[valid]):
        raise ValueError('Boundary weeks do not align with the panel calendar')
    width = max(12, lookback)  # fixed 12-week baseline/support across all history ablations
    history = bi[:, None] + np.arange(1 - width, 1)
    values = asof[wi[:, None], history.clip(0)]
    values = np.where((history >= 0)[..., None], values, np.nan)
    # Before any report exists in this archive/location there is no training evidence
    # of Wednesday availability. Do not manufacture thousands of outage examples.
    active = np.maximum.accumulate(np.isfinite(asof).any(axis=1), axis=0)[wi]
    labels = truth[bi]
    n, _, l = values.shape
    values = values.transpose(0, 2, 1).reshape(n * l, width)
    day = np.array([date.fromisoformat(str(d)).timetuple().tm_yday for d in ends[valid]])
    angle = 2 * np.pi * day / 365.25
    return dict(values=values[:, -lookback:].astype(np.float32),
                baseline_history=values[:, -12:].astype(np.float32),
                truth=labels.reshape(-1).astype(np.float32), active=active.reshape(-1),
                location=np.tile(np.arange(l), n), locations=locations,
                issuance=np.repeat(issuance[valid].astype(str), l),
                boundary=np.repeat(ends[valid].astype(str), l),
                age=np.full(n * l, age, dtype=np.int64),
                elapsed_weeks=np.full(n * l, lag + age + 4 / 7, dtype=np.float32),
                calendar=np.repeat(np.stack((np.sin(angle), np.cos(angle)), 1), l, axis=0).astype(np.float32),
                lag=lag)


def split(panel, rows, scenario, fold):
    """Forward-only partitions, with a training-label age gap and mature score dates.

    Labels still come from the frozen later snapshot: this is retrospective
    evaluation, not a claim that those exact labels were known at the fitting date.
    """
    metadata = json.loads(str(panel['metadata']))
    truth_day = np.datetime64(metadata['truth_day'], 'D')
    maturity = np.timedelta64(7 * scenario.finalization_maturity, 'D')
    issuance = np.array(rows['issuance'], dtype='datetime64[D]')
    boundary = np.array(rows['boundary'], dtype='datetime64[D]')
    if scenario.finalization_cv == 'rolling':
        all_issues = panel['issuance_dates'].astype('datetime64[D]')
        ends = np.array([context_end(str(d)) for d in all_issues], dtype='datetime64[D]')
        eligible = all_issues[ends <= truth_day - maturity]
        if len(eligible) < 12:
            raise ValueError('Rolling evaluation needs twelve eligible Wednesdays')
        block = int(fold.rsplit('_', 1)[1]) - 1
        score_issues = eligible[-12:][4 * block:4 * block + 4]
        start = score_issues[0]
        scoring = np.isin(issuance, score_issues)
    else:
        from .cv import season_start
        start = np.datetime64(season_start(int(fold[:4])))
        scoring = np.array([season(d) == fold for d in rows['boundary']])
    eligible = np.isfinite(rows['truth']) & rows['active']
    training = eligible & (issuance < start) & (boundary < start - maturity)
    scoring &= eligible & (boundary <= truth_day - maturity)
    # A longer window can otherwise score the same observation week that appeared
    # at a younger age during fitting. Purge every scored label date from training.
    score_dates = np.unique(boundary[scoring])
    training &= ~np.isin(boundary, score_dates)
    return training, scoring, dict(fit_cutoff=str(start), truth_day=str(truth_day),
                                   maturity_weeks=scenario.finalization_maturity,
                                   protocol='forward_frozen_reference_labels')


def training_statistics(rows, training):
    """Per-location scales and missing-input anchors derived only from training labels."""
    # Multiple issuance/age rows can share the same reference label. Count that
    # location/week once when estimating scales and fallback values.
    selected = np.flatnonzero(training)
    keys = np.stack((rows['boundary'][selected], rows['location'][selected].astype(str)), axis=1)
    _, first = np.unique(keys, axis=0, return_index=True)
    unique_training = np.zeros_like(training)
    unique_training[selected[first]] = True
    training = unique_training
    y = rows['truth']
    pooled = y[training]
    positive = pooled[pooled > 0]
    global_scale = max(float(np.mean(positive)) if len(positive) else 1., 1e-6)
    count = len(rows['locations'])
    scales, fallback, trained = np.full(count, global_scale), np.zeros(count), np.zeros(count, bool)
    for l in range(count):
        values = y[training & (rows['location'] == l)]
        if len(values):
            scales[l] = max(float(values.mean()), .01 * global_scale, 1e-6)
            fallback[l], trained[l] = np.median(values), True
    return scales.astype(np.float32), fallback.astype(np.float32), trained


def validation_split(rows, training, weeks=12, maturity=4):
    """Last training Wednesdays validate shrinkage; purge their labels from inner fit."""
    issues = np.unique(rows['issuance'][training])
    if len(issues) < weeks + 12:
        return training.copy(), np.zeros_like(training)
    start = issues[-weeks]
    validation = training & (rows['issuance'] >= start)
    cutoff = np.datetime64(start) - np.timedelta64(7 * maturity, 'D')
    fit = training & (rows['issuance'] < start) & (rows['boundary'].astype('datetime64[D]') < cutoff)
    fit &= ~np.isin(rows['boundary'], rows['boundary'][validation])
    return fit, validation



def gap_features(panel, rows, target, training, normalizers=None):
    """Cross-source gap inputs; target's contemporaneous value is always excluded."""
    wi = np.searchsorted(panel['issuance_dates'].astype(str),rows['issuance'])
    ti = np.searchsorted(panel['dates'].astype(str),rows['boundary'])
    origin = ti+rows['age']+rows['lag']
    target_locations = rows['locations'][rows['location']]
    features=[np.ones(len(wi)),rows['age']/8,rows['calendar'][:,0],rows['calendar'][:,1]]
    names=['intercept','age','annual_sin','annual_cos']
    for l in range(len(rows['locations'])):
        features.append((rows['location']==l).astype(float));names.append('location:'+str(l))
    supplied = normalizers is not None
    normalizers={} if normalizers is None else normalizers
    for name,_,asof,locations in signals(panel):
        mapping={v:i for i,v in enumerate(locations)}
        li=np.array([mapping.get(v,-1) for v in target_locations]) if len(locations)>1 else np.zeros(len(wi),int)
        for back in (0,1,2,4):
            if name==target and back==0:
                continue
            tt=ti-back
            valid=(tt>=0)&(li>=0)&(tt<=origin-int(name in LAG_ONE_COVARIATES))
            values=np.where(valid,asof[wi,tt.clip(0),li.clip(0)],np.nan)
            key=f'{name}:{back}'
            if not supplied:
                scale=np.ones(len(rows['locations']))
                for l in range(len(scale)):
                    v=values[training&(rows['location']==l)&np.isfinite(values)&(values>0)]
                    scale[l]=max(float(np.median(v)) if len(v) else 1.,1e-8)
                normalizers[key]=scale
            x=np.log1p(np.maximum(values,0)/normalizers[key][rows['location']])
            features.extend((np.nan_to_num(x,nan=0),np.isfinite(values).astype(float)))
            names.extend((key,key+':observed'))
    return np.column_stack(features),names,normalizers
