"""Silent scientific data failures: regime eligibility and as-of missingness."""
import numpy as np
from tapestry.model_data.forward import recent_eligibility, asof_panel
from tapestry.model_data.wednesday import VintageArchive


def test_regime_mask_does_not_apply_to_nssp_or_turn_missing_into_zero_revision():
    available=np.ones((2,6,1),bool);available[1,0]=False
    covered=np.ones_like(available);covered[0,4]=False
    revision,missing=recent_eligibility(['2024-10-26','2024-11-02'],available,covered)
    assert not revision[0,:3].any()
    assert revision[0,3,0] and not revision[0,4,0]
    assert not revision[1,0,0] and missing[1,0,0]
    assert not missing[0,4,0]


def test_later_final_does_not_fill_a_missing_four_day_report():
    archive=VintageArchive()
    archive.add('delphi_nhsn','2025-01-08','2025-01-04',0,'US',10.)
    archive.add('delphi_nhsn','2025-01-16','2025-01-11',0,'US',20.)
    archive.add('delphi_nhsn','2025-02-01','2025-01-04',0,'US',15.)
    values,available,covered,_,_=asof_panel(archive,['2025-01-04','2025-01-11'],['US'],'2025-01-15')
    assert values[0,0,0]==10.  # This week's own preliminary, not a later final.
    assert available[0,0,0] and not available[1,0,0]
    assert covered[1,0,0]  # Active source, but the four-day report is unavailable.


def test_forward_training_masks_preserve_forecasts_and_cutoff():
    from tapestry.model_data.wednesday import WednesdayDataset
    from tapestry.model_data.forward import DATASET, CUTOFF
    from tapestry.models.forward import episodes
    ds = WednesdayDataset.load(DATASET)
    natural, final = episodes(ds, True), episodes(ds, True, finalized=True)
    for a, b in zip(natural, final):
        i = a['index']
        assert a['issuance_date'] <= CUTOFF
        np.testing.assert_array_equal(a['Y'][2:], b['Y'][2:])
        np.testing.assert_array_equal(a['X'][:-2], b['X'][:-2])
        valid = a['Y'][:, :, 1].astype(bool)
        assert not valid[np.asarray(a['target_dates']) > '2025-06-28'].any()
        eligible = ds.arrays['revision_eligible'][i] | ds.arrays['reconstruction_eligible'][i]
        assert not (valid[:2] & ~eligible).any()
        assert not valid[:2][np.asarray(a['target_dates'][:2]) < '2024-11-01', :3].any()
    test = episodes(ds, False)
    for e in test:
        i=e['index']
        np.testing.assert_array_equal(e['X'][:,:,1].astype(bool),ds.arrays['X_available'][i])
        assert not e['X'][:,:,2].any()


def test_recent_aggregation_keeps_missing_report_baseline_undefined():
    import pandas as pd
    from tapestry.evaluation.forward import geographic_mean
    group = pd.DataFrame({'geography':['states_dc','US'], 'location':['NC','US'],
                          'baseline_ae':[np.nan,np.nan], 'scaled_ae':[1.,2.]})
    result = geographic_mean(group, ['baseline_ae','scaled_ae'])
    assert np.isnan(result['baseline_ae'])
    assert np.isclose(result['scaled_ae'], 1.2)
