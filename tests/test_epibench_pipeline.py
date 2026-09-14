"""Exercise EpiBench itself at the FIPS/zero-reference boundary of the adapter."""
import importlib.util
import shutil

import numpy as np
import pandas as pd
import pytest

from tapestry.evaluation.epibench import package_source, score_case
from tapestry.evaluation.hubs import QCOLS
from tapestry.models.season_cv import LEVELS


def test_full_command_numeric_fips_and_zero_reference(tmp_path):
    if not shutil.which('Rscript') or importlib.util.find_spec('epibench') is None:
        pytest.skip('Requires local R and EpiBench')
    units = pd.DataFrame(dict(reference_date=['2025-01-04', '2025-01-11'],
        target_end_date=['2025-01-04', '2025-01-11'], location=['01', '01'], horizon=[0, 0], observed=[0., 0.]))
    ensemble = units.assign(model='FluSight-ensemble')
    ensemble[QCOLS] = 0.
    candidate = units.assign(model='candidate')
    values = np.arange(5)
    candidate[QCOLS] = values
    case = dict(hub='flusight', target='wk inc flu hosp', ensemble='FluSight-ensemble')
    wide = pd.concat([ensemble, candidate], ignore_index=True)
    result = score_case(wide, units, case, tmp_path / 'scored')
    assert set(result.location) == {'01'}
    assert result.rwis.isna().all()
    error = -values
    expected = 2 * np.maximum(LEVELS * error, (LEVELS - 1) * error).mean()
    np.testing.assert_allclose(result.loc[result.model == 'candidate', 'wis'], expected)
    np.testing.assert_allclose(result.loc[result.model == 'FluSight-ensemble', 'wis'], 0.)
    # Same complete inputs may resume EpiBench's otherwise non-overwriting output.
    pd.testing.assert_frame_equal(result, score_case(wide, units, case, tmp_path / 'scored'))
    with pytest.raises(ValueError, match='Missing frozen tasks'):
        score_case(wide.iloc[:-1], units, case, tmp_path / 'missing')
    with pytest.raises(ValueError, match='Inputs or scorer changed'):
        changed = wide.copy()
        changed.loc[changed.model == 'candidate', QCOLS] += 1
        score_case(changed, units, case, tmp_path / 'scored')


def test_package_source_requires_install_or_explicit_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, 'find_spec', lambda _: None)
    with pytest.raises(ModuleNotFoundError, match='uv sync --upgrade-package epibenchmark'):
        package_source()
    source = tmp_path / 'src' / 'epibench'
    source.mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match='scoring package missing'):
        package_source(tmp_path)
    (source / 'score.py').write_text('# development scorer\n')
    assert package_source(tmp_path) == source
