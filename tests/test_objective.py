"""Check scientific weighting invariants, sparse scaling, and masked gradients."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

from tapestry.models.b0 import fair_crps_cells
from tapestry.models.objective import loss_cell_weights, loss_scales


def test_seasons_channels_and_locations_have_explicit_weight_despite_missingness():
    episodes = []
    for day in ['2023-10-07', '2024-10-05', '2024-10-12']:
        y = np.ones((1, 6, 2, 3), dtype=np.float32)
        if day.startswith('2023'):
            y[:, 1:, 1] = 0  # Only flu admissions available in season one.
        if day.endswith('12'):
            y[:, :, 1, 0] = 0  # Fewer observations at the first state.
        episodes.append(dict(Y=y, target_dates=[day], locations=['NC', 'CA', 'US']))
    w = loss_cell_weights(episodes)
    assert w.sum() == pytest.approx(1)
    assert w[0].sum() == pytest.approx(.5)
    assert w[1:].sum() == pytest.approx(.5)
    np.testing.assert_allclose(w.sum(axis=(0, 1, 2)), [.4, .4, .2], rtol=1e-6)
    np.testing.assert_allclose(w[1:].sum(axis=(0, 1, 3)), np.array([1, 1, 1, .5, .5, .5]) / 9, rtol=1e-6)
    mask = torch.tensor(np.stack([e['Y'][:, :, 1] for e in episodes]))
    truth = torch.ones_like(mask)
    truth[mask == 0] = float('nan')
    samples = torch.full((3, *mask.shape), 2., requires_grad=True)
    scales = torch.ones(6, 3)
    weighted = torch.tensor(w) / scales
    full = (fair_crps_cells(samples, truth, mask) * weighted).sum()
    chunks = sum((fair_crps_cells(samples[:, i:i+1], truth[i:i+1], mask[i:i+1]) * weighted[i:i+1]).sum()
                 for i in range(len(episodes)))
    assert full.item() == pytest.approx(chunks.item())
    full.backward()
    assert not samples.grad[:, mask == 0].any()


def test_loss_scales_use_native_values_unique_weeks_and_sparse_pooling():
    panel = np.ones((30, 6, 2, 3), dtype=np.float32)
    panel[:, :3, 0] = [10, 100, 1000]
    panel[:, 3:, 0] = 0
    scales = np.array(loss_scales(panel))
    np.testing.assert_allclose(scales[:3], [[10, 100, 1000]] * 3)
    np.testing.assert_allclose(scales[3:], .001)
    panel[:, :, 1, 0] = 0
    panel[0, :, 1, 0] = 1
    sparse = np.array(loss_scales(panel))
    assert sparse[0, 0] == pytest.approx(10 / 26 + 1000 * 25 / 26)
    panel[1:, :, 0, 0] = float('nan')
    np.testing.assert_array_equal(loss_scales(panel), sparse)
