"""Round-trip, order-independence, and typo-safety of the `Scenario` string codec."""
import pytest

from tapestry.model.scenario import Scenario


def test_default_scenario_has_an_empty_string():
    assert Scenario().scenario_string == ''
    assert Scenario.from_string('') == Scenario()


def test_only_non_default_fields_are_emitted():
    scenario = Scenario(width=32)
    assert scenario.scenario_string == 'width=32'
    assert Scenario.from_string('width=32') == scenario


def test_round_trip_encode_decode_every_field_type():
    scenario = Scenario(width=32, lr=.01, geography=False, encoder='conv', covariate_set='inpatient+kinsa')
    assert Scenario.from_string(scenario.scenario_string) == scenario


def test_token_order_does_not_matter():
    a = Scenario.from_string('width=32,lr=0.01,geography=0')
    b = Scenario.from_string('geography=0,width=32,lr=0.01')
    assert a == b


def test_a_field_added_later_can_be_appended_anywhere_and_still_resolves():
    # Simulates a saved string from before a hypothetical new field existed:
    # appending a new key=value anywhere must still resolve, taking the
    # dataclass default for anything not named.
    base = Scenario(width=32).scenario_string
    assert Scenario.from_string(base + ',lookback=8') == Scenario(width=32, lookback=8)
    assert Scenario.from_string('lookback=8,' + base) == Scenario(width=32, lookback=8)


def test_partial_strings_fill_remaining_fields_from_defaults():
    scenario = Scenario.from_string('encoder=conv')
    assert scenario.encoder == 'conv'
    assert scenario.width == Scenario().width
    assert scenario.lookback == Scenario().lookback


def test_unknown_field_order_tolerance_with_many_tokens():
    tokens = ['width=32', 'latent=8', 'lr=0.01', 'encoder=conv', 'geography=0', 'dynamics=0']
    import itertools
    results = {Scenario.from_string(','.join(perm)) for perm in itertools.permutations(tokens)}
    assert len(results) == 1


@pytest.mark.parametrize('field,bad', [
    ('encoder', 'mlpx'), ('decoder', 'fancy'), ('spatial', 'nope'), ('noise', 'weird'),
    ('us_error', 'huh'), ('head_sharing', 'group'), ('fit_partition', 'partial'),
    ('count_transform', 'square'), ('ed_transform', 'sigmoid'), ('loss_weights', 'made_up'),
    ('input_mode', 'live'),
])
def test_invalid_enum_value_still_raises(field, bad):
    with pytest.raises(ValueError):
        Scenario.from_string(f'{field}={bad}')


def test_unknown_field_name_raises():
    with pytest.raises(ValueError):
        Scenario.from_string('not_a_real_field=1')


def test_malformed_token_raises():
    with pytest.raises(ValueError):
        Scenario.from_string('width')


def test_unknown_covariate_group_raises():
    with pytest.raises(ValueError):
        Scenario(covariate_set='not_a_group')


def test_mask_probabilities_must_sum_to_one():
    with pytest.raises(ValueError):
        Scenario(mask_recent=.5, mask_gap=.5, mask_outage=.5)


def test_float_fields_round_trip_past_six_significant_digits():
    scenario = Scenario(lr=.0001234567)
    assert Scenario.from_string(scenario.scenario_string) == scenario


def test_covariate_set_spelling_is_canonicalized():
    a = Scenario(covariate_set='inpatient+kinsa')
    b = Scenario(covariate_set='kinsa+inpatient')
    c = Scenario(covariate_set='inpatient+inpatient+kinsa')
    assert a == b == c
    assert a.run_id == b.run_id == c.run_id


def test_covariate_set_and_input_mode_reproduce_b0_b1_b2_shapes():
    plain_b0 = Scenario(covariate_set='', input_mode='finalized')
    b1_direct = Scenario(covariate_set='', input_mode='vintaged', supplied_final=True)
    b2 = Scenario(covariate_set='inpatient', input_mode='vintaged', supplied_final=True)
    assert plain_b0.model_options()['supplied_final'] is False
    assert b1_direct.model_options()['supplied_final'] is True
    assert b2.covariate_set == 'inpatient'
