from prism.config.model_profile import Cost
from prism.utils.cost import compute_cost


def test_zero_cost_profile():
    c = Cost(input_per_mtok=0.0, output_per_mtok=0.0)
    assert compute_cost(c, tokens_in=1000, tokens_out=1000) == 0.0


def test_non_zero_cost():
    c = Cost(input_per_mtok=15.0, output_per_mtok=75.0)
    # 1M tokens input = $15, 1M tokens output = $75
    assert compute_cost(c, tokens_in=1_000_000, tokens_out=0) == 15.0
    assert compute_cost(c, tokens_in=0, tokens_out=1_000_000) == 75.0


def test_partial():
    c = Cost(input_per_mtok=3.0, output_per_mtok=15.0)
    got = compute_cost(c, tokens_in=100_000, tokens_out=50_000)
    expected = 3.0 * 0.1 + 15.0 * 0.05
    assert abs(got - expected) < 1e-9
