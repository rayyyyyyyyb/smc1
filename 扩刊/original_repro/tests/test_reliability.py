import pytest

from smc_repro.reliability import weibull_cdf, weibull_interval_failure_probability


def test_interval_probability_zero_duration_is_zero() -> None:
    assert weibull_interval_failure_probability(100.0, 0.0, 500.0, 2.0) == 0.0


def test_interval_probability_increases_with_age_and_duration() -> None:
    young = weibull_interval_failure_probability(50.0, 10.0, 500.0, 2.0)
    old = weibull_interval_failure_probability(300.0, 10.0, 500.0, 2.0)
    longer = weibull_interval_failure_probability(300.0, 30.0, 500.0, 2.0)
    assert 0.0 <= young < old < longer <= 1.0


def test_conditional_probability_matches_survival_ratio() -> None:
    age, duration, eta, beta = 100.0, 40.0, 500.0, 2.0
    expected = 1.0 - (1.0 - weibull_cdf(age + duration, eta, beta)) / (
        1.0 - weibull_cdf(age, eta, beta)
    )
    assert weibull_interval_failure_probability(age, duration, eta, beta) == pytest.approx(expected)


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_interval_probability_rejects_non_finite_inputs(index: int, value: float) -> None:
    arguments = [100.0, 10.0, 500.0, 2.0]
    arguments[index] = value

    with pytest.raises(ValueError, match="finite"):
        weibull_interval_failure_probability(*arguments)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_cdf_rejects_non_finite_age(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        weibull_cdf(value)
