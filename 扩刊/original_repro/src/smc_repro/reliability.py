from __future__ import annotations

import math


def _validate(age: float, duration: float, eta: float, beta: float) -> None:
    if not all(math.isfinite(value) for value in (age, duration, eta, beta)):
        raise ValueError("age, duration, eta, and beta must be finite")
    if age < 0 or duration < 0:
        raise ValueError("age and duration must be non-negative")
    if eta <= 0 or beta <= 0:
        raise ValueError("eta and beta must be positive")


def weibull_cdf(age: float, eta: float = 500.0, beta: float = 2.0) -> float:
    _validate(age, 0.0, eta, beta)
    return 1.0 - math.exp(-((age / eta) ** beta))


def weibull_interval_failure_probability(
    age: float,
    duration: float,
    eta: float = 500.0,
    beta: float = 2.0,
) -> float:
    _validate(age, duration, eta, beta)
    if duration == 0.0:
        return 0.0
    cumulative_hazard_increment = ((age + duration) / eta) ** beta - (age / eta) ** beta
    value = 1.0 - math.exp(-cumulative_hazard_increment)
    return min(1.0, max(0.0, value))


def health_from_effective_age(
    age: float,
    eta: float = 500.0,
    beta: float = 2.0,
) -> float:
    return 100.0 * (1.0 - weibull_cdf(age, eta, beta))
