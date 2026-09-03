"""Marginal planning costs shared by FHP and switching-SSM models."""

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import sympy


@dataclass(frozen=True)
class LinearMarginalCostSchedule:
    """Δτ(j) = a + b*j for j >= 1, with finite a > 0 and b >= 0."""

    a: float
    b: float = 0.0

    def __post_init__(self):
        if not (math.isfinite(self.a) and self.a > 0):
            raise ValueError(f"Cost schedule requires finite a>0, got a={self.a}")
        if not (math.isfinite(self.b) and self.b >= 0):
            raise ValueError(f"Cost schedule requires finite b>=0, got b={self.b}")

    def delta_tau(self, k_plus_1: int) -> float:
        k_plus_1 = int(k_plus_1)
        if k_plus_1 < 1:
            raise ValueError(f"k_plus_1 must be >= 1, got {k_plus_1}")
        return float(self.a + self.b * k_plus_1)

    def validate_positive(self, k_max: int) -> None:
        if int(k_max) < 0:
            raise ValueError(f"k_max must be >= 0, got {k_max}")
        # The parameter restrictions ensure positive costs at every stage.


@dataclass(frozen=True)
class ExponentialMarginalCostSchedule:
    """Δτ(j) = a*exp(growth*(j-1)); a is the first additional stage's cost."""

    a: float
    growth: float

    def __post_init__(self):
        if not (math.isfinite(self.a) and self.a > 0):
            raise ValueError(f"Cost schedule requires finite a>0, got a={self.a}")
        if not (math.isfinite(self.growth) and self.growth >= 0):
            raise ValueError(
                f"Cost schedule requires finite growth>=0, got growth={self.growth}"
            )

    def delta_tau(self, k_plus_1: int) -> float:
        k_plus_1 = int(k_plus_1)
        if k_plus_1 < 1:
            raise ValueError(f"k_plus_1 must be >= 1, got {k_plus_1}")
        if k_plus_1 == 1 or self.growth == 0:
            return float(self.a)
        # Combining logs avoids premature overflow when a is very small.
        log_cost = math.log(self.a) + self.growth * (k_plus_1 - 1)
        try:
            return math.exp(log_cost)
        except OverflowError:
            # Every finite marginal benefit is below an unrepresentable cost.
            return math.inf

    def validate_positive(self, k_max: int) -> None:
        if int(k_max) < 0:
            raise ValueError(f"k_max must be >= 0, got {k_max}")


MarginalCostSchedule = LinearMarginalCostSchedule | ExponentialMarginalCostSchedule
CostInput = MarginalCostSchedule | tuple[float, float] | float


def as_cost_schedule(value: CostInput) -> MarginalCostSchedule:
    """Accept schedules as well as the historical scalar/(a, b) cost hooks."""
    if isinstance(value, (LinearMarginalCostSchedule, ExponentialMarginalCostSchedule)):
        return value
    if isinstance(value, (int, float, np.number)):
        return LinearMarginalCostSchedule(float(value))
    a, b = value
    return LinearMarginalCostSchedule(float(a), float(b))


def compile_cost_schedule(
    config: Mapping[str, Any],
    *,
    parameter_symbols: Sequence[sympy.Symbol],
    parse_expr: Callable,
    where: str,
) -> Callable[[np.ndarray], MarginalCostSchedule]:
    """Compile parameter-only YAML costs, validating type-specific fields."""
    if not isinstance(config, Mapping) or "a" not in config:
        raise ValueError(f"{where}.a is required.")
    kind = config.get("type", "linear")
    if kind not in {"linear", "exponential"}:
        raise ValueError(f"{where}.type must be 'linear' or 'exponential', got {kind!r}")
    slope = "growth" if kind == "exponential" else "b"
    unknown = set(config) - {"type", "a", slope}
    if unknown:
        raise ValueError(f"Unsupported field(s) in {where} for {kind} costs: {sorted(unknown)}")
    if kind == "exponential" and slope not in config:
        raise ValueError(f"{where}.growth is required for exponential costs.")
    expressions = [
        parse_expr(config.get(field, 0.0), where=f"{where}.{field}")
        for field in ("a", slope)
    ]
    evaluate = sympy.lambdify(parameter_symbols, expressions, modules="numpy")
    schedule_type = ExponentialMarginalCostSchedule if kind == "exponential" else LinearMarginalCostSchedule

    def schedule(params):
        values = evaluate(*np.asarray(params, dtype=float).reshape(-1).tolist())
        return schedule_type(*(float(value) for value in values))

    return schedule
