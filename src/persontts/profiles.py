"""User requirement profiles and the joint satisfaction rate.

A profile states three requirements jointly, following the formulation in
Wang et al. (2026): an accuracy floor and ceilings on latency and cost.
A controller satisfies a profile on one evaluation seed only when all
three hold at once. Optimising any single Pareto frontier (accuracy-cost
or accuracy-latency) can therefore miss the controllers that satisfy a
profile, which is the point of the joint objective.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserProfile:
    name: str
    accuracy_floor: float  # e.g. 0.80
    latency_ceiling_ms: float  # mean latency per problem, milliseconds
    cost_ceiling: float  # mean cost per problem, dollars (modelled rates)

    def features(self) -> tuple[float, float, float]:
        """Normalised features for nearest-profile retrieval."""
        return (
            self.accuracy_floor,
            self.latency_ceiling_ms / 1000.0,
            self.cost_ceiling * 100.0,
        )


def joint_satisfaction_rate(
    seed_metrics: list[dict[str, float]], profile: UserProfile
) -> float:
    """Fraction of seeds meeting accuracy, latency and cost jointly.

    ``seed_metrics`` holds one ``{"accuracy", "latency_ms", "cost"}`` dict
    per evaluation seed. A seed counts only if all three requirements
    hold simultaneously.
    """
    if not seed_metrics:
        return 0.0
    satisfied = sum(
        1
        for m in seed_metrics
        if m["accuracy"] >= profile.accuracy_floor
        and m["latency_ms"] <= profile.latency_ceiling_ms
        and m["cost"] <= profile.cost_ceiling
    )
    return satisfied / len(seed_metrics)


def pareto_satisfaction_rate(
    seed_metrics: list[dict[str, float]],
    profile: UserProfile,
    dimensions: tuple[str, ...],
) -> float:
    """Satisfaction restricted to a subset of dimensions.

    Used to show what a single-frontier (accuracy-cost or
    accuracy-latency) evaluation would have concluded about the same
    controller. ``dimensions`` is a subset of
    ``("accuracy", "latency_ms", "cost")``.
    """
    if not seed_metrics:
        return 0.0
    checks = {
        "accuracy": lambda m: m["accuracy"] >= profile.accuracy_floor,
        "latency_ms": lambda m: m["latency_ms"] <= profile.latency_ceiling_ms,
        "cost": lambda m: m["cost"] <= profile.cost_ceiling,
    }
    satisfied = sum(1 for m in seed_metrics if all(checks[d](m) for d in dimensions))
    return satisfied / len(seed_metrics)
