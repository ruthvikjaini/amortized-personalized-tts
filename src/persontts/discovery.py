"""Amortized discovery: hill-climbing search over controllers.

The discovery procedure proposes executable controllers, evaluates each
one under the target profile on the discovery problem set, and keeps
the controller with the highest joint satisfaction rate. The search is
deliberately simple -- deterministic hill-climbing with seeded random
restarts -- so that the comparison between variants isolates the effect
of reuse, not the cleverness of the optimiser:

* ``from_scratch`` -- starts from a fixed default controller.
* ``warm_start``   -- starts from the retrieved source controller,
  re-evaluated under the target profile first.
* ``guide_only``   -- starts from the Guide's recommendation, with
  Guide-ordered neighbours, and no retrieved controller.
* ``full``         -- warm start plus Guide-ordered neighbours.

Every variant is charged the same candidate-evaluation budget. A
candidate evaluation is one ``evaluate_controller`` call on the
discovery set; discovery time is also recorded. The search score is
the joint satisfaction rate, with a graded partial-satisfaction term
breaking ties (see ``_score``); the reported rate itself stays pure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .controllers import Controller, evaluate_controller, summarise
from .experience_bank import ExperienceBank
from .guide import Guide
from .problems import Problem
from .profiles import UserProfile
from .trajectories import TrajectoryBank

DEFAULT_CONTROLLER = Controller(tier="medium", width=4, passes=1, early_stop=False)

_KNOB_VALUES = {
    "tier": ["small", "medium", "large"],
    "width": [1, 2, 4, 8],
    "passes": [1, 2, 3],
    "early_stop": [False, True],
}


@dataclass
class DiscoveryResult:
    variant: str
    profile: UserProfile
    best_controller: Controller
    joint_satisfaction_rate: float
    accuracy: float
    latency_ms: float
    cost: float
    n_candidates: int
    discovery_seconds: float
    history: list[dict]
    candidates_to_best: int = 0

    def to_dict(self) -> dict:
        return {
            "variant": self.variant,
            "profile": self.profile.name,
            "best_controller": self.best_controller.label(),
            "joint_satisfaction_rate": self.joint_satisfaction_rate,
            "accuracy": self.accuracy,
            "latency_ms": self.latency_ms,
            "cost": self.cost,
            "n_candidates": self.n_candidates,
            "candidates_to_best": self.candidates_to_best,
            "discovery_seconds": self.discovery_seconds,
        }


def _neighbours(controller: Controller, knob_order: list[str]) -> list[Controller]:
    neighbours = []
    current = {
        "tier": controller.tier,
        "width": controller.width,
        "passes": controller.passes,
        "early_stop": controller.early_stop,
    }
    for knob in knob_order:
        for value in _KNOB_VALUES[knob]:
            if value == current[knob]:
                continue
            candidate = Controller(**{**current, knob: value})
            neighbours.append(candidate)
    return neighbours


def _score(result: dict, profile: UserProfile) -> tuple[float, float, float, float, float]:
    # The reported objective is the joint satisfaction rate, but it is
    # coarse (one step per seed), so a search that only rewards it has
    # no gradient when no candidate satisfies a profile on any seed
    # and degenerates to the cheapest controller. The search score
    # therefore adds graded terms after the rate: the mean fraction of
    # the three requirements met per seed, then mean accuracy, which
    # climbs toward the accuracy floor even while the rate is zero.
    # They only break ties between equal satisfaction rates; the
    # reported rate stays pure.
    partial = float(
        np.mean(
            [
                (
                    (1.0 if m["accuracy"] >= profile.accuracy_floor else 0.0)
                    + (1.0 if m["latency_ms"] <= profile.latency_ceiling_ms else 0.0)
                    + (1.0 if m["cost"] <= profile.cost_ceiling else 0.0)
                )
                / 3.0
                for m in result["seed_metrics"]
            ]
        )
    )
    return (
        result["joint_satisfaction_rate"],
        partial,
        float(np.mean([m["accuracy"] for m in result["seed_metrics"]])),
        -result["cost"],
        -result["latency_ms"],
    )


def discover(
    profile: UserProfile,
    problems: list[Problem],
    bank: TrajectoryBank,
    variant: str = "from_scratch",
    bank_experience: ExperienceBank | None = None,
    guide: Guide | None = None,
    budget: int = 30,
    n_seeds: int = 4,
    seed: int = 0,
) -> DiscoveryResult:
    """Run one discovery procedure under a fixed candidate budget."""
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    history: list[dict] = []
    evaluated: dict[tuple, dict] = {}

    def evaluate(controller: Controller) -> dict:
        key = controller.key()
        if key in evaluated:
            return evaluated[key]
        result = summarise(
            evaluate_controller(controller, problems, bank, n_seeds=n_seeds, seed=seed),
            profile,
        )
        evaluated[key] = result
        history.append(
            {"controller": controller.label(), "joint_satisfaction_rate": result["joint_satisfaction_rate"]}
        )
        return result

    knob_order = ["tier", "width", "passes", "early_stop"]
    if variant == "from_scratch":
        current = DEFAULT_CONTROLLER
    elif variant == "warm_start":
        retrieved = bank_experience.retrieve(profile) if bank_experience else None
        current = retrieved.best_controller if retrieved else DEFAULT_CONTROLLER
    elif variant == "guide_only":
        current = guide.recommend(profile) if guide else DEFAULT_CONTROLLER
        if guide:
            knob_order = guide.neighbour_priority(profile)
    elif variant == "full":
        retrieved = bank_experience.retrieve(profile) if bank_experience else None
        current = retrieved.best_controller if retrieved else DEFAULT_CONTROLLER
        if guide:
            knob_order = guide.neighbour_priority(profile)
    else:
        raise ValueError(f"unknown variant: {variant}")

    best = evaluate(current)
    best_controller = current

    # First-improvement hill-climbing. Single-knob moves can strand the
    # search at a local optimum (the improving controller may differ in
    # two knobs at once), so a local optimum triggers a seeded random
    # restart, and restarts continue until the budget is spent. The
    # best controller found anywhere in the run is retained.
    while len(evaluated) < budget:
        improved = False
        for neighbour in _neighbours(best_controller, knob_order):
            if len(evaluated) >= budget:
                break
            result = evaluate(neighbour)
            if _score(result, profile) > _score(best, profile):
                best, best_controller = result, neighbour
                improved = True
                break  # First-improvement: restart the neighbourhood.
        if improved:
            continue
        if len(evaluated) >= budget:
            break
        random_controller = None
        for _ in range(20):
            candidate = Controller(
                tier=str(rng.choice(["small", "medium", "large"])),
                width=int(rng.choice([1, 2, 4, 8])),
                passes=int(rng.choice([1, 2, 3])),
                early_stop=bool(rng.choice([False, True])),
            )
            if candidate.key() not in evaluated:
                random_controller = candidate
                break
        if random_controller is None:
            break
        result = evaluate(random_controller)
        if _score(result, profile) > _score(best, profile):
            best, best_controller = result, random_controller
        # Otherwise keep restarting: hill-climbing resumes from the
        # best controller, whose neighbourhood is already evaluated
        # (cached, so it costs no budget), and the next restart fires.

    # Efficiency is measured by when the eventual best controller was
    # first evaluated: all variants spend the full budget, but a
    # variant that starts closer to a satisfying controller finds its
    # best controller earlier in the run.
    candidates_to_best = len(history)
    for index, entry in enumerate(history, start=1):
        if entry["controller"] == best_controller.label():
            candidates_to_best = index
            break

    return DiscoveryResult(
        variant=variant,
        profile=profile,
        best_controller=best_controller,
        joint_satisfaction_rate=float(best["joint_satisfaction_rate"]),
        accuracy=float(best["accuracy"]),
        latency_ms=float(best["latency_ms"]),
        cost=float(best["cost"]),
        n_candidates=len(evaluated),
        discovery_seconds=float(time.perf_counter() - started),
        history=history,
        candidates_to_best=candidates_to_best,
    )


def exhaustive_oracle(
    profile: UserProfile,
    problems: list[Problem],
    bank: TrajectoryBank,
    n_seeds: int = 8,
    seed: int = 0,
) -> DiscoveryResult:
    """Best controller in the full space, by exhaustive evaluation.

    This is an upper bound for reporting regret, not a discovery method:
    it costs one evaluation per controller in the space.
    """
    from .controllers import controller_space

    started = time.perf_counter()
    best, best_controller = None, None
    for controller in controller_space():
        result = summarise(
            evaluate_controller(controller, problems, bank, n_seeds=n_seeds, seed=seed),
            profile,
        )
        if best is None or _score(result, profile) > _score(best, profile):
            best, best_controller = result, controller
    return DiscoveryResult(
        variant="oracle",
        profile=profile,
        best_controller=best_controller,
        joint_satisfaction_rate=float(best["joint_satisfaction_rate"]),
        accuracy=float(best["accuracy"]),
        latency_ms=float(best["latency_ms"]),
        cost=float(best["cost"]),
        n_candidates=len(controller_space()),
        discovery_seconds=float(time.perf_counter() - started),
        history=[],
        candidates_to_best=len(controller_space()),
    )
