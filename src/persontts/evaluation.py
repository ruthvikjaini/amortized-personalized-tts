"""Baselines and single-frontier comparisons.

Two comparisons carry the paper's argument:

1. A fixed default controller, applied to every profile, versus a
   controller discovered per profile.
2. Controllers chosen by optimising a single Pareto frontier
   (accuracy-cost or accuracy-latency) versus controllers chosen by
   the joint satisfaction rate. A single-frontier winner can look
   strong on its own frontier while failing the profile's other
   requirement, which the joint rate catches.
"""

from __future__ import annotations

from .controllers import Controller, controller_space, evaluate_controller, summarise
from .discovery import DEFAULT_CONTROLLER
from .problems import Problem
from .profiles import UserProfile, pareto_satisfaction_rate
from .trajectories import TrajectoryBank


def evaluate_fixed_default(
    profile: UserProfile,
    problems: list[Problem],
    bank: TrajectoryBank,
    n_seeds: int = 8,
    seed: int = 0,
) -> dict:
    return summarise(
        evaluate_controller(DEFAULT_CONTROLLER, problems, bank, n_seeds=n_seeds, seed=seed),
        profile,
    )


def select_single_frontier_controller(
    profile: UserProfile,
    problems: list[Problem],
    bank: TrajectoryBank,
    dimensions: tuple[str, ...],
    n_seeds: int = 4,
    seed: int = 0,
) -> Controller:
    """Pick the controller that maximises satisfaction on one frontier.

    Among controllers that satisfy the chosen dimensions on every seed
    (or the most seeds), prefer the highest accuracy; ties go to the
    cheaper controller. The winner is then judged on the held-out set
    by the full joint rate, where its ignored dimension can fail it.
    """
    best, best_key = None, None
    for controller in controller_space():
        result = evaluate_controller(controller, problems, bank, n_seeds=n_seeds, seed=seed)
        rate = pareto_satisfaction_rate(result["seed_metrics"], profile, dimensions)
        key = (rate, result["accuracy"], -result["cost"])
        if best_key is None or key > best_key:
            best, best_key = controller, key
    return best


def single_frontier_report(
    profile: UserProfile,
    discovery_problems: list[Problem],
    heldout_problems: list[Problem],
    bank: TrajectoryBank,
    seed: int = 0,
) -> dict:
    """Held-out joint satisfaction of the single-frontier winners."""
    report = {}
    for name, dimensions in (
        ("accuracy_cost", ("accuracy", "cost")),
        ("accuracy_latency", ("accuracy", "latency_ms")),
    ):
        controller = select_single_frontier_controller(
            profile, discovery_problems, bank, dimensions, seed=seed
        )
        heldout = summarise(
            evaluate_controller(controller, heldout_problems, bank, n_seeds=8, seed=seed),
            profile,
        )
        report[name] = {
            "controller": controller.label(),
            "joint_satisfaction_rate": heldout["joint_satisfaction_rate"],
            "accuracy": heldout["accuracy"],
            "latency_ms": heldout["latency_ms"],
            "cost": heldout["cost"],
        }
    return report
