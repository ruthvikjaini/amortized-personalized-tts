"""Executable test-time-scaling controllers and their replay evaluation.

A controller coordinates model selection, reasoning width and depth,
and stopping -- the same decisions PersonTTS's executable controllers
make. Here:

* ``tier``       -- which simulated model tier answers.
* ``width``      -- samples drawn per pass (reasoning width).
* ``passes``     -- sequential passes over the problem (reasoning depth).
                    Each pass draws fresh trajectories; a later pass can
                    overturn an earlier answer only on a unanimous vote,
                    so extra depth helps only when the fresh evidence is
                    decisive.
* ``early_stop`` -- pruning/stopping: stop a pass as soon as the first
                    two samples agree, instead of consuming the full width.

Evaluation replays the pre-sampled trajectory bank. For each seed, a
seeded RNG draws trajectory indices without replacement, so different
seeds exercise different sampled answers while the bank stays fixed.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .problems import Problem
from .profiles import UserProfile, joint_satisfaction_rate
from .trajectories import TIER_RATES, TrajectoryBank


@dataclass(frozen=True)
class Controller:
    tier: str = "medium"
    width: int = 4
    passes: int = 1
    early_stop: bool = False

    def key(self) -> tuple:
        return (self.tier, self.width, self.passes, self.early_stop)

    def label(self) -> str:
        stop = "+stop" if self.early_stop else ""
        return f"{self.tier}/w{self.width}/p{self.passes}{stop}"


def controller_space() -> list[Controller]:
    """The full discrete search space: 72 controllers."""
    space = []
    for tier, width, passes, early_stop in itertools.product(
        ("small", "medium", "large"), (1, 2, 4, 8), (1, 2, 3), (False, True)
    ):
        space.append(Controller(tier=tier, width=width, passes=passes, early_stop=early_stop))
    return space


def _majority_vote(answers: list[int]) -> tuple[int, float]:
    values, counts = np.unique(answers, return_counts=True)
    winner = int(values[int(np.argmax(counts))])
    agreement = float(np.max(counts) / len(answers))
    return winner, agreement


def _run_pass(
    rng: np.random.Generator,
    pool: list,
    width: int,
    early_stop: bool,
    used: set[int],
) -> tuple[list[int], list[int], int]:
    """Draw one pass of trajectories; returns answers, indices, tokens."""
    answers, indices, tokens = [], [], 0
    available = [i for i in range(len(pool)) if i not in used]
    rng.shuffle(available)
    for idx in available[:width]:
        traj = pool[idx]
        answers.append(traj.answer)
        indices.append(idx)
        tokens += traj.tokens
        used.add(idx)
        if early_stop and len(answers) >= 2 and len(set(answers)) == 1:
            break
    return answers, indices, tokens


def execute_controller(
    controller: Controller,
    problem: Problem,
    bank: TrajectoryBank,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Execute one controller on one problem under one seed.

    Returns correctness, latency (ms) and cost for that problem.
    """
    pool = bank.pool(problem.problem_id, controller.tier)
    rates = TIER_RATES[controller.tier]
    used: set[int] = set()
    answer = None
    total_tokens = 0
    for pass_index in range(controller.passes):
        answers, _, tokens = _run_pass(
            rng, pool, controller.width, controller.early_stop, used
        )
        if not answers:
            break
        total_tokens += tokens
        vote, agreement = _majority_vote(answers)
        if pass_index == 0:
            answer = vote
        elif agreement >= 1.0:
            # A later pass overturns only on unanimous fresh evidence.
            answer = vote
        # A non-unanimous later pass is discarded: depth without
        # decisive evidence does not move the answer.
    if answer is None:
        answer = pool[0].answer
    latency_ms = total_tokens * rates["ms_per_token"] + 12.0 * controller.passes
    cost = total_tokens * rates["cost_per_mtok"] / 1_000_000.0
    return {
        "correct": 1.0 if answer == problem.answer else 0.0,
        "latency_ms": float(latency_ms),
        "cost": float(cost),
        "tokens": float(total_tokens),
    }


def evaluate_controller(
    controller: Controller,
    problems: list[Problem],
    bank: TrajectoryBank,
    n_seeds: int = 8,
    seed: int = 0,
) -> dict:
    """Evaluate a controller over problems and seeds.

    Returns per-seed aggregate metrics plus the joint satisfaction
    helper fields; callers compute the satisfaction rate against a
    profile with :func:`summarise`.
    """
    seed_metrics = []
    for seed_offset in range(n_seeds):
        rng = np.random.default_rng(seed * 10_000 + seed_offset)
        correct, latencies, costs = [], [], []
        for problem in problems:
            result = execute_controller(controller, problem, bank, rng)
            correct.append(result["correct"])
            latencies.append(result["latency_ms"])
            costs.append(result["cost"])
        seed_metrics.append(
            {
                "accuracy": float(np.mean(correct)),
                "latency_ms": float(np.mean(latencies)),
                "cost": float(np.mean(costs)),
            }
        )
    return {
        "controller": controller.label(),
        "seed_metrics": seed_metrics,
        "accuracy": float(np.mean([m["accuracy"] for m in seed_metrics])),
        "latency_ms": float(np.mean([m["latency_ms"] for m in seed_metrics])),
        "cost": float(np.mean([m["cost"] for m in seed_metrics])),
    }


def summarise(result: dict, profile: UserProfile) -> dict:
    """Attach the joint satisfaction rate for a profile to a result."""
    jsr = joint_satisfaction_rate(result["seed_metrics"], profile)
    return {**result, "joint_satisfaction_rate": float(jsr)}
