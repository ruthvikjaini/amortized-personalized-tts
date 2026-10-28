"""Pre-sampled reasoning trajectories, mirroring the paper's evaluation design.

PersonTTS evaluates controllers over pre-sampled reasoning trajectories
(128 per problem-model pair in the paper) instead of re-running a model
for every candidate controller. This module does the same at CPU scale:
for each (problem, model tier) pair it samples a fixed bank of
trajectories once, and every controller is evaluated by replaying that
bank. Replay makes controller comparisons deterministic and cheap, and
it is what lets discovery be measured in candidate evaluations rather
than in model calls.

Each trajectory simulates step-by-step reasoning: at every step the
simulated solver may slip, with a per-step error probability that grows
with problem difficulty and shrinks with model tier. A slipped step
perturbs the running value, so later steps inherit the error -- the
same error-accumulation structure that makes width, passes and
verification matter. Tokens, latency and cost are modelled from the
trajectory length and tier rates; rates are stated in
``TIER_RATES`` and are modelled, not vendor quotes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .problems import Problem

# Modelled rates: dollars per million tokens and milliseconds per token.
# These are simulation parameters, chosen so the tiers trade accuracy
# against latency and cost in the intended order. They are not prices
# quoted by any provider.
TIER_RATES = {
    "small": {"err_base": 0.075, "cost_per_mtok": 0.20, "ms_per_token": 0.55, "tokens_per_step": 28},
    "medium": {"err_base": 0.045, "cost_per_mtok": 0.80, "ms_per_token": 0.85, "tokens_per_step": 36},
    "large": {"err_base": 0.022, "cost_per_mtok": 6.00, "ms_per_token": 1.25, "tokens_per_step": 45},
}

TIERS = tuple(TIER_RATES)


@dataclass(frozen=True)
class Trajectory:
    answer: int
    tokens: int
    n_steps: int
    correct: bool


class TrajectoryBank:
    """Fixed bank of trajectories, keyed by (problem_id, tier)."""

    def __init__(self, trajectories: dict[tuple[int, str], list[Trajectory]]):
        self._trajectories = trajectories

    def pool(self, problem_id: int, tier: str) -> list[Trajectory]:
        return self._trajectories[(problem_id, tier)]

    def pool_size(self, problem_id: int, tier: str) -> int:
        return len(self._trajectories[(problem_id, tier)])


def _simulate_trajectory(
    rng: np.random.Generator, problem: Problem, tier: str
) -> Trajectory:
    rates = TIER_RATES[tier]
    value = None
    tokens = 0
    for step_index, true_step in enumerate(problem.steps):
        tokens += rates["tokens_per_step"] + int(rng.integers(0, 5))
        # Per-step slip probability grows with difficulty, shrinks by tier.
        p_err = min(0.30, rates["err_base"] + 0.018 * (problem.difficulty - 1))
        if rng.random() < p_err:
            # Slip: perturb the running value; later steps build on it
            # by replaying the true step-to-step deltas, so the error
            # persists to the final answer.
            slip = int(rng.integers(-6, 7))
            if slip == 0:
                slip = 3
            base = true_step + slip
            deltas = [
                problem.steps[i] - problem.steps[i - 1]
                for i in range(1, len(problem.steps))
            ]
            running = base
            for delta in deltas[step_index:]:
                running = running + delta
            value = running
            break
        value = true_step
    if value is None:
        value = problem.answer
    return Trajectory(
        answer=int(value),
        tokens=int(tokens),
        n_steps=len(problem.steps),
        correct=bool(value == problem.answer),
    )


def build_bank(
    problems: list[Problem], n_trajectories: int = 32, seed: int = 0
) -> TrajectoryBank:
    """Sample the trajectory bank once, deterministically."""
    trajectories: dict[tuple[int, str], list[Trajectory]] = {}
    for problem in problems:
        for tier in TIERS:
            # Stable per-pair seed (not Python's salted hash).
            rng = np.random.default_rng(
                seed * 1_000_003 + problem.problem_id * 10_007 + sum(map(ord, tier)) * 131
            )
            trajectories[(problem.problem_id, tier)] = [
                _simulate_trajectory(rng, problem, tier) for _ in range(n_trajectories)
            ]
    return TrajectoryBank(trajectories)
