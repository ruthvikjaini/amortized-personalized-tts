"""Deterministic multi-step reasoning problems with known ground truth.

Three families, all with integer answers so correctness is unambiguous:

* ``chain``   -- a chain of additions, subtractions and multiplications.
* ``modular`` -- ``(a * b + c) % m``.
* ``nested``  -- a chain whose intermediate result feeds a second chain.

Difficulty is the number of reasoning steps. Problems are generated from a
seeded RNG, so a given seed always produces the same bank, and the
discovery / held-out split is stratified by difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    problem_id: int
    family: str
    difficulty: int  # number of reasoning steps, 2..5
    prompt: str
    answer: int
    steps: tuple[int, ...]  # intermediate values after each step


def _chain(rng: np.random.Generator, problem_id: int) -> Problem:
    difficulty = int(rng.integers(2, 6))
    value = int(rng.integers(3, 20))
    steps = []
    expr = f"{value}"
    for _ in range(difficulty):
        op = rng.choice(["+", "-", "*"])
        operand = int(rng.integers(2, 13))
        if op == "+":
            value = value + operand
        elif op == "-":
            value = value - operand
        else:
            value = value * operand
        steps.append(value)
        expr = f"({expr} {op} {operand})"
    return Problem(
        problem_id=problem_id,
        family="chain",
        difficulty=difficulty,
        prompt=f"Compute {expr}.",
        answer=value,
        steps=tuple(steps),
    )


def _modular(rng: np.random.Generator, problem_id: int) -> Problem:
    difficulty = int(rng.integers(2, 5))
    a, b, c = (int(rng.integers(5, 60)), int(rng.integers(3, 20)), int(rng.integers(1, 40)))
    m = int(rng.integers(7, 30))
    steps = [a * b, a * b + c]
    value = (a * b + c) % m
    if difficulty >= 3:
        steps.append(value)
        value = (value * int(rng.integers(2, 6)) + int(rng.integers(1, 9))) % m
        steps.append(value)
    if difficulty >= 4:
        steps.append(value)
        value = (value + int(rng.integers(3, 15))) % m
        steps.append(value)
    return Problem(
        problem_id=problem_id,
        family="modular",
        difficulty=difficulty,
        prompt=f"Compute ({a} * {b} + {c}) mod {m}, then apply the remaining steps in order.",
        answer=value,
        steps=tuple(steps),
    )


def _nested(rng: np.random.Generator, problem_id: int) -> Problem:
    difficulty = int(rng.integers(3, 6))
    x = int(rng.integers(2, 12))
    steps = []
    value = x * int(rng.integers(2, 6)) + int(rng.integers(1, 10))
    steps.append(value)
    for _ in range(difficulty - 1):
        if rng.random() < 0.5:
            value = value + int(rng.integers(2, 12))
        else:
            value = value * 2 - int(rng.integers(1, 6))
        steps.append(value)
    return Problem(
        problem_id=problem_id,
        family="nested",
        difficulty=difficulty,
        prompt=f"Start with {x}. Double-and-shift through {difficulty} steps as specified by the trace.",
        answer=value,
        steps=tuple(steps),
    )


_GENERATORS = {"chain": _chain, "modular": _modular, "nested": _nested}


def generate_problems(n_problems: int = 120, seed: int = 0) -> list[Problem]:
    """Generate a deterministic problem bank, cycling through families."""
    rng = np.random.default_rng(seed)
    families = ["chain", "modular", "nested"]
    problems = []
    for i in range(n_problems):
        problems.append(_GENERATORS[families[i % len(families)]](rng, i))
    return problems


def split_problems(
    problems: list[Problem], heldout_fraction: float = 0.5, seed: int = 0
) -> tuple[list[Problem], list[Problem]]:
    """Stratified split by difficulty into discovery and held-out sets."""
    rng = np.random.default_rng(seed)
    discovery, heldout = [], []
    by_difficulty: dict[int, list[Problem]] = {}
    for problem in problems:
        by_difficulty.setdefault(problem.difficulty, []).append(problem)
    for difficulty in sorted(by_difficulty):
        group = by_difficulty[difficulty][:]
        rng.shuffle(group)
        cut = int(round(len(group) * (1.0 - heldout_fraction)))
        discovery.extend(group[:cut])
        heldout.extend(group[cut:])
    discovery.sort(key=lambda p: p.problem_id)
    heldout.sort(key=lambda p: p.problem_id)
    return discovery, heldout
