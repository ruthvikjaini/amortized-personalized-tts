import numpy as np

from persontts.controllers import Controller, controller_space, evaluate_controller, execute_controller
from persontts.problems import generate_problems, split_problems
from persontts.profiles import UserProfile, joint_satisfaction_rate
from persontts.trajectories import build_bank


def _setup(n_problems=24, seed=3):
    problems = generate_problems(n_problems=n_problems, seed=seed)
    discovery, heldout = split_problems(problems, seed=seed)
    bank = build_bank(problems, n_trajectories=16, seed=seed)
    return problems, discovery, heldout, bank


def test_problems_deterministic_and_stratified():
    first = generate_problems(n_problems=30, seed=7)
    second = generate_problems(n_problems=30, seed=7)
    assert [p.answer for p in first] == [p.answer for p in second]
    discovery, heldout = split_problems(first, seed=7)
    assert len(discovery) + len(heldout) == 30
    assert {p.problem_id for p in discovery}.isdisjoint({p.problem_id for p in heldout})
    # Every difficulty appears on both sides when the bank is large enough.
    assert {p.difficulty for p in discovery} == {p.difficulty for p in heldout}


def test_bank_deterministic():
    problems = generate_problems(n_problems=12, seed=5)
    bank_a = build_bank(problems, n_trajectories=8, seed=5)
    bank_b = build_bank(problems, n_trajectories=8, seed=5)
    for problem in problems:
        for tier in ("small", "medium", "large"):
            assert [t.answer for t in bank_a.pool(problem.problem_id, tier)] == [
                t.answer for t in bank_b.pool(problem.problem_id, tier)
            ]


def test_larger_tier_is_more_accurate_on_average():
    problems = generate_problems(n_problems=40, seed=1)
    bank = build_bank(problems, n_trajectories=32, seed=1)
    rates = {}
    for tier in ("small", "medium", "large"):
        correct = np.mean(
            [t.correct for p in problems for t in bank.pool(p.problem_id, tier)]
        )
        rates[tier] = correct
    assert rates["large"] > rates["small"]
    assert rates["medium"] > rates["small"]


def test_joint_satisfaction_requires_all_three():
    profile = UserProfile("p", accuracy_floor=0.8, latency_ceiling_ms=250.0, cost_ceiling=0.001)
    seeds = [
        {"accuracy": 0.9, "latency_ms": 200.0, "cost": 0.0005},
        {"accuracy": 0.9, "latency_ms": 300.0, "cost": 0.0005},  # latency fails
        {"accuracy": 0.7, "latency_ms": 200.0, "cost": 0.0005},  # accuracy fails
        {"accuracy": 0.9, "latency_ms": 200.0, "cost": 0.0020},  # cost fails
    ]
    assert joint_satisfaction_rate(seeds, profile) == 0.25
    assert joint_satisfaction_rate([], profile) == 0.0


def test_controller_space_size_and_labels():
    space = controller_space()
    assert len(space) == 72
    assert len({c.key() for c in space}) == 72
    assert Controller().label() == "medium/w4/p1"


def test_early_stop_saves_tokens():
    _, _, _, bank = _setup()
    problem = generate_problems(n_problems=1, seed=3)[0]
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    plain = execute_controller(Controller("medium", 8, 1, False), problem, bank, rng_a)
    stopped = execute_controller(Controller("medium", 8, 1, True), problem, bank, rng_b)
    assert stopped["tokens"] <= plain["tokens"]


def test_evaluate_controller_is_deterministic():
    _, discovery, _, bank = _setup()
    controller = Controller("large", 4, 2, True)
    first = evaluate_controller(controller, discovery, bank, n_seeds=4, seed=9)
    second = evaluate_controller(controller, discovery, bank, n_seeds=4, seed=9)
    assert first == second
    assert 0.0 <= first["accuracy"] <= 1.0
    assert first["latency_ms"] > 0 and first["cost"] > 0
