"""End-to-end reproducibility: the same seed reproduces the same discovery outcome."""

from persontts.discovery import discover
from persontts.experience_bank import ExperienceBank, HistoryEntry
from persontts.guide import distill_guide
from persontts.problems import generate_problems, split_problems
from persontts.profiles import UserProfile
from persontts.trajectories import build_bank


def _run(seed: int):
    problems = generate_problems(n_problems=36, seed=seed)
    discovery, _heldout = split_problems(problems, seed=seed)
    bank = build_bank(problems, n_trajectories=16, seed=seed)
    experience = ExperienceBank()
    for i, (acc, lat, cost) in enumerate([(0.75, 300.0, 0.002), (0.85, 200.0, 0.001)]):
        profile = UserProfile(f"source-{i}", acc, lat, cost)
        found = discover(profile, discovery, bank, variant="from_scratch", budget=10, n_seeds=2, seed=seed)
        experience.add(HistoryEntry(profile, found.best_controller, found.joint_satisfaction_rate))
    guide = distill_guide(experience)
    target = UserProfile("target", 0.80, 250.0, 0.0015)
    found = discover(
        target, discovery, bank, variant="full",
        bank_experience=experience, guide=guide, budget=10, n_seeds=2, seed=seed,
    )
    return found.best_controller.label(), found.joint_satisfaction_rate


def test_end_to_end_reproducible():
    assert _run(4) == _run(4)
