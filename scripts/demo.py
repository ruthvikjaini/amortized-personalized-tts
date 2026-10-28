"""Short demo: discover a controller for one profile, with and without reuse.

Usage:  python scripts/demo.py
"""

from __future__ import annotations

from persontts.controllers import evaluate_controller, summarise
from persontts.discovery import discover
from persontts.experience_bank import ExperienceBank, HistoryEntry
from persontts.guide import distill_guide
from persontts.problems import generate_problems, split_problems
from persontts.profiles import UserProfile
from persontts.trajectories import build_bank

SOURCE_SPECS = [
    (0.70, 500.0, 0.0015),
    (0.75, 400.0, 0.0010),
    (0.80, 350.0, 0.0008),
    (0.85, 500.0, 0.0025),
    (0.70, 300.0, 0.0005),
    (0.80, 700.0, 0.0035),
]


def main() -> None:
    problems = generate_problems(n_problems=60, seed=0)
    discovery_problems, heldout_problems = split_problems(problems, seed=0)
    bank = build_bank(problems, n_trajectories=32, seed=0)

    experience = ExperienceBank()
    for i, (acc, lat, cost) in enumerate(SOURCE_SPECS):
        profile = UserProfile(name=f"source-{i}", accuracy_floor=acc, latency_ceiling_ms=lat, cost_ceiling=cost)
        found = discover(profile, discovery_problems, bank, variant="from_scratch", budget=20, n_seeds=4, seed=0)
        experience.add(HistoryEntry(profile=profile, best_controller=found.best_controller, joint_satisfaction_rate=found.joint_satisfaction_rate))
    guide = distill_guide(experience)

    target = UserProfile(name="target-demo", accuracy_floor=0.80, latency_ceiling_ms=380.0, cost_ceiling=0.0012)
    for variant in ("from_scratch", "full"):
        found = discover(
            target, discovery_problems, bank, variant=variant,
            bank_experience=experience, guide=guide, budget=20, n_seeds=4, seed=0,
        )
        heldout = summarise(
            evaluate_controller(found.best_controller, heldout_problems, bank, n_seeds=8, seed=0),
            target,
        )
        print(
            f"{variant}: {found.best_controller.label()} "
            f"candidates={found.n_candidates} "
            f"held-out JSR={heldout['joint_satisfaction_rate']:.3f} "
            f"accuracy={heldout['accuracy']:.3f} latency={heldout['latency_ms']:.0f}ms "
            f"cost=${heldout['cost']:.5f}"
        )
    print("Guide rules:")
    for rule in guide.rules:
        print(f"  - {rule}")


if __name__ == "__main__":
    main()
