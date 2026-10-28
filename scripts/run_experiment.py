"""Run the full experiment: source discovery, Guide distillation, target discovery.

Pipeline
--------
1. Generate the problem bank and split it into discovery / held-out sets.
2. Build the trajectory bank once.
3. Discover a controller from scratch for each source profile; store the
   histories in the experience bank.
4. Distil the frozen Guide from the source histories.
5. For each unseen target profile, run every discovery variant under
   the same candidate budget, re-evaluate the winner on held-out
   problems, and compare with the fixed default, the single-frontier
   winners, and the exhaustive oracle.
6. Write results/experiment.json, the Guide, the experience bank, and
   figures. All numbers printed by this script come from these runs;
   nothing is hard-coded.

Usage:  python scripts/run_experiment.py [--seed 0] [--fast]
``--fast`` halves the problem counts and budgets for a quick smoke run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from persontts.controllers import controller_space, evaluate_controller, summarise
from persontts.discovery import discover, exhaustive_oracle
from persontts.evaluation import evaluate_fixed_default, single_frontier_report
from persontts.experience_bank import ExperienceBank, HistoryEntry
from persontts.guide import distill_guide
from persontts.plotting import plot_accuracy_cost, plot_candidate_budget, plot_variant_jsr
from persontts.problems import generate_problems, split_problems
from persontts.profiles import UserProfile
from persontts.trajectories import build_bank

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def make_profiles(prefix: str, specs: list[tuple[float, float, float]]) -> list[UserProfile]:
    return [
        UserProfile(
            name=f"{prefix}-{i}",
            accuracy_floor=acc,
            latency_ceiling_ms=lat,
            cost_ceiling=cost,
        )
        for i, (acc, lat, cost) in enumerate(specs)
    ]


SOURCE_SPECS = [
    (0.65, 600.0, 0.0020),
    (0.70, 350.0, 0.0008),
    (0.70, 500.0, 0.0015),
    (0.75, 400.0, 0.0010),
    (0.75, 700.0, 0.0030),
    (0.80, 350.0, 0.0008),
    (0.80, 500.0, 0.0020),
    (0.85, 500.0, 0.0025),
    (0.70, 300.0, 0.0005),
    (0.75, 300.0, 0.0008),
    (0.80, 700.0, 0.0035),
    (0.85, 700.0, 0.0040),
    (0.65, 400.0, 0.0004),
    (0.70, 700.0, 0.0025),
    (0.75, 350.0, 0.0015),
    (0.80, 350.0, 0.0012),
    (0.85, 400.0, 0.0015),
    (0.65, 800.0, 0.0040),
    (0.70, 300.0, 0.0012),
    (0.75, 600.0, 0.0008),
    (0.80, 450.0, 0.0008),
    (0.85, 800.0, 0.0040),
    (0.70, 450.0, 0.0006),
    (0.75, 450.0, 0.0020),
]

TARGET_SPECS = [
    (0.75, 380.0, 0.0010),
    (0.80, 420.0, 0.0015),
    (0.70, 320.0, 0.0006),
    (0.85, 600.0, 0.0030),
    (0.65, 350.0, 0.0005),
    (0.80, 330.0, 0.0009),
    (0.75, 650.0, 0.0018),
    (0.85, 450.0, 0.0020),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    n_problems = 60 if args.fast else 120
    budget = 20 if args.fast else 30

    problems = generate_problems(n_problems=n_problems, seed=args.seed)
    discovery_problems, heldout_problems = split_problems(problems, seed=args.seed)
    bank = build_bank(problems, n_trajectories=32, seed=args.seed)

    source_profiles = make_profiles("source", SOURCE_SPECS)
    target_profiles = make_profiles("target", TARGET_SPECS)

    # 1. Source discovery (from scratch) -> experience bank.
    experience = ExperienceBank()
    for profile in source_profiles:
        result = discover(
            profile, discovery_problems, bank, variant="from_scratch",
            budget=budget, n_seeds=4, seed=args.seed,
        )
        experience.add(
            HistoryEntry(
                profile=profile,
                best_controller=result.best_controller,
                joint_satisfaction_rate=result.joint_satisfaction_rate,
                candidates=result.history,
            )
        )

    # 2. Frozen Guide.
    guide = distill_guide(experience)

    # 3. Target discovery: every variant, same budget, held-out evaluation.
    variants = ["from_scratch", "warm_start", "guide_only", "full"]
    target_results: list[dict] = []
    for profile in target_profiles:
        entry: dict = {"profile": profile.name, "variants": {}}
        for variant in variants:
            found = discover(
                profile, discovery_problems, bank, variant=variant,
                bank_experience=experience, guide=guide,
                budget=budget, n_seeds=4, seed=args.seed,
            )
            heldout = summarise(
                evaluate_controller(
                    found.best_controller, heldout_problems, bank, n_seeds=8, seed=args.seed
                ),
                profile,
            )
            entry["variants"][variant] = {
                **found.to_dict(),
                "heldout_joint_satisfaction_rate": heldout["joint_satisfaction_rate"],
                "heldout_accuracy": heldout["accuracy"],
                "heldout_latency_ms": heldout["latency_ms"],
                "heldout_cost": heldout["cost"],
            }
        entry["fixed_default"] = evaluate_fixed_default(
            profile, heldout_problems, bank, seed=args.seed
        )
        entry["oracle"] = exhaustive_oracle(
            profile, heldout_problems, bank, seed=args.seed
        ).to_dict()
        entry["single_frontier"] = single_frontier_report(
            profile, discovery_problems, heldout_problems, bank, seed=args.seed
        )
        target_results.append(entry)

    # 4. Summaries.
    summary: dict = {}
    for variant in variants + ["fixed_default", "oracle"]:
        if variant in ("fixed_default", "oracle"):
            jsrs = [
                e[variant]["joint_satisfaction_rate"] if variant == "fixed_default"
                else e[variant]["joint_satisfaction_rate"]
                for e in target_results
            ]
            summary[variant] = {
                "mean_jsr": float(np.mean(jsrs)),
                "mean_candidates": None,
            }
        else:
            jsrs = [e["variants"][variant]["heldout_joint_satisfaction_rate"] for e in target_results]
            cands = [e["variants"][variant]["n_candidates"] for e in target_results]
            to_best = [e["variants"][variant]["candidates_to_best"] for e in target_results]
            summary[variant] = {
                "mean_jsr": float(np.mean(jsrs)),
                "mean_candidates": float(np.mean(cands)),
                "mean_candidates_to_best": float(np.mean(to_best)),
            }
    frontier_means = {
        name: float(np.mean([e["single_frontier"][name]["joint_satisfaction_rate"] for e in target_results]))
        for name in ("accuracy_cost", "accuracy_latency")
    }

    # 5. Controller-space figure data (held-out, first target profile).
    space_summary = []
    first_target = target_profiles[0]
    for controller in controller_space():
        result = evaluate_controller(controller, heldout_problems, bank, n_seeds=4, seed=args.seed)
        space_summary.append(
            {"controller": controller.label(), **{k: result[k] for k in ("accuracy", "latency_ms", "cost")}}
        )

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "seed": args.seed,
        "n_problems": n_problems,
        "n_discovery_problems": len(discovery_problems),
        "n_heldout_problems": len(heldout_problems),
        "budget_candidates": budget,
        "runtime_seconds": float(time.perf_counter() - started),
        "guide": guide.to_dict(),
        "summary": summary,
        "single_frontier_mean_jsr": frontier_means,
        "targets": target_results,
    }
    (RESULTS / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    experience.save(str(RESULTS / "experience_bank.json"))
    guide.save(str(RESULTS / "guide.json"))

    plot_variant_jsr(
        {k: v for k, v in summary.items() if k in variants + ["fixed_default"]},
        str(RESULTS / "jsr_by_variant.png"),
    )
    plot_candidate_budget(
        {
            k: {"mean_candidates": v["mean_candidates_to_best"]}
            for k, v in summary.items()
            if k in variants
        },
        str(RESULTS / "candidates_by_variant.png"),
    )
    plot_accuracy_cost(space_summary, str(RESULTS / "accuracy_cost_space.png"))

    print(f"problems: {n_problems} ({len(discovery_problems)} discovery / {len(heldout_problems)} held out)")
    print(f"runtime: {payload['runtime_seconds']:.1f}s")
    for variant, values in summary.items():
        print(f"{variant}: mean held-out JSR {values['mean_jsr']:.3f}  mean candidates {values['mean_candidates']}")
    print(f"single-frontier means: {frontier_means}")
    print("Guide rules:")
    for rule in guide.rules:
        print(f"  - {rule}")


if __name__ == "__main__":
    main()
