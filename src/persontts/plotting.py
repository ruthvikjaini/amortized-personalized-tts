"""Figures for the experiment report."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_variant_jsr(results: dict, path: str) -> None:
    """Mean held-out joint satisfaction rate by discovery variant."""
    variants = list(results)
    means = [results[v]["mean_jsr"] for v in variants]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(variants, means, color="#4a7fb5")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Mean joint satisfaction rate (held out)")
    ax.set_title("Per-profile discovery beats a fixed controller")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_candidate_budget(results: dict, path: str) -> None:
    """Mean candidate evaluations until the best controller was first found."""
    variants = list(results)
    means = [results[v]["mean_candidates"] for v in variants]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(variants, means, color="#6a9e6f")
    ax.set_ylabel("Mean evaluations to first find the best controller")
    ax.set_title("How early each variant finds its best controller (budget: 30)")
    for bar, value in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, f"{value:.1f}", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_accuracy_cost(problems_summary: list[dict], path: str) -> None:
    """Accuracy vs cost for every controller, coloured by latency feasibility."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for item in problems_summary:
        ax.scatter(
            item["cost"],
            item["accuracy"],
            s=28,
            c="#c0392b" if item["latency_ms"] > 350 else "#4a7fb5",
            alpha=0.7,
        )
    ax.set_xlabel("Mean cost per problem ($, modelled rates)")
    ax.set_ylabel("Mean accuracy")
    ax.set_title("Controller space: accuracy vs cost (red = latency over 350ms)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
