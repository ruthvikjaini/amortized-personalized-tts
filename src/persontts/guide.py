"""The frozen Guide: procedural guidance distilled from source histories.

PersonTTS distils a frozen Guide from source discovery histories and
uses it to inform proposals for new profiles, instead of re-running a
discovery agent from scratch. Here the Guide is a small, inspectable
set of rules plus controller priors, distilled from the experience
bank:

* ``early_stop_prior`` -- how often the best controller for
  latency-tight profiles used early stopping, versus other profiles.
* ``tier_prior`` -- how often each tier won, split by accuracy floor.
* ``width_prior`` -- the median winning width, split by cost ceiling.
* ``rules`` -- plain-language statements with the counts behind them.

The Guide is frozen once distilled: discovery for a target profile may
read it but never updates it mid-search, so its effect can be measured
cleanly against the no-guide variants.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from .controllers import Controller
from .experience_bank import ExperienceBank
from .profiles import UserProfile


@dataclass
class Guide:
    early_stop_rate_tight_latency: float
    early_stop_rate_other: float
    tier_share_high_accuracy: dict[str, float]
    tier_share_other: dict[str, float]
    median_width_tight_cost: float
    median_width_other: float
    rules: list[str]

    def recommend(self, profile: UserProfile) -> Controller:
        """A starting controller built from the profile and the Guide alone.

        No source controller is copied: the recommendation is assembled
        from the distilled priors and the target profile's own ceilings.
        """
        tight_latency = profile.latency_ceiling_ms <= 350.0
        tight_cost = profile.cost_ceiling <= 0.0010
        high_accuracy = profile.accuracy_floor >= 0.80
        tier_pool = (
            self.tier_share_high_accuracy if high_accuracy else self.tier_share_other
        )
        tier = max(tier_pool, key=tier_pool.get)
        width = 4 if not tight_cost else 2
        if self.median_width_tight_cost and tight_cost:
            width = int(max(1, min(8, round(self.median_width_tight_cost))))
        passes = 2 if high_accuracy else 1
        early_stop = (
            self.early_stop_rate_tight_latency >= 0.5
            if tight_latency
            else self.early_stop_rate_other >= 0.5
        )
        return Controller(tier=tier, width=width, passes=passes, early_stop=bool(early_stop))

    def neighbour_priority(self, profile: UserProfile) -> list[str]:
        """Which knob to try first, given the profile's tightest constraint.

        Discovery orders its neighbour proposals by this list, so the
        Guide changes search order (and therefore what is found within
        a fixed candidate budget) without changing the search space.
        """
        if profile.latency_ceiling_ms <= 350.0:
            return ["early_stop", "width", "tier", "passes"]
        if profile.cost_ceiling <= 0.0010:
            return ["width", "tier", "early_stop", "passes"]
        if profile.accuracy_floor >= 0.80:
            return ["tier", "passes", "width", "early_stop"]
        return ["width", "passes", "tier", "early_stop"]

    def to_dict(self) -> dict:
        return {
            "early_stop_rate_tight_latency": self.early_stop_rate_tight_latency,
            "early_stop_rate_other": self.early_stop_rate_other,
            "tier_share_high_accuracy": self.tier_share_high_accuracy,
            "tier_share_other": self.tier_share_other,
            "median_width_tight_cost": self.median_width_tight_cost,
            "median_width_other": self.median_width_other,
            "rules": self.rules,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)


def _rate(entries, predicate) -> float:
    if not entries:
        return 0.0
    return float(np.mean([1.0 if predicate(e) else 0.0 for e in entries]))


def _tier_share(entries) -> dict[str, float]:
    counts = {"small": 0, "medium": 0, "large": 0}
    for entry in entries:
        counts[entry.best_controller.tier] += 1
    total = max(1, sum(counts.values()))
    return {tier: count / total for tier, count in counts.items()}


def distill_guide(bank: ExperienceBank) -> Guide:
    """Distil the Guide from completed source histories."""
    entries = bank.entries
    tight_latency = [e for e in entries if e.profile.latency_ceiling_ms <= 350.0]
    other_latency = [e for e in entries if e.profile.latency_ceiling_ms > 250.0]
    high_accuracy = [e for e in entries if e.profile.accuracy_floor >= 0.80]
    other_accuracy = [e for e in entries if e.profile.accuracy_floor < 0.80]
    tight_cost = [e for e in entries if e.profile.cost_ceiling <= 0.0010]
    other_cost = [e for e in entries if e.profile.cost_ceiling > 0.0015]

    def median_width(group) -> float:
        if not group:
            return 4.0
        return float(np.median([e.best_controller.width for e in group]))

    rules = [
        (
            f"Latency-tight profiles (<=350ms) used early stopping in "
            f"{_rate(tight_latency, lambda e: e.best_controller.early_stop):.0%} of "
            f"winning controllers, vs {_rate(other_latency, lambda e: e.best_controller.early_stop):.0%} otherwise."
        ),
        (
            f"High-accuracy profiles (>=0.80) favoured tier "
            f"{max(_tier_share(high_accuracy), key=_tier_share(high_accuracy).get)}; "
            f"other profiles favoured tier "
            f"{max(_tier_share(other_accuracy), key=_tier_share(other_accuracy).get)}."
        ),
        (
            f"Median winning width was {median_width(tight_cost):.0f} under a tight "
            f"cost ceiling vs {median_width(other_cost):.0f} otherwise."
        ),
    ]
    return Guide(
        early_stop_rate_tight_latency=_rate(
            tight_latency, lambda e: e.best_controller.early_stop
        ),
        early_stop_rate_other=_rate(other_latency, lambda e: e.best_controller.early_stop),
        tier_share_high_accuracy=_tier_share(high_accuracy),
        tier_share_other=_tier_share(other_accuracy),
        median_width_tight_cost=median_width(tight_cost),
        median_width_other=median_width(other_cost),
        rules=rules,
    )
