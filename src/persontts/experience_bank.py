"""The experience bank: source profiles and their discovery histories.

Discovery for a source profile is run once and stored: the profile, the
best controller found, its joint satisfaction rate, and the sequence of
candidates tried. A new target profile retrieves the source profile
with the most similar requirements (Euclidean distance over normalised
profile features) and re-evaluates that source's controller under the
target profile before searching further -- requirement-matched
controller initialisation, as in PersonTTS. Re-evaluation matters: a
controller is never trusted on the source profile's numbers alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .controllers import Controller
from .profiles import UserProfile


@dataclass
class HistoryEntry:
    profile: UserProfile
    best_controller: Controller
    joint_satisfaction_rate: float
    candidates: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "profile": {
                "name": self.profile.name,
                "accuracy_floor": self.profile.accuracy_floor,
                "latency_ceiling_ms": self.profile.latency_ceiling_ms,
                "cost_ceiling": self.profile.cost_ceiling,
            },
            "best_controller": self.best_controller.key(),
            "best_controller_label": self.best_controller.label(),
            "joint_satisfaction_rate": self.joint_satisfaction_rate,
            "n_candidates": len(self.candidates),
        }


class ExperienceBank:
    def __init__(self, entries: list[HistoryEntry] | None = None):
        self.entries: list[HistoryEntry] = list(entries or [])

    def add(self, entry: HistoryEntry) -> None:
        self.entries.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve(self, target: UserProfile) -> HistoryEntry | None:
        """Nearest source profile by normalised requirement distance."""
        if not self.entries:
            return None
        target_features = np.array(target.features(), dtype=float)
        best, best_distance = None, float("inf")
        for entry in self.entries:
            distance = float(
                np.linalg.norm(np.array(entry.profile.features()) - target_features)
            )
            if distance < best_distance:
                best, best_distance = entry, distance
        return best

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump([e.to_dict() for e in self.entries], handle, indent=2)

    @classmethod
    def load(cls, path: str) -> "ExperienceBank":
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        entries = []
        for item in raw:
            profile = UserProfile(**item["profile"])
            tier, width, passes, early_stop = item["best_controller"]
            entries.append(
                HistoryEntry(
                    profile=profile,
                    best_controller=Controller(
                        tier=tier, width=width, passes=passes, early_stop=early_stop
                    ),
                    joint_satisfaction_rate=item["joint_satisfaction_rate"],
                )
            )
        return cls(entries)
