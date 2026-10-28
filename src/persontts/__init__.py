"""Personalized test-time scaling with amortized controller discovery."""

from .controllers import Controller, controller_space
from .profiles import UserProfile
from .problems import Problem, generate_problems, split_problems
from .trajectories import TrajectoryBank, build_bank

__all__ = [
    "Controller",
    "controller_space",
    "UserProfile",
    "Problem",
    "generate_problems",
    "split_problems",
    "TrajectoryBank",
    "build_bank",
]

__version__ = "0.1.0"
