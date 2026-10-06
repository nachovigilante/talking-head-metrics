"""
BAS (Beat Alignment Score) for head motion.

Gaussian-kernel score from AI Choreographer (Li et al., ICCV 2021): each motion
beat contributes exp(-d^2 / (2 sigma^2)), with d the distance to the nearest
reference beat, and the score is the mean over motion beats. Head-motion beats
are peaks of the SO(3) angular velocity, acceleration and jerk.
"""

from metrics.bas.bas_calculator import BASCalculator, beat_alignment_score
from metrics.bas.so3_beat_detector import SO3BeatDetector

__all__ = ["BASCalculator", "SO3BeatDetector", "beat_alignment_score"]
