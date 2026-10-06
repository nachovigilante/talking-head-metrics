"""
PBAS (Prosodic Beat Alignment Score).

Same kernel as BAS, with prosodic events of the driving speech (pitch accents,
onset strength, RMS energy, combined prominence) as the reference instead of the
ground-truth motion beats. It measures whether head motion is synchronized with
the speech it accompanies.
"""

from metrics.pbas.pbas_calculator import PBASCalculator, optimal_lag
from metrics.pbas.prosodic_extractor import ProsodicExtractor

__all__ = ["PBASCalculator", "ProsodicExtractor", "optimal_lag"]
