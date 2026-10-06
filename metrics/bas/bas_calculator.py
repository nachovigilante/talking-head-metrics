"""
BAS (Beat Alignment Score) for head motion.

Gaussian-kernel score from AI Choreographer (Li et al., ICCV 2021). Here it
compares the head-motion beats of a prediction against those of the ground
truth, the use DiffPoseTalk gives it for talking heads; PBAS (metrics/pbas)
keeps the kernel and uses prosodic events of the speech as the reference.
"""

from typing import Dict

import numpy as np
import torch

from utils.flame_utils import load_flame_params
from metrics.bas.so3_beat_detector import SO3BeatDetector

BEAT_TYPES = ["omega", "alpha", "jerk", "direction_change"]


def beat_alignment_score(pred_beats, ref_beats, sigma=3.0):
    """
    Beat Alignment Score (Li et al., ICCV 2021).

    Each predicted beat is scored with a Gaussian kernel on its distance to the
    nearest reference beat, and the scores are averaged over predicted beats:

        BAS = (1/n) * sum_i exp( -min_j |t_i - t_j|^2 / (2 * sigma^2) )

    Args:
        pred_beats: Predicted beat times, in frames.
        ref_beats: Reference beat times, in frames.
        sigma: Gaussian kernel width in frames (default=3, ~120ms at 25fps).

    Returns:
        float: BAS in [0, 1]; 0 if either set of beats is empty.
    """
    pred_beats = np.asarray(pred_beats, dtype=float)
    ref_beats = np.asarray(ref_beats, dtype=float)

    if len(pred_beats) == 0 or len(ref_beats) == 0:
        return 0.0

    min_dists = np.abs(pred_beats[:, None] - ref_beats[None, :]).min(axis=1)
    scores = np.exp(-(min_dists**2) / (2 * sigma**2))
    return float(scores.mean())


class BASCalculator:
    """BAS between the SO(3) head-motion beats of a prediction and of the ground truth."""

    def __init__(
        self,
        device: torch.device | None = None,
        sigma: float = 3.0,
        dt: float = 1.0,
        apply_filtering: bool = True,
        filter_window: int = 9,
        filter_polyorder: int = 3,
    ):
        """
        Args:
            device: Device for FLAME parameter loading ('cpu', 'cuda', or None for auto).
            sigma: Gaussian kernel width in frames (default=3 for 25fps).
            dt: Time step between frames for the SO(3) derivatives.
            apply_filtering: Whether to smooth the rotations (Savitzky-Golay on quaternions).
            filter_window: Window length of the filter.
            filter_polyorder: Polynomial order of the filter.
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.sigma = sigma
        self.beat_detector = SO3BeatDetector(
            dt=dt,
            apply_filtering=apply_filtering,
            filter_window=filter_window,
            filter_polyorder=filter_polyorder,
        )

    def _load_head_rotation(self, file_path: str, format: str = "th1kh"):
        """Head rotation (T, 3), axis-angle radians, from a FLAME parameter file."""
        _, _, pose, _, _ = load_flame_params(str(file_path), self.device, format=format)
        return pose[:, :3].cpu().numpy()

    def calculate_bas(
        self,
        pred_beats: np.ndarray,
        ref_beats: np.ndarray,
        sigma: float | None = None,
    ) -> float:
        """BAS between two sets of beat times (uses the instance sigma if None)."""
        s = sigma if sigma is not None else self.sigma
        return beat_alignment_score(pred_beats, ref_beats, s)

    def calculate_bas_from_files(
        self,
        reference_file: str,
        prediction_file: str,
        reference_format: str = "th1kh",
        prediction_format: str = "ensemble",
    ) -> Dict[str, float]:
        """
        BAS for every beat type between two FLAME parameter files.

        Args:
            reference_file: Ground-truth motion.
            prediction_file: Predicted motion.
            reference_format: load_flame_params format of the reference.
            prediction_format: load_flame_params format of the prediction.

        Returns:
            Dict mapping beat type to BAS, e.g.
            {'omega': 0.35, 'alpha': 0.42, 'jerk': 0.38, 'direction_change': 0.81}.
        """
        ref = self.beat_detector.detect_all_beats_so3(
            self._load_head_rotation(reference_file, reference_format)
        )
        pred = self.beat_detector.detect_all_beats_so3(
            self._load_head_rotation(prediction_file, prediction_format)
        )
        return {
            bt: self.calculate_bas(pred[bt]["beats"], ref[bt]["beats"]) for bt in BEAT_TYPES
        }
