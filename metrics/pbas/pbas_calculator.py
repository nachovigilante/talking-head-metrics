"""
PBAS (Prosodic Beat Alignment Score).

Scores how close the head-motion beats fall to prosodic events of the driving
speech, with the Gaussian kernel of BAS (Li et al., ICCV 2021). Where BAS
compares a prediction with the ground-truth motion, PBAS compares the motion,
predicted or real, with the audio.
"""

import numpy as np
import torch
from typing import Dict

from utils.flame_utils import load_flame_params
from metrics.bas.so3_beat_detector import SO3BeatDetector
from metrics.bas.bas_calculator import beat_alignment_score
from metrics.pbas.prosodic_extractor import ProsodicExtractor


PROSODIC_TYPES = ["pitch_accent", "onset", "energy", "prominence"]
MOTION_TYPES = ["omega", "alpha", "jerk"]


def optimal_lag(motion_signal, speech_signal, max_lag_frames=12):
    """
    Lag that maximizes the absolute normalized cross-correlation between a
    continuous motion signal and a continuous prosodic signal.

    A positive lag means the motion follows the speech; a negative lag means it
    precedes it. Both signals are truncated to the shorter length.

    Args:
        motion_signal: Motion magnitude per frame (e.g. ||omega(t)||).
        speech_signal: Prosodic signal per frame (e.g. onset strength).
        max_lag_frames: Search window in frames (default=12, ±480ms at 25fps).

    Returns:
        int: Optimal lag in frames (0 if either signal is constant).
    """
    n = min(len(motion_signal), len(speech_signal))
    x = np.asarray(motion_signal[:n], dtype=float)
    y = np.asarray(speech_signal[:n], dtype=float)
    x = x - x.mean()
    y = y - y.mean()
    sx, sy = x.std(), y.std()
    if n < 2 or sx < 1e-10 or sy < 1e-10:
        return 0
    x, y = x / sx, y / sy

    max_lag_frames = min(max_lag_frames, n - 1)
    lags = np.arange(-max_lag_frames, max_lag_frames + 1)
    corr = [
        np.mean(x[lag:] * y[: n - lag]) if lag >= 0 else np.mean(x[: n + lag] * y[-lag:])
        for lag in lags
    ]
    return int(lags[np.argmax(np.abs(corr))])


class PBASCalculator:
    """
    PBAS between SO(3) head-motion beats and prosodic events of the speech.

    Motion beats: peaks of the angular velocity ('omega'), acceleration
    ('alpha') and jerk ('jerk'). Prosodic events: 'pitch_accent', 'onset',
    'energy' and 'prominence' (see ProsodicExtractor). The 3 x 4 combinations
    are the 12 variants of the metric.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        sigma: float = 3.0,
        fps: int = 25,
        sr: int = 16000,
        dt: float = 1.0,
        apply_filtering: bool = True,
        filter_window: int = 9,
        filter_polyorder: int = 3,
    ):
        """
        Args:
            device: Device for FLAME parameter loading.
            sigma: Gaussian kernel width in frames (default=3 for 25fps).
            fps: Video frame rate.
            sr: Audio sample rate.
            dt: Time step for SO(3) derivative computation.
            apply_filtering: Whether to filter quaternions before differentiation.
            filter_window: Savitzky-Golay filter window length.
            filter_polyorder: Savitzky-Golay polynomial order.
        """
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.sigma = sigma
        self.fps = fps

        self.beat_detector = SO3BeatDetector(
            dt=dt,
            apply_filtering=apply_filtering,
            filter_window=filter_window,
            filter_polyorder=filter_polyorder,
        )

        self.prosodic_extractor = ProsodicExtractor(fps=fps, sr=sr)

    def _load_head_rotation(self, file_path: str, format: str = "th1kh"):
        """Load head rotation (axis-angle radians) from FLAME parameters."""
        _, _, pose, _, _ = load_flame_params(
            str(file_path), self.device, format=format
        )
        return pose[:, :3].cpu().numpy()

    def _detect_motion_beats(self, head_rot_rad: np.ndarray) -> Dict:
        """Detect SO(3) motion beats."""
        return self.beat_detector.detect_all_beats_so3(head_rot_rad)

    def calculate_pbas(
        self,
        motion_beats: np.ndarray,
        speech_beats: np.ndarray,
        sigma: float | None = None,
    ) -> float:
        """
        Calculate PBAS between motion beats and speech prosodic events.

        Args:
            motion_beats: Motion beat times in video frames.
            speech_beats: Speech prosodic event times in video frames.
            sigma: Override for Gaussian kernel width.

        Returns:
            float: PBAS in [0, 1].
        """
        s = sigma if sigma is not None else self.sigma
        return beat_alignment_score(motion_beats, speech_beats, s)

    def calculate_pbas_lag_corrected(
        self,
        motion_beats: np.ndarray,
        speech_beats: np.ndarray,
        motion_signal: np.ndarray,
        speech_signal: np.ndarray,
        max_lag_frames: int = 12,
        sigma: float | None = None,
    ) -> tuple[float, int]:
        """
        PBAS after removing the per-clip temporal offset between motion and speech.

        The offset is the optimal lag between the continuous motion magnitude and
        the continuous prosodic signal (see `optimal_lag`). The motion beats are
        shifted by that lag before scoring, so the score reflects the relative
        timing of the beats and not their global phase.

        Args:
            motion_beats: Motion beat times in video frames.
            speech_beats: Speech prosodic event times in video frames.
            motion_signal: Motion magnitude on the video frame grid (the 'signal'
                entry of SO3BeatDetector results).
            speech_signal: Prosodic signal on the video frame grid.
            max_lag_frames: Lag search window in frames.
            sigma: Override for Gaussian kernel width.

        Returns:
            Tuple of (lag-corrected PBAS, lag in frames).
        """
        lag = optimal_lag(motion_signal, speech_signal, max_lag_frames)
        shifted = np.asarray(motion_beats, dtype=float) - lag
        return self.calculate_pbas(shifted, speech_beats, sigma), lag

    def calculate_pbas_from_files(
        self,
        motion_file: str,
        wav_file: str,
        motion_format: str = "ensemble",
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate PBAS for all motion x prosodic type combinations.

        Args:
            motion_file: Path to FLAME parameter file (prediction or GT).
            wav_file: Path to speech audio WAV file.
            motion_format: Format of the motion file.

        Returns:
            Nested dict: {motion_type: {prosodic_type: score}}, e.g.:
            {'omega': {'pitch_accent': 0.35, 'onset': 0.72, ...}, ...}
        """
        rot = self._load_head_rotation(motion_file, motion_format)
        motion_beats = self._detect_motion_beats(rot)
        prosodic = self.prosodic_extractor.extract_all(wav_file)

        results = {}
        for mt in MOTION_TYPES:
            results[mt] = {}
            mb = motion_beats[mt]["beats"]
            for pt in PROSODIC_TYPES:
                sb = prosodic[pt]["beats"]
                results[mt][pt] = self.calculate_pbas(mb, sb)

        return results
