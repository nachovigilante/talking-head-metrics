"""
Prosodic events of a speech recording, the reference of PBAS.

- pitch_accent: peaks of the F0 contour (Praat, through parselmouth)
- onset: peaks of the onset strength, which marks syllable and consonant onsets (librosa)
- energy: peaks of the RMS energy (librosa)
- prominence: peaks of a weighted sum of F0, intensity and RMS
"""

import numpy as np
from scipy.signal import find_peaks
from typing import Dict

import librosa
import parselmouth


class ProsodicExtractor:
    """
    Extracts prosodic events from speech audio files.

    All signals are sampled on the video frame grid: index k is at time k / fps,
    the convention of librosa's centered frames and of the motion beats. Praat
    frames start at a different time, so F0 and intensity are interpolated onto
    that grid, and pitch accents are reported at their Praat times converted to
    frames (not necessarily integers).
    """

    def __init__(
        self,
        fps: int = 25,
        sr: int = 16000,
        min_distance_sec: float = 0.2,
        f0_floor: float = 75.0,
        f0_ceiling: float = 600.0,
        onset_percentile: float = 70.0,
        energy_percentile: float = 70.0,
        f0_prominence_factor: float = 0.5,
        prominence_weights: tuple = (0.4, 0.3, 0.3),
    ):
        """
        Args:
            fps: Video frame rate (features are sampled at this rate).
            sr: Audio sample rate.
            min_distance_sec: Minimum distance between peaks in seconds.
            f0_floor: F0 analysis floor frequency (Hz).
            f0_ceiling: F0 analysis ceiling frequency (Hz).
            onset_percentile: Percentile threshold for onset peak detection.
            energy_percentile: Percentile threshold for energy peak detection.
            f0_prominence_factor: F0 peak prominence as factor of F0 std.
            prominence_weights: (f0_weight, intensity_weight, rms_weight) for
                combined prominence. Must sum to 1.
        """
        self.fps = fps
        self.sr = sr
        self.hop = sr // fps
        self.min_dist_frames = max(1, int(min_distance_sec * fps))
        self.f0_floor = f0_floor
        self.f0_ceiling = f0_ceiling
        self.onset_percentile = onset_percentile
        self.energy_percentile = energy_percentile
        self.f0_prominence_factor = f0_prominence_factor
        self.prominence_weights = prominence_weights

    def _frame_grid(self, snd):
        """Times (s) of the video frames covering a parselmouth Sound."""
        n_frames = 1 + int(snd.get_total_duration() * self.fps + 1e-6)
        return np.arange(n_frames) / self.fps

    def extract_pitch_accents(self, wav_file):
        """
        Extract pitch accent locations (local F0 peaks on stressed syllables).

        Peaks are detected on the Praat frames and converted to video frames.

        Args:
            wav_file: Path to WAV file.

        Returns:
            Tuple of (peak_frames, raw_f0, interpolated_f0). The two F0 contours
            are on the video frame grid; raw_f0 is NaN on unvoiced frames.
        """
        snd = parselmouth.Sound(str(wav_file))
        pitch = snd.to_pitch_ac(
            time_step=1.0 / self.fps,
            pitch_floor=self.f0_floor,
            pitch_ceiling=self.f0_ceiling,
        )
        times = pitch.xs()
        grid = self._frame_grid(snd)

        f0 = pitch.selected_array["frequency"]
        f0[f0 == 0] = np.nan

        voiced = ~np.isnan(f0)
        if voiced.sum() < 3:
            return np.array([]), np.full(len(grid), np.nan), np.zeros(len(grid))

        f0_interp = np.interp(
            np.arange(len(f0)), np.where(voiced)[0], f0[voiced]
        )

        peaks, _ = find_peaks(
            f0_interp,
            distance=self.min_dist_frames,
            prominence=np.nanstd(f0[voiced]) * self.f0_prominence_factor,
        )

        raw_on_grid = np.interp(grid, times, np.nan_to_num(f0))
        raw_on_grid[np.interp(grid, times, voiced.astype(float)) < 0.5] = np.nan

        return times[peaks] * self.fps, raw_on_grid, np.interp(grid, times, f0_interp)

    def extract_onset_beats(self, wav_file):
        """
        Extract onset strength peaks (syllable onsets, spectral transients).

        Args:
            wav_file: Path to WAV file.

        Returns:
            Tuple of (peak_frames, onset_envelope).
        """
        y, _ = librosa.load(str(wav_file), sr=self.sr)

        onset_env = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop,
            aggregate=np.median, fmax=8000,
        )

        threshold = np.percentile(onset_env, self.onset_percentile)
        peaks, _ = find_peaks(
            onset_env, distance=self.min_dist_frames, height=threshold
        )

        return peaks, onset_env

    def extract_energy_peaks(self, wav_file):
        """
        Extract energy (RMS) peaks.

        Args:
            wav_file: Path to WAV file.

        Returns:
            Tuple of (peak_frames, rms_envelope).
        """
        y, _ = librosa.load(str(wav_file), sr=self.sr)

        rms = librosa.feature.rms(
            y=y, frame_length=self.hop * 2, hop_length=self.hop
        ).squeeze()

        threshold = np.percentile(rms, self.energy_percentile)
        peaks, _ = find_peaks(
            rms, distance=self.min_dist_frames, height=threshold
        )

        return peaks, rms

    def extract_combined_prominence(self, wav_file):
        """
        Extract combined prosodic prominence peaks (F0 + intensity + RMS).

        Args:
            wav_file: Path to WAV file.

        Returns:
            Tuple of (peak_frames, prominence_signal).
        """
        _, _, f0_interp = self.extract_pitch_accents(wav_file)

        snd = parselmouth.Sound(str(wav_file))
        intensity = snd.to_intensity(
            time_step=1.0 / self.fps, minimum_pitch=self.f0_floor
        )
        int_values = np.interp(
            self._frame_grid(snd), intensity.xs(), intensity.values.T.squeeze()
        )

        y, _ = librosa.load(str(wav_file), sr=self.sr)
        rms = librosa.feature.rms(
            y=y, frame_length=self.hop * 2, hop_length=self.hop
        ).squeeze()

        n = min(len(f0_interp), len(int_values), len(rms))
        f0_interp = f0_interp[:n]
        int_values = int_values[:n]
        rms = rms[:n]

        def normalize(x):
            x = x.copy().astype(float)
            x[np.isnan(x)] = 0
            mn, mx = x.min(), x.max()
            return (x - mn) / (mx - mn) if mx > mn else np.zeros_like(x)

        w_f0, w_int, w_rms = self.prominence_weights
        prominence = (
            w_f0 * normalize(f0_interp)
            + w_int * normalize(int_values)
            + w_rms * normalize(rms)
        )

        peaks, _ = find_peaks(
            prominence, distance=self.min_dist_frames, prominence=0.1
        )

        return peaks, prominence

    def extract_all(self, wav_file) -> Dict:
        """
        Extract all prosodic event types from an audio file.

        Args:
            wav_file: Path to WAV file.

        Returns:
            Dict with keys 'pitch_accent', 'onset', 'energy', 'prominence',
            each mapping to {'beats': np.ndarray, 'signal': np.ndarray}.
        """
        pa_peaks, f0, f0_interp = self.extract_pitch_accents(wav_file)
        on_peaks, onset_env = self.extract_onset_beats(wav_file)
        en_peaks, rms = self.extract_energy_peaks(wav_file)
        pr_peaks, prom = self.extract_combined_prominence(wav_file)

        return {
            "pitch_accent": {"beats": pa_peaks, "signal": f0_interp, "raw_f0": f0},
            "onset": {"beats": on_peaks, "signal": onset_env},
            "energy": {"beats": en_peaks, "signal": rms},
            "prominence": {"beats": pr_peaks, "signal": prom},
        }
