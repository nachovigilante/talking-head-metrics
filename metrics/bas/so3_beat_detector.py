import numpy as np
import scipy.signal as signal

from utils.so3_utils import convert_pose_to_so3_derivatives

# Position, in video frames, of sample 0 of each derivative. omega[i] = Log(R_i^T R_{i+1})
# lies between frames i and i+1, and each further finite difference adds half a frame.
FRAME_OFFSET = {"omega": 0.5, "alpha": 1.0, "jerk": 1.5}


def on_frame_grid(values, offset, n_frames):
    """Resample a derivative sampled at frames offset, offset + 1, ... onto frames 0..n_frames-1."""
    if len(values) == 0:
        return np.zeros(n_frames)
    return np.interp(np.arange(n_frames), np.arange(len(values)) + offset, values)


def detect_magnitude_peaks(vectors, threshold_percentile, min_distance):
    """
    Peaks of the norm of a (T, 3) signal above a percentile of that norm.

    Returns:
        Tuple of (peak indices, magnitude (T,), threshold).
    """
    magnitude = np.linalg.norm(vectors, axis=1)
    threshold = np.percentile(magnitude, threshold_percentile)
    peaks, _ = signal.find_peaks(magnitude, height=threshold, distance=min_distance)
    return peaks, magnitude, threshold


def detect_direction_changes(omega, angle_threshold=0.1, min_distance=10):
    """
    Samples where the axis of the angular velocity turns by more than a threshold.

    At sample i the directions of omega[i - 1] and omega[i + 1] are compared; samples
    closer than min_distance to the previous accepted one are skipped.

    Args:
        omega: Body-frame angular velocity (T, 3).
        angle_threshold: Minimum angle between the two directions, in radians.
        min_distance: Minimum separation between accepted samples.

    Returns:
        Tuple of (sample indices, angles in radians).
    """
    candidates, angles = [], []
    for i in range(2, len(omega) - 2):
        before, after = omega[i - 1], omega[i + 1]
        norm_before, norm_after = np.linalg.norm(before), np.linalg.norm(after)
        if norm_before > 1e-6 and norm_after > 1e-6:
            cos_angle = np.clip(np.dot(before / norm_before, after / norm_after), -1, 1)
            angle = np.arccos(cos_angle)
            if angle > angle_threshold:
                candidates.append(i)
                angles.append(angle)

    kept, kept_angles, last = [], [], -min_distance
    for i, angle in zip(candidates, angles):
        if i - last >= min_distance:
            kept.append(i)
            kept_angles.append(angle)
            last = i
    return np.array(kept), np.array(kept_angles)


class SO3BeatDetector:
    """
    Head-motion beats from the derivatives of the head rotation in SO(3).

    The rotations are differentiated in the Lie group (body-frame angular
    velocity, then finite differences for acceleration and jerk), so the result
    does not depend on the rotation parametrization and has no Euler-angle
    singularities or wrap-around jumps.
    """

    def __init__(self, dt=1.0, apply_filtering=True, filter_window=9, filter_polyorder=3):
        """
        Args:
            dt: Time step between frames.
            apply_filtering: Whether to smooth the rotations (Savitzky-Golay on quaternions).
            filter_window: Window length of the filter.
            filter_polyorder: Polynomial order of the filter.
        """
        self.dt = dt
        self.apply_filtering = apply_filtering
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder

    def detect_all_beats_so3(
        self,
        head_pose_axis_angle,
        omega_threshold=75,
        alpha_threshold=75,
        jerk_threshold=80,
        direction_threshold=0.1,
        min_distance=10,
    ):
        """
        Beats of the SO(3) angular velocity, acceleration and jerk.

        For each derivative, 'beats' are the times of the detected peaks in video
        frames (frame k is at time k / fps). They are not integers: a peak at
        sample i of omega, alpha or jerk is at frame i + FRAME_OFFSET[type].
        'peak_indices' keeps the sample indices into 'magnitude', and 'signal' is
        the magnitude resampled onto the video frames.

        Args:
            head_pose_axis_angle: Head rotation (T, 3), axis-angle in radians.
            omega_threshold: Percentile of ||omega|| used as peak threshold.
            alpha_threshold: Percentile of ||alpha|| used as peak threshold.
            jerk_threshold: Percentile of ||jerk|| used as peak threshold.
            direction_threshold: Angle threshold for direction changes, in radians.
            min_distance: Minimum separation between beats, in samples.

        Returns:
            Dict with one entry per beat type ('omega', 'alpha', 'jerk',
            'direction_change'), plus 'so3_data' with the derivatives and rotations.
        """
        so3_data = convert_pose_to_so3_derivatives(
            head_pose_axis_angle,
            dt=self.dt,
            apply_filtering=self.apply_filtering,
            filter_window=self.filter_window,
            filter_polyorder=self.filter_polyorder,
        )
        n_frames = len(head_pose_axis_angle)

        results = {}
        for name, percentile in [
            ("omega", omega_threshold),
            ("alpha", alpha_threshold),
            ("jerk", jerk_threshold),
        ]:
            peaks, magnitude, threshold = detect_magnitude_peaks(
                so3_data[name], percentile, min_distance
            )
            offset = FRAME_OFFSET[name]
            results[name] = {
                "beats": peaks + offset,
                "peak_indices": peaks,
                "frame_offset": offset,
                "magnitude": magnitude,
                "signal": on_frame_grid(magnitude, offset, n_frames),
                "threshold": threshold,
                "data": so3_data[name],
            }

        # A direction change at omega sample i compares omega[i - 1] and omega[i + 1],
        # so it sits at the time of omega[i].
        direction_beats, direction_angles = detect_direction_changes(
            so3_data["omega"], direction_threshold, min_distance
        )
        results["direction_change"] = {
            "beats": direction_beats + FRAME_OFFSET["omega"],
            "peak_indices": direction_beats,
            "magnitudes": direction_angles,
        }
        results["so3_data"] = so3_data
        return results
