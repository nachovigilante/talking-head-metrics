"""Angular velocity, acceleration and jerk of a rotation sequence, computed in SO(3)."""

import numpy as np
import scipy.spatial.transform as spt
from scipy import signal


def rotation_matrix_to_axis_angle(R):
    """
    Convert rotation matrix to axis-angle representation (Log map).

    Args:
        R: 3x3 rotation matrix

    Returns:
        3D axis-angle vector
    """
    return spt.Rotation.from_matrix(R).as_rotvec()


def enforce_quaternion_continuity(quats):
    """
    Flip quaternion signs so that consecutive ones have a non-negative dot product.

    q and -q are the same rotation; choosing the sign closest to the previous
    quaternion keeps the sequence continuous before it is filtered.
    """
    continuous_quats = quats.copy()
    for i in range(1, len(quats)):
        if np.dot(continuous_quats[i], continuous_quats[i - 1]) < 0:
            continuous_quats[i] *= -1
    return continuous_quats


def compute_body_angular_velocity(rotations, dt=1.0):
    """
    Body-frame angular velocity of a rotation sequence: omega_t = Log(R_t^T R_{t+1}) / dt.

    The body-frame velocity is left-invariant: it does not change if every R_t is
    premultiplied by the same rotation. Sample t lies between frames t and t + 1.

    Args:
        rotations: Array of rotation matrices (T, 3, 3)
        dt: Time step between frames

    Returns:
        omega: Angular velocity vectors (T-1, 3) in rad/time_unit
    """
    omega = []
    for t in range(len(rotations) - 1):
        delta_R = rotations[t].T @ rotations[t + 1]
        omega.append(rotation_matrix_to_axis_angle(delta_R) / dt)

    return np.array(omega)


def compute_derivatives(signal_array, dt=1.0):
    """
    First and second finite differences of a (T, N) signal.

    Applied to the angular velocity, they give the angular acceleration (T-1, N)
    and the angular jerk (T-2, N).
    """
    alpha = np.diff(signal_array, axis=0) / dt
    jerk = np.diff(alpha, axis=0) / dt
    return alpha, jerk


def apply_savgol_filter(data, window_length=9, polyorder=3):
    """
    Savitzky-Golay filter applied to each column of a (T, N) array.

    The window shrinks to the largest odd length that fits short sequences; if it
    becomes shorter than polyorder + 1, the data is returned unfiltered.
    """
    if len(data) < window_length:
        window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
        if window_length < polyorder + 1:
            return data

    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = signal.savgol_filter(data[:, i], window_length, polyorder)
    return filtered


def convert_pose_to_so3_derivatives(
    head_pose, dt=1.0, apply_filtering=True, filter_window=9, filter_polyorder=3
):
    """
    Angular velocity, acceleration and jerk of an axis-angle rotation sequence.

    The rotations can be smoothed first: they are converted to sign-continuous
    quaternions, filtered component-wise with Savitzky-Golay, and renormalized.

    Args:
        head_pose: Axis-angle rotations (T, 3), in radians.
        dt: Time step between frames.
        apply_filtering: Whether to smooth the rotations.
        filter_window: Window length of the filter.
        filter_polyorder: Polynomial order of the filter.

    Returns:
        dict with 'omega' (T-1, 3), 'alpha' (T-2, 3), 'jerk' (T-3, 3) and the
        (possibly smoothed) 'rotations' (T, 3, 3).
    """
    rotations = np.array([spt.Rotation.from_rotvec(r).as_matrix() for r in head_pose])

    if apply_filtering:
        quats = enforce_quaternion_continuity(spt.Rotation.from_matrix(rotations).as_quat())
        quats = apply_savgol_filter(quats, window_length=filter_window, polyorder=filter_polyorder)
        rotations = np.array([spt.Rotation.from_quat(q).as_matrix() for q in quats])

    omega = compute_body_angular_velocity(rotations, dt)
    alpha, jerk = compute_derivatives(omega, dt)

    return {"omega": omega, "alpha": alpha, "jerk": jerk, "rotations": rotations}
