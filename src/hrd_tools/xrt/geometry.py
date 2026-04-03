"""Pure-numpy geometry for the multi-analyzer diffractometer arm.

All positions and orientations are computed using 4×4 homogeneous
(affine) transforms so that diffractometer positioning (roll, yaw,
displacement) can be cleanly composed with the ideal arm geometry.

Coordinate system (xrt convention):
    x – inboard / outboard
    y – beam direction (upstream → downstream)
    z – vertical (up)

In the ideal case every component sits in the y-z plane (x = 0).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from ..config import AnalyzerConfig, Diffractometer


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ComponentPlacement:
    """Position and orientation for one crystal / baffle / screen triplet."""

    # Crystal
    crystal_center: NDArray[np.floating]  # (3,)
    crystal_pitch: float  # rad

    # Baffle
    baffle_center: NDArray[np.floating]  # (3,)
    baffle_z: NDArray[np.floating]  # (3,) unit vector

    # Screen / detector
    screen_center: NDArray[np.floating]  # (3,)
    screen_z: NDArray[np.floating]  # (3,) unit vector


# ---------------------------------------------------------------------------
# Affine-transform helpers
# ---------------------------------------------------------------------------

def _translation_matrix(dx: float, dy: float, dz: float) -> NDArray:
    """4×4 homogeneous translation matrix."""
    T = np.eye(4)
    T[:3, 3] = [dx, dy, dz]
    return T


def _rotation_matrix_4x4(rot: Rotation) -> NDArray:
    """Embed a 3×3 rotation into a 4×4 homogeneous matrix."""
    M = np.eye(4)
    M[:3, :3] = rot.as_matrix()
    return M


def diffractometer_transform(
    diffractometer: Diffractometer,
) -> NDArray:
    """Build the 4×4 affine transform for diffractometer positioning.

    Order of operations (applied right-to-left to column vectors):
        1. Roll  – rotation about y-axis
        2. Yaw   – rotation about z-axis
        3. Translate by (dx, dy, dz)

    Parameters
    ----------
    diffractometer : Diffractometer
        Roll and yaw are in degrees, displacements in mm.

    Returns
    -------
    NDArray, shape (4, 4)
    """
    roll_rad = np.deg2rad(diffractometer.roll)
    yaw_rad = np.deg2rad(diffractometer.yaw)
    R_roll = Rotation.from_rotvec([0, roll_rad, 0])
    R_yaw = Rotation.from_rotvec([0, 0, yaw_rad])
    R = R_yaw * R_roll  # composed rotation
    T = _translation_matrix(diffractometer.dx, diffractometer.dy, diffractometer.dz)
    return T @ _rotation_matrix_4x4(R)


def _apply_affine_point(M: NDArray, pts: NDArray) -> NDArray:
    """Apply 4×4 affine *M* to one or more 3-D points.

    Parameters
    ----------
    M : (4, 4)
    pts : (3,) or (N, 3)

    Returns
    -------
    Transformed points, same shape as *pts*.
    """
    single = pts.ndim == 1
    if single:
        pts = pts[np.newaxis, :]
    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (M @ homo.T).T[:, :3]
    return out[0] if single else out


def _apply_affine_direction(M: NDArray, dirs: NDArray) -> NDArray:
    """Apply the rotation part of *M* to direction vectors (no translation).

    Parameters
    ----------
    M : (4, 4)
    dirs : (3,) or (N, 3)

    Returns
    -------
    Rotated direction vectors, same shape as *dirs*.
    """
    R3 = M[:3, :3]
    return (R3 @ dirs.T).T if dirs.ndim == 2 else R3 @ dirs


# ---------------------------------------------------------------------------
# Core geometry computation
# ---------------------------------------------------------------------------

def compute_arm_positions(
    arm_tth: float,
    analyzer: AnalyzerConfig,
    theta_b: float,
    diffractometer: Diffractometer | None = None,
) -> list[ComponentPlacement]:
    """Compute positions and orientations for every crystal on the arm.

    Parameters
    ----------
    arm_tth : float
        Two-theta angle of the arm in **radians**.
    analyzer : AnalyzerConfig
        Analyzer geometry parameters.
    theta_b : float
        Bragg angle in **radians**.
    diffractometer : Diffractometer, optional
        Diffractometer configuration.  ``None`` is treated as all-zeros.

    Returns
    -------
    list[ComponentPlacement]
        One entry per crystal (length ``analyzer.N``).
    """
    if diffractometer is None:
        diffractometer = Diffractometer()

    M = diffractometer_transform(diffractometer)
    offset = analyzer.cry_offset
    R = analyzer.R
    Rd = analyzer.Rd

    placements: list[ComponentPlacement] = []
    for j in range(analyzer.N):
        # --- crystal ---
        cry_tth = arm_tth + j * offset
        cry_y = R * np.cos(cry_tth)
        cry_z = R * np.sin(cry_tth)
        pitch = -cry_tth + theta_b

        ideal_cry_center = np.array([0.0, cry_y, cry_z])

        # --- baffle ---
        baffle_tth = arm_tth + (j - 0.5) * offset
        baffle_pitch = 2 * theta_b - baffle_tth

        baffle_y = R * np.cos(baffle_tth) + Rd / 2 * np.cos(baffle_pitch)
        baffle_z = R * np.sin(baffle_tth) - Rd / 2 * np.sin(baffle_pitch)

        ideal_baffle_center = np.array([0.0, baffle_y, baffle_z])
        ideal_baffle_z = np.array(
            [0.0, np.sin(baffle_pitch - np.pi / 2), np.cos(baffle_pitch - np.pi / 2)]
        )

        # --- screen ---
        theta_pp = theta_b + pitch  # = 2*theta_b - cry_tth
        screen_y = cry_y + Rd * np.cos(theta_pp)
        screen_z = cry_z - Rd * np.sin(theta_pp)

        ideal_screen_center = np.array([0.0, screen_y, screen_z])
        ideal_screen_z = np.array([0.0, np.sin(theta_pp), np.cos(theta_pp)])

        # --- apply diffractometer transform ---
        placements.append(
            ComponentPlacement(
                crystal_center=_apply_affine_point(M, ideal_cry_center),
                crystal_pitch=pitch,
                baffle_center=_apply_affine_point(M, ideal_baffle_center),
                baffle_z=_apply_affine_direction(M, ideal_baffle_z),
                screen_center=_apply_affine_point(M, ideal_screen_center),
                screen_z=_apply_affine_direction(M, ideal_screen_z),
            )
        )

    return placements
