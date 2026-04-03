"""Tests for the endstation geometry computations.

These baseline tests encode the current inline geometry equations from
Endstation.set_arm() so that refactoring to a helper function can be
verified against known-correct values.
"""

import numpy as np
import numpy.testing as npt
import pytest

from hrd_tools.config import AnalyzerConfig, Diffractometer
from hrd_tools.xrt.geometry import (
    ComponentPlacement,
    compute_arm_positions,
    diffractometer_transform,
)


# ---------------------------------------------------------------------------
# Reference implementation of the *current* set_arm geometry (pure numpy,
# no xrt).  These functions reproduce the equations line-for-line from
# Endstation.set_arm() and are used to generate expected values.
# ---------------------------------------------------------------------------


def _reference_crystal(arm_tth, j, offset, R, theta_b):
    """Current crystal placement from set_arm()."""
    cry_tth = arm_tth + j * offset
    cry_y = R * np.cos(cry_tth)
    cry_z = R * np.sin(cry_tth)
    pitch = -cry_tth + theta_b
    return dict(
        center=np.array([0.0, cry_y, cry_z]),
        pitch=pitch,
    )


def _reference_baffle(arm_tth, j, offset, R, Rd, theta_b):
    """Current baffle placement from set_arm()."""
    baffle_tth = arm_tth + (j - 0.5) * offset
    baffle_pitch = 2 * theta_b - baffle_tth
    baffle_y = R * np.cos(baffle_tth) + Rd / 2 * np.cos(baffle_pitch)
    baffle_z = R * np.sin(baffle_tth) - Rd / 2 * np.sin(baffle_pitch)
    z = (0, np.sin(baffle_pitch - np.pi / 2), np.cos(baffle_pitch - np.pi / 2))
    return dict(
        center=np.array([0.0, baffle_y, baffle_z]),
        z=np.array(z),
    )


def _reference_screen(arm_tth, j, offset, R, Rd, theta_b):
    """Current screen placement from set_arm()."""
    cry = _reference_crystal(arm_tth, j, offset, R, theta_b)
    cry_y, cry_z = cry["center"][1], cry["center"][2]
    theta_pp = theta_b + cry["pitch"]  # = 2*theta_b - cry_tth
    screen_y = cry_y + Rd * np.cos(theta_pp)
    screen_z = cry_z - Rd * np.sin(theta_pp)
    z = (0, np.sin(theta_pp), np.cos(theta_pp))
    return dict(
        center=np.array([0.0, screen_y, screen_z]),
        z=np.array(z),
    )


# ---------------------------------------------------------------------------
# Fixtures providing typical analyzer parameters
# ---------------------------------------------------------------------------


@pytest.fixture
def analyzer_params():
    """Return a dict of analyzer parameters for testing."""
    return dict(
        R=950.0,
        Rd=115.0,
        cry_offset=np.deg2rad(2.5),
    )


@pytest.fixture
def theta_b():
    """A representative Bragg angle (Si 111 at ~29.4 keV)."""
    return np.deg2rad(3.8)


# ---------------------------------------------------------------------------
# Baseline tests — lock in the current geometry
# ---------------------------------------------------------------------------


class TestBaselineCrystal:
    """Verify crystal placement matches the inline equations."""

    @pytest.mark.parametrize("arm_deg", [10, 15, 20, 30])
    def test_single_crystal_center(self, analyzer_params, theta_b, arm_deg):
        arm_tth = np.deg2rad(arm_deg)
        ref = _reference_crystal(
            arm_tth, 0, analyzer_params["cry_offset"], analyzer_params["R"], theta_b
        )
        # crystal sits on a circle of radius R in y-z plane
        r = np.linalg.norm(ref["center"][1:])
        npt.assert_allclose(r, analyzer_params["R"], atol=1e-10)
        # x is always zero in the ideal case
        assert ref["center"][0] == 0.0

    @pytest.mark.parametrize("j", [0, 1, 2, 3])
    def test_multi_crystal_offset(self, analyzer_params, theta_b, j):
        arm_tth = np.deg2rad(15)
        ref = _reference_crystal(
            arm_tth, j, analyzer_params["cry_offset"], analyzer_params["R"], theta_b
        )
        expected_tth = arm_tth + j * analyzer_params["cry_offset"]
        actual_tth = np.arctan2(ref["center"][2], ref["center"][1])
        npt.assert_allclose(actual_tth, expected_tth, atol=1e-12)

    def test_pitch_formula(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        ref = _reference_crystal(
            arm_tth, 0, analyzer_params["cry_offset"], analyzer_params["R"], theta_b
        )
        expected_pitch = -arm_tth + theta_b
        npt.assert_allclose(ref["pitch"], expected_pitch, atol=1e-12)


class TestBaselineBaffle:
    """Verify baffle placement matches the inline equations."""

    def test_baffle_center(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        ref = _reference_baffle(
            arm_tth,
            1,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        # baffle x is always zero
        assert ref["center"][0] == 0.0
        # z-vector should be unit length
        npt.assert_allclose(np.linalg.norm(ref["z"]), 1.0, atol=1e-12)

    def test_baffle_z_perpendicular_to_x(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        ref = _reference_baffle(
            arm_tth,
            0,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        # z-vector has zero x component in ideal case
        assert ref["z"][0] == 0.0


class TestBaselineScreen:
    """Verify screen placement matches the inline equations."""

    def test_screen_center(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        ref_cry = _reference_crystal(
            arm_tth, 0, analyzer_params["cry_offset"], analyzer_params["R"], theta_b
        )
        ref_scr = _reference_screen(
            arm_tth,
            0,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        # screen is Rd away from crystal
        dist = np.linalg.norm(ref_scr["center"] - ref_cry["center"])
        npt.assert_allclose(dist, analyzer_params["Rd"], atol=1e-10)

    def test_screen_x_zero(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(20)
        ref = _reference_screen(
            arm_tth,
            0,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        assert ref["center"][0] == 0.0

    def test_screen_z_unit(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        ref = _reference_screen(
            arm_tth,
            0,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        npt.assert_allclose(np.linalg.norm(ref["z"]), 1.0, atol=1e-12)


class TestBaselineConsistency:
    """Cross-check relationships between components."""

    @pytest.mark.parametrize("arm_deg", [10, 15, 20, 25])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_all_components_in_yz_plane(self, analyzer_params, theta_b, arm_deg, j):
        """In ideal (no diffractometer offset) case, everything has x=0."""
        arm_tth = np.deg2rad(arm_deg)
        p = analyzer_params
        cry = _reference_crystal(arm_tth, j, p["cry_offset"], p["R"], theta_b)
        baf = _reference_baffle(
            arm_tth, j, p["cry_offset"], p["R"], p["Rd"], theta_b
        )
        scr = _reference_screen(
            arm_tth, j, p["cry_offset"], p["R"], p["Rd"], theta_b
        )
        assert cry["center"][0] == 0.0
        assert baf["center"][0] == 0.0
        assert scr["center"][0] == 0.0

    def test_screen_beyond_crystal(self, analyzer_params, theta_b):
        """Screen should be further from origin than crystal (in radial sense)."""
        arm_tth = np.deg2rad(15)
        p = analyzer_params
        cry = _reference_crystal(arm_tth, 0, p["cry_offset"], p["R"], theta_b)
        scr = _reference_screen(
            arm_tth, 0, p["cry_offset"], p["R"], p["Rd"], theta_b
        )
        r_cry = np.linalg.norm(cry["center"])
        r_scr = np.linalg.norm(scr["center"])
        assert r_scr > r_cry

    @pytest.mark.parametrize("arm_deg", [10, 15, 20, 25])
    def test_crystal_on_radius(self, analyzer_params, theta_b, arm_deg):
        """Crystal should be exactly R from origin."""
        arm_tth = np.deg2rad(arm_deg)
        cry = _reference_crystal(
            arm_tth, 0, analyzer_params["cry_offset"], analyzer_params["R"], theta_b
        )
        npt.assert_allclose(
            np.linalg.norm(cry["center"]), analyzer_params["R"], atol=1e-10
        )


# ---------------------------------------------------------------------------
# Helper to build an AnalyzerConfig from the fixture dict
# ---------------------------------------------------------------------------


def _make_analyzer(params, *, N=3):
    return AnalyzerConfig(
        R=params["R"],
        Rd=params["Rd"],
        cry_offset=params["cry_offset"],
        cry_width=102,
        cry_depth=54,
        N=N,
        incident_angle=0,
        thickness=10,
        roll=0,
    )


# ---------------------------------------------------------------------------
# Tests: compute_arm_positions with zero diffractometer offset matches baseline
# ---------------------------------------------------------------------------


class TestHelperZeroDiffractometer:
    """compute_arm_positions with no diffractometer offset must reproduce the inline eqs."""

    @pytest.mark.parametrize("arm_deg", [10, 15, 20, 30])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_crystal_center_matches(self, analyzer_params, theta_b, arm_deg, j):
        arm_tth = np.deg2rad(arm_deg)
        analyzer = _make_analyzer(analyzer_params)
        placements = compute_arm_positions(arm_tth, analyzer, theta_b)
        ref = _reference_crystal(
            arm_tth, j, analyzer_params["cry_offset"], analyzer_params["R"], theta_b
        )
        npt.assert_allclose(placements[j].crystal_center, ref["center"], atol=1e-12)

    @pytest.mark.parametrize("arm_deg", [10, 15, 20, 30])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_crystal_pitch_matches(self, analyzer_params, theta_b, arm_deg, j):
        arm_tth = np.deg2rad(arm_deg)
        analyzer = _make_analyzer(analyzer_params)
        placements = compute_arm_positions(arm_tth, analyzer, theta_b)
        ref = _reference_crystal(
            arm_tth, j, analyzer_params["cry_offset"], analyzer_params["R"], theta_b
        )
        npt.assert_allclose(placements[j].crystal_pitch, ref["pitch"], atol=1e-12)

    @pytest.mark.parametrize("arm_deg", [10, 15, 20])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_baffle_center_matches(self, analyzer_params, theta_b, arm_deg, j):
        arm_tth = np.deg2rad(arm_deg)
        analyzer = _make_analyzer(analyzer_params)
        placements = compute_arm_positions(arm_tth, analyzer, theta_b)
        ref = _reference_baffle(
            arm_tth,
            j,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        npt.assert_allclose(placements[j].baffle_center, ref["center"], atol=1e-10)

    @pytest.mark.parametrize("arm_deg", [10, 15, 20])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_baffle_z_matches(self, analyzer_params, theta_b, arm_deg, j):
        arm_tth = np.deg2rad(arm_deg)
        analyzer = _make_analyzer(analyzer_params)
        placements = compute_arm_positions(arm_tth, analyzer, theta_b)
        ref = _reference_baffle(
            arm_tth,
            j,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        npt.assert_allclose(placements[j].baffle_z, ref["z"], atol=1e-12)

    @pytest.mark.parametrize("arm_deg", [10, 15, 20])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_screen_center_matches(self, analyzer_params, theta_b, arm_deg, j):
        arm_tth = np.deg2rad(arm_deg)
        analyzer = _make_analyzer(analyzer_params)
        placements = compute_arm_positions(arm_tth, analyzer, theta_b)
        ref = _reference_screen(
            arm_tth,
            j,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        npt.assert_allclose(placements[j].screen_center, ref["center"], atol=1e-10)

    @pytest.mark.parametrize("arm_deg", [10, 15, 20])
    @pytest.mark.parametrize("j", [0, 1, 2])
    def test_screen_z_matches(self, analyzer_params, theta_b, arm_deg, j):
        arm_tth = np.deg2rad(arm_deg)
        analyzer = _make_analyzer(analyzer_params)
        placements = compute_arm_positions(arm_tth, analyzer, theta_b)
        ref = _reference_screen(
            arm_tth,
            j,
            analyzer_params["cry_offset"],
            analyzer_params["R"],
            analyzer_params["Rd"],
            theta_b,
        )
        npt.assert_allclose(placements[j].screen_z, ref["z"], atol=1e-12)


# ---------------------------------------------------------------------------
# Tests: diffractometer_transform identity for zero offset
# ---------------------------------------------------------------------------


class TestDiffractometerTransform:
    """Verify the affine transform construction."""

    def test_zero_is_identity(self):
        M = diffractometer_transform(Diffractometer())
        npt.assert_allclose(M, np.eye(4), atol=1e-15)

    def test_pure_translation(self):
        m = Diffractometer(dx=1.0, dy=2.0, dz=3.0)
        M = diffractometer_transform(m)
        npt.assert_allclose(M[:3, :3], np.eye(3), atol=1e-15)
        npt.assert_allclose(M[:3, 3], [1.0, 2.0, 3.0], atol=1e-15)

    def test_pure_roll_rotation_matrix(self):
        """Roll about y: x-z plane rotates, y unchanged."""
        roll = 5  # degrees
        m = Diffractometer(roll=roll)
        M = diffractometer_transform(m)
        # y-axis column should be [0, 1, 0]
        npt.assert_allclose(M[:3, 1], [0, 1, 0], atol=1e-15)
        # no translation
        npt.assert_allclose(M[:3, 3], [0, 0, 0], atol=1e-15)
        # determinant should be 1 (proper rotation)
        npt.assert_allclose(np.linalg.det(M[:3, :3]), 1.0, atol=1e-15)

    def test_pure_yaw_rotation_matrix(self):
        """Yaw about z: x-y plane rotates, z unchanged."""
        yaw = 5  # degrees
        m = Diffractometer(yaw=yaw)
        M = diffractometer_transform(m)
        # z-axis column should be [0, 0, 1]
        npt.assert_allclose(M[:3, 2], [0, 0, 1], atol=1e-15)
        npt.assert_allclose(M[:3, 3], [0, 0, 0], atol=1e-15)


# ---------------------------------------------------------------------------
# Tests: roll diffractometer
# ---------------------------------------------------------------------------


class TestRollDiffractometer:
    """Roll rotates the crystal plane about the y-axis."""

    def test_roll_moves_crystal_out_of_yz_plane(self, analyzer_params, theta_b):
        """With nonzero roll, crystal x != 0."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        roll = 2  # degrees
        m = Diffractometer(roll=roll)
        placements = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        # x should be nonzero
        assert abs(placements[0].crystal_center[0]) > 0.1

    def test_roll_preserves_y(self, analyzer_params, theta_b):
        """Roll about y should leave the y-coordinate unchanged."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        roll = 2  # degrees
        m = Diffractometer(roll=roll)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(
            with_m[0].crystal_center[1], no_m[0].crystal_center[1], atol=1e-10
        )

    def test_roll_preserves_crystal_radius(self, analyzer_params, theta_b):
        """Roll is a rotation, so distance from origin is preserved."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        roll = 5  # degrees
        m = Diffractometer(roll=roll)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(
            np.linalg.norm(with_m[0].crystal_center),
            np.linalg.norm(no_m[0].crystal_center),
            atol=1e-10,
        )

    def test_roll_sign_convention(self, analyzer_params, theta_b):
        """Positive roll about y should move +z towards +x (right-hand rule)."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        # small positive roll
        roll = 1  # degrees
        m = Diffractometer(roll=roll)
        p = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        # crystal is at positive z, roll about y → positive x
        assert p[0].crystal_center[0] > 0


# ---------------------------------------------------------------------------
# Tests: yaw diffractometer
# ---------------------------------------------------------------------------


class TestYawDiffractometer:
    """Yaw rotates the crystal plane about the z-axis."""

    def test_yaw_moves_crystal_out_of_yz_plane(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        yaw = 2  # degrees
        m = Diffractometer(yaw=yaw)
        p = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        assert abs(p[0].crystal_center[0]) > 0.1

    def test_yaw_preserves_z(self, analyzer_params, theta_b):
        """Yaw about z should leave the z-coordinate unchanged."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        yaw = 2  # degrees
        m = Diffractometer(yaw=yaw)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(
            with_m[0].crystal_center[2], no_m[0].crystal_center[2], atol=1e-10
        )

    def test_yaw_preserves_distance(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        yaw = 5  # degrees
        m = Diffractometer(yaw=yaw)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(
            np.linalg.norm(with_m[0].crystal_center),
            np.linalg.norm(no_m[0].crystal_center),
            atol=1e-10,
        )


# ---------------------------------------------------------------------------
# Tests: displacement
# ---------------------------------------------------------------------------


class TestDisplacement:
    """Translation shifts all component positions uniformly."""

    def test_displacement_shifts_crystal(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        dx, dy, dz = 1.0, 2.0, -0.5
        m = Diffractometer(dx=dx, dy=dy, dz=dz)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(
            with_m[0].crystal_center - no_m[0].crystal_center,
            [dx, dy, dz],
            atol=1e-12,
        )

    def test_displacement_shifts_baffle(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        dx, dy, dz = 1.0, 2.0, -0.5
        m = Diffractometer(dx=dx, dy=dy, dz=dz)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(
            with_m[0].baffle_center - no_m[0].baffle_center,
            [dx, dy, dz],
            atol=1e-12,
        )

    def test_displacement_shifts_screen(self, analyzer_params, theta_b):
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        dx, dy, dz = 1.0, 2.0, -0.5
        m = Diffractometer(dx=dx, dy=dy, dz=dz)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(
            with_m[0].screen_center - no_m[0].screen_center,
            [dx, dy, dz],
            atol=1e-12,
        )

    def test_displacement_preserves_orientations(self, analyzer_params, theta_b):
        """Translation should not change baffle_z or screen_z."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        m = Diffractometer(dx=5.0, dy=-3.0, dz=1.0)
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)
        npt.assert_allclose(with_m[0].baffle_z, no_m[0].baffle_z, atol=1e-15)
        npt.assert_allclose(with_m[0].screen_z, no_m[0].screen_z, atol=1e-15)


# ---------------------------------------------------------------------------
# Tests: composed diffractometer (roll + yaw + displacement)
# ---------------------------------------------------------------------------


class TestComposedDiffractometer:
    """Verify combined diffractometer transform against manual computation."""

    def test_roll_yaw_displacement(self, analyzer_params, theta_b):
        """Full composition matches manual matrix multiplication."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=1)
        roll_deg = 1  # degrees
        yaw_deg = 0.5  # degrees
        dx, dy, dz = 0.5, -0.3, 0.1
        m = Diffractometer(roll=roll_deg, yaw=yaw_deg, dx=dx, dy=dy, dz=dz)

        # compute via helper
        p = compute_arm_positions(arm_tth, analyzer, theta_b, m)

        # compute manually: ideal position then transform (convert to radians for matrix)
        cry_tth = arm_tth
        ideal = np.array([0.0, analyzer.R * np.cos(cry_tth), analyzer.R * np.sin(cry_tth)])
        roll_rad = np.deg2rad(roll_deg)
        yaw_rad = np.deg2rad(yaw_deg)
        # roll about y
        cr, sr = np.cos(roll_rad), np.sin(roll_rad)
        R_roll = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
        # yaw about z
        cy, sy = np.cos(yaw_rad), np.sin(yaw_rad)
        R_yaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        expected = R_yaw @ R_roll @ ideal + np.array([dx, dy, dz])

        npt.assert_allclose(p[0].crystal_center, expected, atol=1e-10)

    def test_multiple_crystals_consistent(self, analyzer_params, theta_b):
        """All crystals should be affected by the same transform."""
        arm_tth = np.deg2rad(15)
        analyzer = _make_analyzer(analyzer_params, N=3)
        m = Diffractometer(
            roll=1, yaw=0.5, dx=1, dy=2, dz=3  # degrees for angles
        )
        no_m = compute_arm_positions(arm_tth, analyzer, theta_b)
        with_m = compute_arm_positions(arm_tth, analyzer, theta_b, m)

        M = diffractometer_transform(m)
        for j in range(3):
            # manually transform ideal crystal center
            ideal = no_m[j].crystal_center
            homo = np.append(ideal, 1.0)
            expected = (M @ homo)[:3]
            npt.assert_allclose(with_m[j].crystal_center, expected, atol=1e-10)
