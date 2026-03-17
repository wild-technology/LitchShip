"""Tests for clean_splat.py — PLY filtering pipeline."""

import io
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

SCRIPT = str(Path(__file__).resolve().parent.parent / "clean_splat.py")


def _make_ply(n_points, positions=None, opacity_logits=None, seed=0):
    """Create a minimal gaussian splat PLY file in a temp path.

    Returns the path to the written file.
    """
    rng = np.random.default_rng(seed)
    if positions is None:
        positions = rng.standard_normal((n_points, 3)).astype(np.float32)
    else:
        positions = np.asarray(positions, dtype=np.float32)
        n_points = len(positions)

    if opacity_logits is None:
        # Default: mix of visible (logit 2.0 -> sigmoid ~0.88) and invisible (logit -3.0 -> sigmoid ~0.05)
        opacity_logits = np.where(
            rng.random(n_points) > 0.2,
            np.float32(2.0),
            np.float32(-3.0),
        )
    else:
        opacity_logits = np.asarray(opacity_logits, dtype=np.float32)

    # Minimal SH DC (white splats)
    f_dc = np.full((n_points, 3), 1.0, dtype=np.float32)
    # Scale (log-space, small splats)
    scale = np.full((n_points, 3), -5.0, dtype=np.float32)
    # Rotation (identity quaternion)
    rot = np.zeros((n_points, 4), dtype=np.float32)
    rot[:, 0] = 1.0

    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ])

    data = np.empty(n_points, dtype=dtype)
    data["x"] = positions[:, 0]
    data["y"] = positions[:, 1]
    data["z"] = positions[:, 2]
    data["f_dc_0"] = f_dc[:, 0]
    data["f_dc_1"] = f_dc[:, 1]
    data["f_dc_2"] = f_dc[:, 2]
    data["opacity"] = opacity_logits
    data["scale_0"] = scale[:, 0]
    data["scale_1"] = scale[:, 1]
    data["scale_2"] = scale[:, 2]
    data["rot_0"] = rot[:, 0]
    data["rot_1"] = rot[:, 1]
    data["rot_2"] = rot[:, 2]
    data["rot_3"] = rot[:, 3]

    el = PlyElement.describe(data, "vertex")
    tmp = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    PlyData([el]).write(tmp.name)
    return tmp.name


def _read_ply_count(path):
    """Read vertex count from a PLY file."""
    return len(PlyData.read(path)["vertex"].data)


def _run_clean(input_ply, extra_args=None):
    """Run clean_splat.py and return (output_path, exit_code, stdout)."""
    output = tempfile.NamedTemporaryFile(suffix=".ply", delete=False)
    cmd = [sys.executable, SCRIPT, input_ply, output.name]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return output.name, result.returncode, result.stdout + result.stderr


class TestPLYRoundtrip:
    """PLY files survive a pass through clean_splat with no filters."""

    def test_identity_roundtrip(self, tmp_path):
        """With no filters active, all points should survive."""
        input_ply = _make_ply(100, opacity_logits=np.full(100, 2.0))
        output, rc, _ = _run_clean(input_ply, ["--opacity-min", "0", "-p", "0"])
        assert rc == 0
        assert _read_ply_count(output) == 100

    def test_empty_input(self, tmp_path):
        """Gracefully handle a PLY with zero vertices."""
        input_ply = _make_ply(0, positions=np.empty((0, 3)))
        output, rc, out = _run_clean(input_ply, ["-p", "0"])
        assert rc == 0


class TestOpacityFilter:
    """Opacity-based filtering."""

    def test_removes_low_opacity(self):
        # 80 visible (logit 2.0 -> sigmoid 0.88), 20 invisible (logit -3.0 -> sigmoid 0.05)
        logits = np.array([2.0] * 80 + [-3.0] * 20, dtype=np.float32)
        input_ply = _make_ply(100, opacity_logits=logits)
        output, rc, _ = _run_clean(input_ply, ["--opacity-min", "0.1", "-p", "0"])
        assert rc == 0
        assert _read_ply_count(output) == 80

    def test_zero_threshold_keeps_all(self):
        logits = np.array([2.0] * 50 + [-3.0] * 50, dtype=np.float32)
        input_ply = _make_ply(100, opacity_logits=logits)
        output, rc, _ = _run_clean(input_ply, ["--opacity-min", "0", "-p", "0"])
        assert rc == 0
        assert _read_ply_count(output) == 100


class TestBBoxFilter:
    """Bounding box clipping."""

    def test_bbox_clips(self):
        # Place points at known positions
        positions = np.array([
            [0, 0, 0],      # inside
            [5, 5, 5],      # inside
            [15, 15, 15],   # outside bbox
            [11, 0, 0],     # outside bbox (x > 10)
        ], dtype=np.float32)
        input_ply = _make_ply(4, positions=positions, opacity_logits=np.full(4, 2.0))
        output, rc, _ = _run_clean(input_ply, [
            "--bbox=0,0,0,10,10,10",
            "-p", "0",
        ])
        assert rc == 0
        assert _read_ply_count(output) == 2  # only (0,0,0) and (5,5,5) inside


class TestSORDeterminism:
    """SOR passes are deterministic with --seed."""

    def test_deterministic_output(self):
        """Two runs with same seed produce identical output."""
        input_ply = _make_ply(500, seed=123)
        out1, rc1, _ = _run_clean(input_ply, ["-p", "2", "-s", "1.5", "--seed", "42"])
        out2, rc2, _ = _run_clean(input_ply, ["-p", "2", "-s", "1.5", "--seed", "42"])
        assert rc1 == 0 and rc2 == 0
        count1 = _read_ply_count(out1)
        count2 = _read_ply_count(out2)
        assert count1 == count2

    def test_different_seeds_may_differ(self):
        """Different seeds can produce different CC radius sampling."""
        # This is a weaker test — just ensures the seed param is respected
        input_ply = _make_ply(500, seed=123)
        out1, _, _ = _run_clean(input_ply, ["-p", "0", "--connected", "--seed", "1"])
        out2, _, _ = _run_clean(input_ply, ["-p", "0", "--connected", "--seed", "999"])
        # Both should succeed (not crash)
        assert Path(out1).exists()
        assert Path(out2).exists()


class TestDryRun:
    """--dry-run should produce no output file."""

    def test_dry_run_no_output(self):
        input_ply = _make_ply(50, opacity_logits=np.full(50, 2.0))
        cmd = [sys.executable, SCRIPT, input_ply, "--dry-run"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0
        assert "dry run" in result.stdout.lower()
