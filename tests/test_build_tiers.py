"""Tests for build_tiers.py — tier assignment and _is_pose_line."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "realityscan-progressive"))

from build_tiers import _is_pose_line, parse_points3d_raw


class TestIsPoseLine:
    """Verify build_tiers._is_pose_line matches analyze_colmap's version."""

    def test_valid_jpg(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 photo.jpg".split()
        assert _is_pose_line(parts) is True

    def test_scientific_notation_filename(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 1e10.jpg".split()
        assert _is_pose_line(parts) is True

    def test_observation_line(self):
        parts = "100.5 200.3 12345".split()
        assert _is_pose_line(parts) is False

    def test_non_image_extension(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 config.yaml".split()
        assert _is_pose_line(parts) is False


class TestParsePoints3DRaw:
    """Test raw points3D.txt parsing for tier building."""

    def test_basic_raw_parsing(self, tmp_path):
        content = """\
# 3D point list
# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]
1 1.0 2.0 3.0 255 128 0 0.5 1 10 2 20
2 4.0 5.0 6.0 100 100 100 1.2 3 30
"""
        p = tmp_path / "points3D.txt"
        p.write_text(content)
        header, point_lines = parse_points3d_raw(p)
        assert len(header) == 2  # two comment lines
        assert len(point_lines) == 2
        assert 1 in point_lines
        assert 2 in point_lines
        assert "1.0 2.0 3.0" in point_lines[1]

    def test_empty_file(self, tmp_path):
        p = tmp_path / "points3D.txt"
        p.write_text("# Empty\n")
        header, point_lines = parse_points3d_raw(p)
        assert len(header) == 1
        assert len(point_lines) == 0
