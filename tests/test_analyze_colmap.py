"""Tests for analyze_colmap.py — COLMAP parsing and pose-line detection."""

import sys
from pathlib import Path

# Add the module path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "realityscan-progressive"))

from analyze_colmap import _is_pose_line, parse_images, parse_points3d


class TestIsPoseLine:
    """Tests for _is_pose_line heuristic."""

    def test_valid_pose_line(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 image001.jpg".split()
        assert _is_pose_line(parts) is True

    def test_valid_png(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 frame.png".split()
        assert _is_pose_line(parts) is True

    def test_valid_tiff(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 scan.tiff".split()
        assert _is_pose_line(parts) is True

    def test_valid_bmp(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 photo.bmp".split()
        assert _is_pose_line(parts) is True

    def test_scientific_notation_filename(self):
        """Regression: '1e10.jpg' should be detected as an image, not a number."""
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 1e10.jpg".split()
        assert _is_pose_line(parts) is True

    def test_numeric_prefix_filename(self):
        """Filenames like '0001.jpeg' should match."""
        parts = "42 0.1 0.2 0.3 0.4 0.5 0.6 0.7 2 0001.jpeg".split()
        assert _is_pose_line(parts) is True

    def test_observation_line_rejected(self):
        """Observation lines (X Y POINT3D_ID triplets) should NOT match."""
        parts = "100.5 200.3 12345 101.2 201.4 12346 102.0 202.1 -1".split()
        assert _is_pose_line(parts) is False

    def test_too_few_fields(self):
        parts = "1 0.5 0.5 0.5".split()
        assert _is_pose_line(parts) is False

    def test_non_integer_id(self):
        parts = "abc 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 image.jpg".split()
        assert _is_pose_line(parts) is False

    def test_non_integer_camera_id(self):
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 xyz image.jpg".split()
        assert _is_pose_line(parts) is False

    def test_no_extension(self):
        """Filename without extension should not match."""
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 noextension".split()
        assert _is_pose_line(parts) is False

    def test_wrong_extension(self):
        """Non-image extension should not match."""
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 data.txt".split()
        assert _is_pose_line(parts) is False

    def test_case_insensitive_extension(self):
        """JPG, PNG etc. in upper case should still match."""
        parts = "1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 PHOTO.JPG".split()
        assert _is_pose_line(parts) is True


class TestParseImages:
    """Test images.txt parsing with sample data."""

    def _write_images_txt(self, tmp_path, content):
        p = tmp_path / "images.txt"
        p.write_text(content)
        return p

    def test_standard_paired_format(self, tmp_path):
        content = """\
# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID) ...
1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 image001.jpg
100.0 200.0 42 150.0 250.0 -1
2 0.6 0.6 0.6 0.6 4.0 5.0 6.0 1 image002.jpg
300.0 400.0 99
"""
        images = parse_images(self._write_images_txt(tmp_path, content))
        assert len(images) == 2
        assert images[1]["name"] == "image001.jpg"
        assert images[2]["name"] == "image002.jpg"
        assert len(images[1]["observations"]) == 2
        assert images[1]["observations"][0] == (100.0, 200.0, 42)
        assert images[1]["observations"][1] == (150.0, 250.0, -1)

    def test_missing_observation_line(self, tmp_path):
        """Image with no observation line (next line is another pose)."""
        content = """\
1 0.5 0.5 0.5 0.5 1.0 2.0 3.0 1 image001.jpg
2 0.6 0.6 0.6 0.6 4.0 5.0 6.0 1 image002.jpg
100.0 200.0 42
"""
        images = parse_images(self._write_images_txt(tmp_path, content))
        assert len(images) == 2
        assert len(images[1]["observations"]) == 0
        assert len(images[2]["observations"]) == 1

    def test_empty_file(self, tmp_path):
        content = "# Empty file\n"
        images = parse_images(self._write_images_txt(tmp_path, content))
        assert len(images) == 0


class TestParsePoints3D:
    """Test points3D.txt parsing."""

    def test_basic_parsing(self, tmp_path):
        content = """\
# 3D point list
# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
1 1.0 2.0 3.0 255 128 0 0.5 1 10 2 20
2 4.0 5.0 6.0 100 100 100 1.2 3 30
"""
        p = tmp_path / "points3D.txt"
        p.write_text(content)
        points = parse_points3d(p)
        assert len(points) == 2
        assert points[1]["x"] == 1.0
        assert points[1]["error"] == 0.5
        assert points[1]["track_length"] == 2
        assert points[1]["track"] == [(1, 10), (2, 20)]
        assert points[2]["track_length"] == 1
