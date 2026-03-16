#!/usr/bin/env python3
"""
analyze_colmap.py — Parse COLMAP text files and compute per-point/per-image quality metrics.

Reads points3D.txt and images.txt from a COLMAP sparse reconstruction, computes
reprojection error statistics and track lengths, and optionally merges in a
RealityScan quality report JSON. Outputs analysis.json for use by build_tiers.py.

Usage:
    python3 analyze_colmap.py /path/to/dataset [--rs-report quality_report.json]
"""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path


def find_sparse_dir(dataset: Path) -> Path:
    """Locate the COLMAP sparse directory within a dataset."""
    candidates = [
        dataset / "sparse" / "0",
        dataset / "sparse",
        dataset,
    ]
    for d in candidates:
        if (d / "cameras.txt").exists() and (d / "images.txt").exists() and (d / "points3D.txt").exists():
            return d
    # Search up to 3 levels deep
    for depth in range(1, 4):
        pattern = "/".join(["*"] * depth)
        for cameras in dataset.glob(f"{pattern}/cameras.txt"):
            parent = cameras.parent
            if (parent / "images.txt").exists() and (parent / "points3D.txt").exists():
                return parent
    print(f"ERROR: Could not find COLMAP text files (cameras.txt, images.txt, points3D.txt) in {dataset}", file=sys.stderr)
    sys.exit(1)


def parse_points3d(filepath: Path) -> dict:
    """Parse points3D.txt → dict of {point_id: {x, y, z, r, g, b, error, track_length, track}}."""
    points = {}
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            point_id = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
            error = float(parts[7])
            # Track: pairs of (IMAGE_ID, POINT2D_IDX)
            track_entries = parts[8:]
            track_length = len(track_entries) // 2
            track = []
            for i in range(0, len(track_entries) - 1, 2):
                track.append((int(track_entries[i]), int(track_entries[i + 1])))
            points[point_id] = {
                "x": x, "y": y, "z": z,
                "r": r, "g": g, "b": b,
                "error": error,
                "track_length": track_length,
                "track": track,
            }
    return points


def parse_images(filepath: Path) -> dict:
    """Parse images.txt (paired-line format) → dict of {image_id: {name, camera_id, qvec, tvec, observations}}.

    observations is a list of (x, y, point3d_id) tuples.
    """
    images = {}
    with open(filepath, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.strip().startswith("#")]

    # Paired-line format: line 1 = pose data, line 2 = 2D observations
    i = 0
    while i < len(lines) - 1:
        pose_parts = lines[i].split()
        obs_parts = lines[i + 1].split()
        i += 2

        if len(pose_parts) < 10:
            continue

        image_id = int(pose_parts[0])
        qw, qx, qy, qz = float(pose_parts[1]), float(pose_parts[2]), float(pose_parts[3]), float(pose_parts[4])
        tx, ty, tz = float(pose_parts[5]), float(pose_parts[6]), float(pose_parts[7])
        camera_id = int(pose_parts[8])
        name = pose_parts[9]

        # Parse 2D observations: triplets of (X, Y, POINT3D_ID)
        observations = []
        for j in range(0, len(obs_parts) - 2, 3):
            px, py = float(obs_parts[j]), float(obs_parts[j + 1])
            p3d_id = int(obs_parts[j + 2])
            observations.append((px, py, p3d_id))

        images[image_id] = {
            "name": name,
            "camera_id": camera_id,
            "qvec": [qw, qx, qy, qz],
            "tvec": [tx, ty, tz],
            "observations": observations,
        }

    return images


def compute_point_stats(points: dict) -> dict:
    """Compute aggregate statistics over all 3D points."""
    if not points:
        return {"count": 0}

    errors = [p["error"] for p in points.values()]
    tracks = [p["track_length"] for p in points.values()]

    return {
        "count": len(points),
        "error_min": min(errors),
        "error_max": max(errors),
        "error_mean": statistics.mean(errors),
        "error_median": statistics.median(errors),
        "error_stdev": statistics.stdev(errors) if len(errors) > 1 else 0.0,
        "track_min": min(tracks),
        "track_max": max(tracks),
        "track_mean": statistics.mean(tracks),
        "track_median": statistics.median(tracks),
    }


def compute_image_quality(images: dict, points: dict) -> dict:
    """Compute per-image quality metrics from cross-referencing images and points."""
    image_quality = {}

    for image_id, img in images.items():
        # Collect errors of 3D points observed by this image
        observed_point_ids = [obs[2] for obs in img["observations"] if obs[2] != -1]
        observed_errors = [points[pid]["error"] for pid in observed_point_ids if pid in points]
        total_obs = len(img["observations"])
        matched_obs = len(observed_point_ids)

        image_quality[img["name"]] = {
            "image_id": image_id,
            "total_observations": total_obs,
            "matched_3d_points": matched_obs,
            "match_ratio": matched_obs / total_obs if total_obs > 0 else 0.0,
            "avg_point_error": statistics.mean(observed_errors) if observed_errors else 0.0,
            "median_point_error": statistics.median(observed_errors) if observed_errors else 0.0,
        }

    return image_quality


def merge_rs_report(image_quality: dict, rs_report_path: Path) -> dict:
    """Merge RealityScan quality report into image quality metrics."""
    try:
        with open(rs_report_path, "r") as f:
            rs_data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"WARNING: Could not parse RS report {rs_report_path}: {e}", file=sys.stderr)
        return image_quality

    # Build lookup by image name from RS report
    rs_by_name = {}
    for cam in rs_data.get("cameras", []):
        rs_by_name[cam["image"]] = cam

    merged_count = 0
    for img_name, quality in image_quality.items():
        rs_cam = rs_by_name.get(img_name)
        if rs_cam:
            quality["rs_num_points"] = rs_cam.get("numPoints", 0)
            quality["rs_image_coverage"] = rs_cam.get("imageCoverage", 0.0)
            quality["rs_reproj_median"] = rs_cam.get("reprojError", {}).get("median", 0.0)
            quality["rs_reproj_mean"] = rs_cam.get("reprojError", {}).get("mean", 0.0)
            merged_count += 1

    if merged_count > 0:
        print(f"  Merged RS report data for {merged_count}/{len(image_quality)} images")
    else:
        print("  WARNING: RS report had no matching image names — using COLMAP-only metrics", file=sys.stderr)

    # Compute component-level stats from RS report
    component = rs_data.get("component", {})
    if component:
        print(f"  RS component: medianError={component.get('medianError', 'N/A')}, "
              f"avgTrackLength={component.get('avgTrackLength', 'N/A')}")

    return image_quality


def compute_point_tiers(points: dict, point_stats: dict) -> dict:
    """Assign each point a preliminary tier based on error and track length."""
    median_error = point_stats["error_median"]

    tier_assignments = {}
    for pid, p in points.items():
        if p["track_length"] >= 5 and p["error"] < median_error:
            tier = 1
        elif p["track_length"] >= 3 and p["error"] < 2 * median_error:
            tier = 2
        elif p["error"] < 3 * median_error:
            tier = 3
        else:
            tier = 0  # dropped — outlier
        tier_assignments[pid] = tier

    return tier_assignments


def main():
    parser = argparse.ArgumentParser(
        description="Analyze COLMAP text files for per-point/per-image quality metrics"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset with COLMAP sparse reconstruction")
    parser.add_argument("--rs-report", type=Path, default=None,
                        help="Path to RealityScan quality_report.json (optional)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for analysis.json (default: <dataset>/analysis.json)")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"ERROR: {dataset} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Auto-detect RS report if not specified
    rs_report = args.rs_report
    if rs_report is None:
        candidate = dataset / "quality_report.json"
        if candidate.exists():
            rs_report = candidate
            print(f"  Auto-detected RS report: {rs_report}")

    output_path = args.output or (dataset / "analysis.json")

    # Find sparse directory
    sparse_dir = find_sparse_dir(dataset)
    print(f"Analyzing COLMAP data in: {sparse_dir}")

    # Parse COLMAP files
    print("  Parsing points3D.txt...")
    points = parse_points3d(sparse_dir / "points3D.txt")
    print(f"  Found {len(points)} 3D points")

    print("  Parsing images.txt...")
    images = parse_images(sparse_dir / "images.txt")
    print(f"  Found {len(images)} registered images")

    # Compute statistics
    point_stats = compute_point_stats(points)
    print(f"\n  Point cloud statistics:")
    print(f"    Error:  median={point_stats['error_median']:.3f}px, "
          f"mean={point_stats['error_mean']:.3f}px, "
          f"stdev={point_stats['error_stdev']:.3f}px")
    print(f"    Tracks: median={point_stats['track_median']:.1f}, "
          f"mean={point_stats['track_mean']:.1f}, "
          f"range=[{point_stats['track_min']}, {point_stats['track_max']}]")

    # Per-image quality
    image_quality = compute_image_quality(images, points)

    # Merge RS report if available
    if rs_report and rs_report.exists():
        print(f"\n  Merging RealityScan report: {rs_report}")
        image_quality = merge_rs_report(image_quality, rs_report)
    else:
        print("\n  No RealityScan report found — using COLMAP-only metrics")

    # Assign point tiers
    tier_assignments = compute_point_tiers(points, point_stats)
    tier_counts = {t: 0 for t in range(4)}
    for t in tier_assignments.values():
        tier_counts[t] += 1

    print(f"\n  Point tier distribution:")
    print(f"    Tier 1 (seed):    {tier_counts[1]:>7,} points ({100*tier_counts[1]/len(points):.1f}%)")
    print(f"    Tier 2 (expand):  {tier_counts[2]:>7,} points ({100*tier_counts[2]/len(points):.1f}%)")
    print(f"    Tier 3 (full):    {tier_counts[3]:>7,} points ({100*tier_counts[3]/len(points):.1f}%)")
    print(f"    Dropped:          {tier_counts[0]:>7,} points ({100*tier_counts[0]/len(points):.1f}%)")

    # Build output — points section uses string keys for JSON compatibility
    analysis = {
        "dataset": str(dataset),
        "sparse_dir": str(sparse_dir),
        "point_stats": point_stats,
        "point_tiers": {str(pid): tier for pid, tier in tier_assignments.items()},
        "image_quality": image_quality,
        "has_rs_report": rs_report is not None and rs_report.exists(),
        "tier_counts": {
            "tier1": tier_counts[1],
            "tier2": tier_counts[2],
            "tier3": tier_counts[3],
            "dropped": tier_counts[0],
            "total": len(points),
        },
    }

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n  Analysis written to: {output_path}")


if __name__ == "__main__":
    main()
