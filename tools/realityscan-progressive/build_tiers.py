#!/usr/bin/env python3
"""
build_tiers.py — Build tiered COLMAP datasets using growing-seed approach.

Takes the analysis.json from analyze_colmap.py and creates three COLMAP-format
dataset directories with progressively more 3D seed points. All images stay in
every tier — only points3D.txt changes.

Usage:
    python3 build_tiers.py /path/to/dataset --output /path/to/staging
    python3 build_tiers.py /path/to/dataset --t1-min-track 5 --t1-max-error-pct 50
"""

import argparse
import json
import os
import shutil
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
    for depth in range(1, 4):
        pattern = "/".join(["*"] * depth)
        for cameras in dataset.glob(f"{pattern}/cameras.txt"):
            parent = cameras.parent
            if (parent / "images.txt").exists() and (parent / "points3D.txt").exists():
                return parent
    print(f"ERROR: Could not find COLMAP text files in {dataset}", file=sys.stderr)
    sys.exit(1)


def find_images_dir(dataset: Path) -> Path:
    """Locate the images directory within a dataset."""
    candidates = [dataset / "images", dataset]
    for d in candidates:
        if d.is_dir():
            image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
            for f in d.iterdir():
                if f.suffix.lower() in image_exts:
                    return d
    print(f"ERROR: Could not find images directory in {dataset}", file=sys.stderr)
    sys.exit(1)


def parse_points3d_raw(filepath: Path) -> tuple[list[str], dict[int, str]]:
    """Parse points3D.txt, returning header comments and per-point raw lines.

    Returns:
        header_lines: list of comment/empty lines at the top
        point_lines: dict of {point_id: raw_line_string}
    """
    header_lines = []
    point_lines = {}
    in_header = True

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if in_header and (not stripped or stripped.startswith("#")):
                header_lines.append(line)
                continue
            in_header = False
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) >= 8:
                point_id = int(parts[0])
                point_lines[point_id] = line
    return header_lines, point_lines


def rewrite_images_txt(src_path: Path, dst_path: Path, kept_point_ids: set[int]):
    """Copy images.txt, replacing POINT3D_ID references to removed points with -1.

    Preserves paired-line format. All images are kept — only observation references change.
    """
    with open(src_path, "r") as f:
        lines = f.readlines()

    output = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Pass through comments and empty lines
        if not stripped or stripped.startswith("#"):
            output.append(line)
            i += 1
            continue

        # Line 1 of pair: pose data — pass through unchanged
        output.append(line)
        i += 1

        # Line 2 of pair: observations (X Y POINT3D_ID) ...
        if i < len(lines):
            obs_line = lines[i]
            obs_stripped = obs_line.strip()
            i += 1

            if not obs_stripped:
                output.append(obs_line)
                continue

            parts = obs_stripped.split()
            new_parts = []
            # Process in triplets: X Y POINT3D_ID
            for j in range(0, len(parts) - 2, 3):
                x_str = parts[j]
                y_str = parts[j + 1]
                p3d_id = int(parts[j + 2])

                if p3d_id != -1 and p3d_id not in kept_point_ids:
                    new_parts.extend([x_str, y_str, "-1"])
                else:
                    new_parts.extend([x_str, y_str, str(p3d_id)])

            output.append(" ".join(new_parts) + "\n")

    with open(dst_path, "w") as f:
        f.writelines(output)


def build_tier(
    tier_name: str,
    tier_num: int,
    output_dir: Path,
    sparse_dir: Path,
    images_dir: Path,
    header_lines: list[str],
    point_lines: dict[int, str],
    kept_point_ids: set[int],
    analysis: dict,
):
    """Build a single tier's COLMAP dataset directory."""
    tier_dir = output_dir / tier_name
    tier_sparse = tier_dir / "sparse" / "0"
    tier_images = tier_dir / "images"

    tier_sparse.mkdir(parents=True, exist_ok=True)

    # Copy cameras.txt verbatim
    shutil.copy2(sparse_dir / "cameras.txt", tier_sparse / "cameras.txt")

    # Write filtered points3D.txt
    with open(tier_sparse / "points3D.txt", "w") as f:
        f.writelines(header_lines)
        for pid in sorted(kept_point_ids):
            if pid in point_lines:
                f.write(point_lines[pid])

    # Rewrite images.txt with invalidated observation references
    rewrite_images_txt(
        sparse_dir / "images.txt",
        tier_sparse / "images.txt",
        kept_point_ids,
    )

    # Symlink images directory to avoid duplication
    if tier_images.exists() or tier_images.is_symlink():
        tier_images.unlink() if tier_images.is_symlink() else shutil.rmtree(tier_images)
    tier_images.symlink_to(images_dir.resolve())

    # Compute tier stats
    tier_errors = []
    tier_tracks = []
    for pid in kept_point_ids:
        if pid in analysis.get("_points_cache", {}):
            p = analysis["_points_cache"][pid]
            tier_errors.append(p["error"])
            tier_tracks.append(p["track_length"])

    total = len(point_lines)
    count = len(kept_point_ids)
    avg_error = sum(tier_errors) / len(tier_errors) if tier_errors else 0
    avg_track = sum(tier_tracks) / len(tier_tracks) if tier_tracks else 0

    print(f"  Tier {tier_num} ({tier_name}): {count:>7,} / {total:,} points "
          f"({100*count/total:.1f}%) — avg track {avg_track:.1f}, avg error {avg_error:.2f}px")

    return {"count": count, "avg_error": avg_error, "avg_track": avg_track}


def main():
    parser = argparse.ArgumentParser(
        description="Build tiered COLMAP datasets using growing-seed approach"
    )
    parser.add_argument("dataset", type=Path, help="Path to dataset with COLMAP sparse reconstruction")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for staged tiers (default: <dataset>/staging)")
    parser.add_argument("--analysis", type=Path, default=None,
                        help="Path to analysis.json (default: <dataset>/analysis.json)")
    parser.add_argument("--t1-min-track", type=int, default=5,
                        help="Tier 1: minimum track length (default: 5)")
    parser.add_argument("--t1-max-error-pct", type=float, default=100.0,
                        help="Tier 1: max error as %% of median (default: 100 = median)")
    parser.add_argument("--t2-min-track", type=int, default=3,
                        help="Tier 2: minimum track length (default: 3)")
    parser.add_argument("--t2-max-error-pct", type=float, default=200.0,
                        help="Tier 2: max error as %% of median (default: 200 = 2× median)")
    parser.add_argument("--t3-max-error-pct", type=float, default=300.0,
                        help="Tier 3: max error as %% of median (default: 300 = 3× median)")
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    if not dataset.is_dir():
        print(f"ERROR: {dataset} is not a directory", file=sys.stderr)
        sys.exit(1)

    output_dir = (args.output or (dataset / "staging")).resolve()
    analysis_path = args.analysis or (dataset / "analysis.json")

    # Load analysis
    if not analysis_path.exists():
        print(f"ERROR: analysis.json not found at {analysis_path}", file=sys.stderr)
        print("  Run analyze_colmap.py first.", file=sys.stderr)
        sys.exit(1)

    with open(analysis_path, "r") as f:
        analysis = json.load(f)

    point_tiers = analysis["point_tiers"]
    point_stats = analysis["point_stats"]
    median_error = point_stats["error_median"]

    # Find COLMAP and image directories
    sparse_dir = find_sparse_dir(dataset)
    images_dir = find_images_dir(dataset)
    print(f"Building tiered datasets from: {sparse_dir}")
    print(f"Images from: {images_dir}")
    print(f"Output to: {output_dir}")

    # Parse raw points3D.txt to preserve exact formatting
    header_lines, point_lines = parse_points3d_raw(sparse_dir / "points3D.txt")

    # Re-parse points for stats (we need error and track_length per point)
    # Import the parser from analyze_colmap
    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_colmap import parse_points3d
    points_data = parse_points3d(sparse_dir / "points3D.txt")
    analysis["_points_cache"] = points_data

    # Compute tier thresholds
    t1_max_error = median_error * (args.t1_max_error_pct / 100.0)
    t2_max_error = median_error * (args.t2_max_error_pct / 100.0)
    t3_max_error = median_error * (args.t3_max_error_pct / 100.0)

    print(f"\nThresholds (median error = {median_error:.3f}px):")
    print(f"  Tier 1: track ≥ {args.t1_min_track}, error < {t1_max_error:.3f}px")
    print(f"  Tier 2: track ≥ {args.t2_min_track}, error < {t2_max_error:.3f}px")
    print(f"  Tier 3: error < {t3_max_error:.3f}px")
    print()

    # Build point sets — each tier is cumulative (growing seed)
    tier1_ids = set()
    tier2_ids = set()
    tier3_ids = set()

    for pid_str, tier in point_tiers.items():
        pid = int(pid_str)
        p = points_data.get(pid)
        if p is None:
            continue

        # Re-evaluate against user-specified thresholds (analysis.json used defaults)
        if p["track_length"] >= args.t1_min_track and p["error"] < t1_max_error:
            tier1_ids.add(pid)
            tier2_ids.add(pid)
            tier3_ids.add(pid)
        elif p["track_length"] >= args.t2_min_track and p["error"] < t2_max_error:
            tier2_ids.add(pid)
            tier3_ids.add(pid)
        elif p["error"] < t3_max_error:
            tier3_ids.add(pid)
        # else: dropped (outlier)

    dropped = len(point_lines) - len(tier3_ids)

    # Build tier directories
    output_dir.mkdir(parents=True, exist_ok=True)

    build_tier("tier1", 1, output_dir, sparse_dir, images_dir, header_lines, point_lines, tier1_ids, analysis)
    build_tier("tier2", 2, output_dir, sparse_dir, images_dir, header_lines, point_lines, tier2_ids, analysis)
    build_tier("tier3", 3, output_dir, sparse_dir, images_dir, header_lines, point_lines, tier3_ids, analysis)

    print(f"  Dropped:          {dropped:>7,} points ({100*dropped/len(point_lines):.1f}%) — error ≥ {t3_max_error:.3f}px")

    # Write tier manifest
    manifest = {
        "dataset": str(dataset),
        "output_dir": str(output_dir),
        "median_error": median_error,
        "thresholds": {
            "t1_min_track": args.t1_min_track,
            "t1_max_error": t1_max_error,
            "t2_min_track": args.t2_min_track,
            "t2_max_error": t2_max_error,
            "t3_max_error": t3_max_error,
        },
        "tier_counts": {
            "tier1": len(tier1_ids),
            "tier2": len(tier2_ids),
            "tier3": len(tier3_ids),
            "dropped": dropped,
            "total": len(point_lines),
        },
    }
    manifest_path = output_dir / "tier_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
