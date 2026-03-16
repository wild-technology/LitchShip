#!/usr/bin/env python3
"""
clean_splat.py — Iterative Statistical Outlier Removal for Gaussian Splats

Loads a .ply gaussian splat, removes outlier gaussians based on local density
(mean K-nearest-neighbor distance), and writes the cleaned result.

Each pass computes the mean KNN distance per splat, then removes splats whose
mean distance exceeds (global_mean + std_factor * global_std). Multiple passes
tighten the distribution progressively.

Usage:
    python3 clean_splat.py input.ply output.ply [options]

Options:
    -k, --neighbors     K nearest neighbors to consider (default: 20)
    -s, --std-factor    Std deviation threshold (default: 2.0, lower = more aggressive)
    -p, --passes        Number of iterative passes (default: 3)
    -o, --opacity-min   Also remove splats with opacity below this (default: 0.0 = off)
    -r, --scale-max     Remove splats with scale above this (default: 0.0 = off)
    --bbox              Clip to bounding box: xmin,ymin,zmin,xmax,ymax,zmax
    --dry-run           Report what would be removed without writing output
"""

import argparse
import sys
import time
import numpy as np
from plyfile import PlyData


def load_ply(path):
    """Load a gaussian splat PLY file."""
    print(f"Loading {path}...")
    t0 = time.time()
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    print(f"  Loaded {len(vertex.data):,} splats in {time.time()-t0:.1f}s")
    return plydata, vertex


def get_positions(vertex):
    """Extract XYZ positions from vertex data."""
    x = np.array(vertex['x'], dtype=np.float32)
    y = np.array(vertex['y'], dtype=np.float32)
    z = np.array(vertex['z'], dtype=np.float32)
    return np.column_stack([x, y, z])


def get_opacity(vertex):
    """Extract opacity values (stored as logit in some formats)."""
    if 'opacity' in vertex.data.dtype.names:
        raw = np.array(vertex['opacity'], dtype=np.float32)
        # Gaussian splat PLYs store opacity as logit (inverse sigmoid)
        # sigmoid(x) = 1 / (1 + exp(-x))
        opacity = 1.0 / (1.0 + np.exp(-raw))
        return opacity
    return None


def get_scale(vertex):
    """Extract scale magnitude from log-scale components."""
    names = vertex.data.dtype.names
    scale_names = [n for n in names if n.startswith('scale_')]
    if len(scale_names) >= 3:
        scales = np.column_stack([
            np.array(vertex[n], dtype=np.float32) for n in sorted(scale_names)[:3]
        ])
        # Scales are stored as log-scale; convert to actual scale
        scales = np.exp(scales)
        # Return max scale per splat (the "widest" dimension)
        return np.max(scales, axis=1)
    return None


def statistical_outlier_removal(positions, k, std_factor):
    """Remove outliers based on mean K-nearest-neighbor distance.

    Returns a boolean mask (True = keep).
    """
    from scipy.spatial import cKDTree

    n = len(positions)
    print(f"  Building KD-tree for {n:,} points...")
    t0 = time.time()
    tree = cKDTree(positions)
    print(f"  KD-tree built in {time.time()-t0:.1f}s")

    print(f"  Querying {k} nearest neighbors...")
    t0 = time.time()
    # k+1 because the closest neighbor is the point itself
    dists, _ = tree.query(positions, k=k+1, workers=-1)
    # Drop self-distance (column 0)
    dists = dists[:, 1:]
    print(f"  KNN query done in {time.time()-t0:.1f}s")

    mean_dists = np.mean(dists, axis=1)
    global_mean = np.mean(mean_dists)
    global_std = np.std(mean_dists)
    threshold = global_mean + std_factor * global_std

    keep = mean_dists <= threshold
    n_removed = n - np.sum(keep)

    print(f"  Mean KNN dist: {global_mean:.4f} +/- {global_std:.4f}")
    print(f"  Threshold: {threshold:.4f} ({std_factor:.1f} sigma)")
    print(f"  Removing {n_removed:,} outliers ({100*n_removed/n:.1f}%)")

    return keep


def apply_bbox(positions, bbox_str):
    """Clip to axis-aligned bounding box."""
    parts = [float(x) for x in bbox_str.split(',')]
    if len(parts) != 6:
        print("  ERROR: bbox must be xmin,ymin,zmin,xmax,ymax,zmax")
        sys.exit(1)
    xmin, ymin, zmin, xmax, ymax, zmax = parts
    keep = (
        (positions[:, 0] >= xmin) & (positions[:, 0] <= xmax) &
        (positions[:, 1] >= ymin) & (positions[:, 1] <= ymax) &
        (positions[:, 2] >= zmin) & (positions[:, 2] <= zmax)
    )
    n_removed = len(positions) - np.sum(keep)
    print(f"  Bbox clip: removing {n_removed:,} splats outside [{xmin},{ymin},{zmin}]-[{xmax},{ymax},{zmax}]")
    return keep


def main():
    parser = argparse.ArgumentParser(description="Clean gaussian splat PLY by removing outliers")
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", nargs='?', help="Output .ply file (default: input_cleaned.ply)")
    parser.add_argument("-k", "--neighbors", type=int, default=20, help="K nearest neighbors (default: 20)")
    parser.add_argument("-s", "--std-factor", type=float, default=2.0, help="Std dev threshold (default: 2.0)")
    parser.add_argument("-p", "--passes", type=int, default=3, help="Iterative passes (default: 3)")
    parser.add_argument("-o", "--opacity-min", type=float, default=0.0, help="Min opacity threshold (default: off)")
    parser.add_argument("-r", "--scale-max", type=float, default=0.0, help="Max scale threshold (default: off)")
    parser.add_argument("--bbox", type=str, default=None, help="Bounding box: xmin,ymin,zmin,xmax,ymax,zmax")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write output")
    args = parser.parse_args()

    if not args.output:
        base = args.input.rsplit('.', 1)[0]
        args.output = f"{base}_cleaned.ply"

    plydata, vertex = load_ply(args.input)
    positions = get_positions(vertex)
    original_count = len(positions)

    # Track which splats to keep (start with all True)
    keep_mask = np.ones(len(positions), dtype=bool)

    # --- Pre-filters ---

    # Bounding box clip
    if args.bbox:
        print("\n=== Bounding Box Clip ===")
        keep_mask &= apply_bbox(positions, args.bbox)

    # Opacity filter
    if args.opacity_min > 0:
        opacity = get_opacity(vertex)
        if opacity is not None:
            low_opacity = opacity < args.opacity_min
            n_low = np.sum(low_opacity & keep_mask)
            print(f"\n=== Opacity Filter ===")
            print(f"  Removing {n_low:,} splats with opacity < {args.opacity_min}")
            keep_mask &= ~low_opacity
        else:
            print("  WARNING: No opacity field found in PLY")

    # Scale filter
    if args.scale_max > 0:
        scale = get_scale(vertex)
        if scale is not None:
            big_scale = scale > args.scale_max
            n_big = np.sum(big_scale & keep_mask)
            print(f"\n=== Scale Filter ===")
            print(f"  Removing {n_big:,} splats with scale > {args.scale_max}")
            keep_mask &= ~big_scale
        else:
            print("  WARNING: No scale fields found in PLY")

    # --- Iterative SOR passes ---
    # Each pass operates on the surviving subset, recalculating distances
    for pass_num in range(1, args.passes + 1):
        print(f"\n=== SOR Pass {pass_num}/{args.passes} ===")
        current_indices = np.where(keep_mask)[0]
        current_positions = positions[current_indices]

        if len(current_positions) == 0:
            print("  No splats remaining!")
            break

        pass_keep = statistical_outlier_removal(
            current_positions, args.neighbors, args.std_factor
        )
        # Map back to original indices
        removed_indices = current_indices[~pass_keep]
        keep_mask[removed_indices] = False

    # --- Summary ---
    final_count = np.sum(keep_mask)
    total_removed = original_count - final_count
    print(f"\n{'='*60}")
    print(f"  Original:  {original_count:>12,} splats")
    print(f"  Removed:   {total_removed:>12,} splats ({100*total_removed/original_count:.1f}%)")
    print(f"  Remaining: {final_count:>12,} splats ({100*final_count/original_count:.1f}%)")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n  (dry run — no output written)")
        return

    # --- Write output ---
    print(f"\nWriting {args.output}...")
    t0 = time.time()
    keep_indices = np.where(keep_mask)[0]
    new_vertex_data = vertex.data[keep_indices]
    from plyfile import PlyElement
    new_el = PlyElement.describe(new_vertex_data, 'vertex')
    PlyData([new_el], byte_order=plydata.byte_order).write(args.output)
    print(f"  Written {final_count:,} splats in {time.time()-t0:.1f}s")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
