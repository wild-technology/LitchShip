#!/usr/bin/env python3
"""
clean_splat.py — Post-processing for Gaussian Splats

Loads a .ply gaussian splat, removes unwanted gaussians based on opacity,
color, spatial density, scale, and bounding box, then writes the result.

Filters (applied in this order):
  1. Bounding box clip
  2. Opacity filter (remove near-invisible splats)
  3. Color filter (remove by hue range, saturation, or keep-hue)
  4. Scale filter (remove oversized splats)
  5. Statistical Outlier Removal (iterative KNN-based density filter)

Usage:
    python3 clean_splat.py input.ply output.ply [options]

Examples:
    # Default cleanup (opacity + SOR):
    python3 clean_splat.py splat.ply splat_clean.ply --opacity-min 0.05

    # Remove blue artifacts from a yellow-green object:
    python3 clean_splat.py splat.ply splat_clean.ply --remove-hue 180,300 --sat-min 0.2

    # Keep only yellow-green splats:
    python3 clean_splat.py splat.ply splat_clean.ply --keep-hue 30,160

    # Full pipeline:
    python3 clean_splat.py splat.ply splat_clean.ply \\
        --opacity-min 0.05 --remove-hue 200,280 --sat-min 0.3 -s 2.0 -p 3
"""

import argparse
import sys
import time
import numpy as np
from plyfile import PlyData


# SH basis function constant for degree 0: C0 = 1 / (2 * sqrt(pi))
SH_C0 = 0.28209479177387814


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
    """Extract opacity values (stored as logit — needs sigmoid)."""
    if 'opacity' in vertex.data.dtype.names:
        raw = np.array(vertex['opacity'], dtype=np.float32)
        opacity = 1.0 / (1.0 + np.exp(-raw))
        return opacity
    return None


def get_scale(vertex):
    """Extract max scale per splat from log-scale components."""
    names = vertex.data.dtype.names
    scale_names = [n for n in names if n.startswith('scale_')]
    if len(scale_names) >= 3:
        scales = np.column_stack([
            np.array(vertex[n], dtype=np.float32) for n in sorted(scale_names)[:3]
        ])
        scales = np.exp(scales)
        return np.max(scales, axis=1)
    return None


def get_rgb(vertex):
    """Extract RGB colors from SH DC coefficients (f_dc_0, f_dc_1, f_dc_2).

    Gaussian splat PLYs store the base color as spherical harmonics degree-0
    coefficients. Convert to linear RGB: rgb = sh_dc * C0 + 0.5, then clamp.
    """
    names = vertex.data.dtype.names
    if 'f_dc_0' in names and 'f_dc_1' in names and 'f_dc_2' in names:
        r = np.array(vertex['f_dc_0'], dtype=np.float32) * SH_C0 + 0.5
        g = np.array(vertex['f_dc_1'], dtype=np.float32) * SH_C0 + 0.5
        b = np.array(vertex['f_dc_2'], dtype=np.float32) * SH_C0 + 0.5
        rgb = np.column_stack([r, g, b])
        np.clip(rgb, 0.0, 1.0, out=rgb)
        return rgb
    return None


def rgb_to_hsv(rgb):
    """Convert Nx3 RGB (0-1) array to Nx3 HSV (H in degrees 0-360, S/V 0-1)."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    # Value
    v = maxc

    # Saturation
    s = np.zeros_like(v)
    nonzero = maxc > 0
    s[nonzero] = delta[nonzero] / maxc[nonzero]

    # Hue
    h = np.zeros_like(v)
    mask = delta > 1e-8
    # Red is max
    idx = mask & (maxc == r)
    h[idx] = 60.0 * (((g[idx] - b[idx]) / delta[idx]) % 6.0)
    # Green is max
    idx = mask & (maxc == g)
    h[idx] = 60.0 * (((b[idx] - r[idx]) / delta[idx]) + 2.0)
    # Blue is max
    idx = mask & (maxc == b)
    h[idx] = 60.0 * (((r[idx] - g[idx]) / delta[idx]) + 4.0)

    h[h < 0] += 360.0

    return np.column_stack([h, s, v])


def print_color_histogram(hsv, keep_mask=None):
    """Print a hue histogram of the splats for diagnostics."""
    if keep_mask is not None:
        h = hsv[keep_mask, 0]
        s = hsv[keep_mask, 1]
    else:
        h = hsv[:, 0]
        s = hsv[:, 1]

    # Only count splats with visible saturation
    saturated = s > 0.1
    h_sat = h[saturated]
    n = len(h_sat)
    if n == 0:
        print("  (no saturated splats to histogram)")
        return

    bins = np.arange(0, 375, 15)
    counts, _ = np.histogram(h_sat, bins=bins)
    labels = [
        "Red", "Red-Org", "Orange", "Org-Yel",
        "Yellow", "Yel-Grn", "Green", "Grn-Cyn",
        "Cyan", "Cyn-Cyn", "Cyn-Blu", "Lt Blue",
        "Blue", "Blu-Blu", "Indigo", "Violet",
        "Purple", "Magenta", "Pink", "Pink-Red",
        "Red2", "Red2+", "Red3", "Red3+"
    ]
    print(f"  Hue histogram ({n:,} saturated splats):")
    max_count = max(counts) if max(counts) > 0 else 1
    for i, count in enumerate(counts):
        pct = 100 * count / n
        bar_len = int(40 * count / max_count)
        bar = "#" * bar_len
        label = labels[i] if i < len(labels) else f"{bins[i]}"
        print(f"    {bins[i]:>3}-{bins[i+1]:<3} {label:<9} {bar:<40} {count:>8,} ({pct:>5.1f}%)")


def connected_component_filter(positions, radius):
    """Keep only the largest spatially connected cluster of splats.

    Two splats are "connected" if they are within `radius` of each other.
    Uses cKDTree.sparse_distance_matrix for fast sparse graph construction,
    then scipy connected_components.

    Returns a boolean mask (True = keep).
    """
    from scipy.spatial import cKDTree
    from scipy.sparse.csgraph import connected_components

    n = len(positions)
    print(f"  Building KD-tree for {n:,} points...")
    t0 = time.time()
    tree = cKDTree(positions)
    print(f"  KD-tree built in {time.time()-t0:.1f}s")

    # Build sparse adjacency directly — much faster than query_ball + lil_matrix
    print(f"  Building sparse adjacency (r={radius:.4f})...")
    t0 = time.time()
    adj = tree.sparse_distance_matrix(tree, max_distance=radius, output_type='coo_matrix')
    print(f"  Adjacency built in {time.time()-t0:.1f}s ({adj.nnz:,} edges)")

    # Find connected components
    print(f"  Finding connected components...")
    t0 = time.time()
    n_components, labels = connected_components(adj, directed=False)
    print(f"  Found {n_components:,} components in {time.time()-t0:.1f}s")

    # Find the largest component
    unique, counts = np.unique(labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]
    largest_count = counts[np.argmax(counts)]

    # Show top components
    sorted_idx = np.argsort(-counts)
    print(f"  Top components:")
    for rank, idx in enumerate(sorted_idx[:10]):
        pct = 100 * counts[idx] / n
        marker = " <-- keeping" if unique[idx] == largest_label else ""
        print(f"    #{rank+1}: {counts[idx]:>10,} splats ({pct:>5.1f}%){marker}")
    if n_components > 10:
        rest = sum(counts[sorted_idx[10:]])
        print(f"    ... +{n_components - 10} more components: {rest:,} splats")

    return labels, unique, counts, sorted_idx


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
    dists, _ = tree.query(positions, k=k+1, workers=-1)
    dists = dists[:, 1:]  # drop self-distance
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


def parse_hue_range(s):
    """Parse 'min,max' hue string to (float, float)."""
    parts = s.split(',')
    if len(parts) != 2:
        print(f"  ERROR: hue range must be min,max (got '{s}')")
        sys.exit(1)
    return float(parts[0]), float(parts[1])


def main():
    parser = argparse.ArgumentParser(
        description="Clean gaussian splat PLY by removing outliers and unwanted colors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Color filtering examples:
  --remove-hue 180,300               Remove blue/cyan/purple splats
  --remove-hue 180,300 --sat-min 0.3 Remove only saturated blue (spare grays)
  --keep-hue 30,160                  Keep only yellow/green splats
  --color-stats                      Print color histogram without filtering
        """,
    )
    parser.add_argument("input", help="Input .ply file")
    parser.add_argument("output", nargs='?', help="Output .ply file (default: input_cleaned.ply)")

    # Spatial filters
    parser.add_argument("-k", "--neighbors", type=int, default=20, help="K nearest neighbors (default: 20)")
    parser.add_argument("-s", "--std-factor", type=float, default=2.0, help="Std dev threshold (default: 2.0)")
    parser.add_argument("-p", "--passes", type=int, default=3, help="Iterative SOR passes (default: 3)")
    parser.add_argument("--bbox", type=str, default=None, help="Bounding box: xmin,ymin,zmin,xmax,ymax,zmax")
    parser.add_argument("--connected", action="store_true",
                        help="Keep only the largest spatially connected component(s) (after all other filters)")
    parser.add_argument("--connect-radius", type=float, default=0.0,
                        help="Radius for connectivity test (default: auto from mean KNN dist)")
    parser.add_argument("--keep-components", type=int, default=1,
                        help="Number of largest components to keep (default: 1)")

    # Opacity/scale filters
    parser.add_argument("-o", "--opacity-min", type=float, default=0.0, help="Min opacity threshold (default: off)")
    parser.add_argument("-r", "--scale-max", type=float, default=0.0, help="Max scale threshold (default: off)")

    # Color filters
    parser.add_argument("--remove-hue", type=str, default=None,
                        help="Remove splats in hue range: min,max (degrees 0-360). E.g. '180,300' for blue")
    parser.add_argument("--keep-hue", type=str, default=None,
                        help="Keep ONLY splats in hue range: min,max. E.g. '30,160' for yellow-green")
    parser.add_argument("--sat-min", type=float, default=0.0,
                        help="Min saturation for color filter to apply (0-1). Spares gray/neutral splats. Default: 0")
    parser.add_argument("--val-min", type=float, default=0.0,
                        help="Min value/brightness for color filter (0-1). Spares dark splats. Default: 0")
    parser.add_argument("--color-stats", action="store_true",
                        help="Print color histogram and stats (can combine with --dry-run)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dry-run", action="store_true", help="Report only, don't write output")
    args = parser.parse_args()

    if not args.output and not args.dry_run:
        base = args.input.rsplit('.', 1)[0]
        args.output = f"{base}_cleaned.ply"

    plydata, vertex = load_ply(args.input)
    positions = get_positions(vertex)
    original_count = len(positions)

    keep_mask = np.ones(len(positions), dtype=bool)

    # --- Color stats (early, before any filtering) ---
    rgb = None
    hsv = None
    if args.remove_hue or args.keep_hue or args.color_stats:
        rgb = get_rgb(vertex)
        if rgb is not None:
            print(f"\n=== Color Analysis ===")
            hsv = rgb_to_hsv(rgb)
            mean_rgb = np.mean(rgb, axis=0)
            print(f"  Mean RGB: ({mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f})")
            print_color_histogram(hsv)
        else:
            print("  WARNING: No SH DC color fields (f_dc_0/1/2) found in PLY")

    # --- Pre-filters ---

    # Bounding box clip
    if args.bbox:
        print(f"\n=== Bounding Box Clip ===")
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

    # Color filter
    if hsv is not None and (args.remove_hue or args.keep_hue):
        print(f"\n=== Color Filter ===")
        h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

        # Only apply color filter to splats with sufficient saturation/brightness.
        # Gray/dark splats don't have meaningful hue — spare them.
        color_eligible = np.ones(len(h), dtype=bool)
        if args.sat_min > 0:
            color_eligible &= s >= args.sat_min
            print(f"  Saturation threshold: >= {args.sat_min} ({np.sum(color_eligible & keep_mask):,} eligible)")
        if args.val_min > 0:
            color_eligible &= v >= args.val_min
            print(f"  Value threshold: >= {args.val_min} ({np.sum(color_eligible & keep_mask):,} eligible)")

        if args.remove_hue:
            hue_min, hue_max = parse_hue_range(args.remove_hue)
            # Handle wrap-around (e.g. 350,30 means red)
            if hue_min <= hue_max:
                in_range = (h >= hue_min) & (h <= hue_max)
            else:
                in_range = (h >= hue_min) | (h <= hue_max)

            to_remove = in_range & color_eligible & keep_mask
            n_remove = np.sum(to_remove)
            print(f"  Removing hue [{hue_min:.0f}, {hue_max:.0f}]: {n_remove:,} splats")
            keep_mask &= ~to_remove

        if args.keep_hue:
            hue_min, hue_max = parse_hue_range(args.keep_hue)
            if hue_min <= hue_max:
                in_range = (h >= hue_min) & (h <= hue_max)
            else:
                in_range = (h >= hue_min) | (h <= hue_max)

            # Keep: splats in range, OR not color-eligible (gray/dark get a pass)
            to_remove = ~in_range & color_eligible & keep_mask
            n_remove = np.sum(to_remove)
            print(f"  Keeping hue [{hue_min:.0f}, {hue_max:.0f}]: removing {n_remove:,} out-of-range splats")
            keep_mask &= ~to_remove

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
        removed_indices = current_indices[~pass_keep]
        keep_mask[removed_indices] = False

    # --- Connected component filter ---
    # Runs last: after SOR has thinned the field, keep only the main cluster.
    if args.connected:
        print(f"\n=== Connected Component Filter ===")
        current_indices = np.where(keep_mask)[0]
        current_positions = positions[current_indices]

        if len(current_positions) > 0:
            # Auto-detect radius from mean KNN distance if not specified
            radius = args.connect_radius
            if radius <= 0:
                from scipy.spatial import cKDTree
                print(f"  Auto-detecting connectivity radius...")
                tree = cKDTree(current_positions)
                sample_size = min(50000, len(current_positions))
                rng = np.random.default_rng(seed=args.seed)
                sample_idx = rng.choice(len(current_positions), sample_size, replace=False)
                dists, _ = tree.query(current_positions[sample_idx], k=2, workers=-1)
                nn_dist = np.median(dists[:, 1])
                # Use 3x median nearest-neighbor distance as connectivity radius
                radius = nn_dist * 3.0
                print(f"  Median NN dist: {nn_dist:.4f}, using radius: {radius:.4f}")

            labels, unique, counts, sorted_idx = connected_component_filter(current_positions, radius)
            n_keep = args.keep_components
            keep_labels = set(unique[sorted_idx[:n_keep]])
            kept_count = sum(counts[sorted_idx[:n_keep]])
            n_removed = len(current_positions) - kept_count
            print(f"  Keeping top {n_keep} component(s) ({kept_count:,} splats), removing {n_removed:,} ({100*n_removed/len(current_positions):.1f}%)")
            cc_keep = np.isin(labels, list(keep_labels))
            removed_indices = current_indices[~cc_keep]
            keep_mask[removed_indices] = False

    # --- Post-filter color stats ---
    if args.color_stats and hsv is not None:
        print(f"\n=== Color Distribution After Filtering ===")
        print_color_histogram(hsv, keep_mask)

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
