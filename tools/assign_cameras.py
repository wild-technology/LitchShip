#!/usr/bin/env python3
"""
Assign COLMAP camera models by filename prefix.

After feature extraction, COLMAP assigns one camera per image (or one global
camera). For multi-camera ROV rigs, we want images from the same physical
camera to share a camera model while different cameras get separate models.

This script reads a camera config file and reassigns camera IDs in the
COLMAP database so that all images matching a prefix share one camera
with the correct model and initial focal length.

Config file format (semicolon-delimited, # comments):
    camlower;OPENCV;16;36
    cammid;OPENCV_FISHEYE;12;36
    HERC;OPENCV;24;36

Fields: prefix;camera_model;focal_mm;sensor_width_mm

Usage:
    python3 assign_cameras.py <database.db> <camera_config.conf>
    python3 assign_cameras.py <database.db> <camera_config.conf> --preview

Requires: COLMAP database with images already registered (run feature_extractor first).
"""

import argparse
import sqlite3
import sys
from pathlib import Path


COLMAP_CAMERA_MODELS = {
    "SIMPLE_PINHOLE": 0,
    "PINHOLE": 1,
    "SIMPLE_RADIAL": 2,
    "RADIAL": 3,
    "OPENCV": 4,
    "OPENCV_FISHEYE": 5,
    "FULL_OPENCV": 6,
    "SIMPLE_RADIAL_FISHEYE": 8,
    "RADIAL_FISHEYE": 9,
}

# Number of parameters per model
CAMERA_MODEL_PARAMS = {
    "SIMPLE_PINHOLE": 3,   # f, cx, cy
    "PINHOLE": 4,          # fx, fy, cx, cy
    "SIMPLE_RADIAL": 4,    # f, cx, cy, k
    "RADIAL": 5,           # f, cx, cy, k1, k2
    "OPENCV": 8,           # fx, fy, cx, cy, k1, k2, p1, p2
    "OPENCV_FISHEYE": 8,   # fx, fy, cx, cy, k1, k2, k3, k4
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign COLMAP camera models by filename prefix"
    )
    parser.add_argument("database", help="Path to COLMAP database.db")
    parser.add_argument("config", help="Camera config file (prefix;model;focal_mm;sensor_width_mm)")
    parser.add_argument("--preview", action="store_true", help="Preview without writing")
    parser.add_argument("--batch", action="store_true", help="Non-interactive mode")
    return parser.parse_args()


def load_config(config_path):
    """Load camera config file. Returns list of (prefix, model, focal_mm, sensor_width_mm)."""
    configs = []
    with open(config_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 4:
                print(f"WARNING: Skipping malformed line: {line}", file=sys.stderr)
                continue
            prefix = parts[0].strip()
            model = parts[1].strip().upper()
            focal_mm = float(parts[2].strip())
            sensor_width_mm = float(parts[3].strip())
            if model not in COLMAP_CAMERA_MODELS:
                print(f"ERROR: Unknown camera model '{model}'. Valid: {list(COLMAP_CAMERA_MODELS.keys())}", file=sys.stderr)
                sys.exit(1)
            configs.append((prefix, model, focal_mm, sensor_width_mm))
    return configs


def get_image_dimensions(conn):
    """Get image dimensions from the first camera entry."""
    row = conn.execute("SELECT width, height FROM cameras LIMIT 1").fetchone()
    if row:
        return row[0], row[1]
    return None, None


def assign_cameras(db_path, configs, preview=False, batch=False):
    conn = sqlite3.connect(db_path)

    # Get all images
    images = conn.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    if not images:
        print("ERROR: No images in database", file=sys.stderr)
        sys.exit(1)

    # Get image dimensions from existing camera
    width, height = get_image_dimensions(conn)
    if width is None:
        print("ERROR: No camera entries found — run feature_extractor first", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Database:   {db_path}")
    print(f"  Images:     {len(images)}")
    print(f"  Resolution: {width}x{height}")
    print(f"  Config:     {len(configs)} camera definitions")
    print()

    # Match images to config entries by prefix
    assignments = {}  # image_id -> (config_index, prefix)
    unmatched = []

    for image_id, name, camera_id in images:
        matched = False
        for idx, (prefix, model, focal_mm, sensor_width_mm) in enumerate(configs):
            if prefix in name:
                assignments[image_id] = (idx, name)
                matched = True
                break
        if not matched:
            unmatched.append(name)

    # Report per-camera counts
    camera_counts = {}
    for image_id, (idx, name) in assignments.items():
        camera_counts[idx] = camera_counts.get(idx, 0) + 1

    print("  Camera assignments:")
    for idx, (prefix, model, focal_mm, sensor_width_mm) in enumerate(configs):
        focal_px = focal_mm * width / sensor_width_mm
        count = camera_counts.get(idx, 0)
        print(f"    {prefix:12s}  {model:16s}  focal={focal_mm}mm ({focal_px:.0f}px)  → {count} images")

    if unmatched:
        print(f"\n  Unmatched images: {len(unmatched)}")
        for name in unmatched[:5]:
            print(f"    - {name}")
        if len(unmatched) > 5:
            print(f"    ... and {len(unmatched) - 5} more")

    if preview:
        print("\n  [Preview mode — no changes written]")
        conn.close()
        return

    if not batch:
        reply = input(f"\n  Reassign {len(assignments)} images to {len(camera_counts)} cameras? [Y/n] ").strip()
        if reply.lower() in ("n", "no"):
            print("  Aborted.")
            conn.close()
            return

    # Create new camera entries (one per config)
    # First, find the max existing camera_id
    max_cam_id = conn.execute("SELECT MAX(camera_id) FROM cameras").fetchone()[0] or 0

    new_camera_ids = {}
    for idx, (prefix, model, focal_mm, sensor_width_mm) in enumerate(configs):
        if idx not in camera_counts:
            continue  # No images match this config

        new_cam_id = max_cam_id + idx + 1
        focal_px = focal_mm * width / sensor_width_mm
        cx = width / 2.0
        cy = height / 2.0

        model_id = COLMAP_CAMERA_MODELS[model]
        num_params = CAMERA_MODEL_PARAMS[model]

        if model == "SIMPLE_PINHOLE":
            params_blob = _pack_params([focal_px, cx, cy])
        elif model == "PINHOLE":
            params_blob = _pack_params([focal_px, focal_px, cx, cy])
        elif model == "SIMPLE_RADIAL":
            params_blob = _pack_params([focal_px, cx, cy, 0.0])
        elif model == "RADIAL":
            params_blob = _pack_params([focal_px, cx, cy, 0.0, 0.0])
        elif model in ("OPENCV", "OPENCV_FISHEYE"):
            params_blob = _pack_params([focal_px, focal_px, cx, cy, 0.0, 0.0, 0.0, 0.0])
        else:
            params_blob = _pack_params([focal_px, cx, cy])

        conn.execute(
            "INSERT INTO cameras (camera_id, model, width, height, params, prior_focal_length) VALUES (?, ?, ?, ?, ?, 1)",
            (new_cam_id, model_id, width, height, params_blob),
        )
        new_camera_ids[idx] = new_cam_id

    # Reassign images and rebuild rig/frame/frame_data from scratch.
    # COLMAP 4.x model:
    #   - One rig per unique camera (ref_sensor_id = camera_id, UNIQUE constraint)
    #   - Each image gets a frame under its camera's rig
    #   - frame_data links frame → image with sensor_id = camera_id

    # Update image → camera assignments
    for image_id, (idx, name) in assignments.items():
        if idx in new_camera_ids:
            conn.execute(
                "UPDATE images SET camera_id=? WHERE image_id=?",
                (new_camera_ids[idx], image_id),
            )

    # Rebuild rig/frame/frame_data tables
    conn.execute("DELETE FROM frame_data")
    conn.execute("DELETE FROM frames")
    conn.execute("DELETE FROM rig_sensors")
    conn.execute("DELETE FROM rigs")

    # Create one trivial rig per camera (not per image)
    for idx, cam_id in new_camera_ids.items():
        conn.execute(
            "INSERT INTO rigs (rig_id, ref_sensor_id, ref_sensor_type) VALUES (?, ?, 0)",
            (cam_id, cam_id),
        )

    # Create frames and frame_data for each image
    for image_id, (idx, name) in assignments.items():
        if idx not in new_camera_ids:
            continue
        cam_id = new_camera_ids[idx]
        conn.execute(
            "INSERT INTO frames (frame_id, rig_id) VALUES (?, ?)",
            (image_id, cam_id),
        )
        conn.execute(
            "INSERT INTO frame_data (frame_id, data_id, sensor_id, sensor_type) VALUES (?, ?, ?, 0)",
            (image_id, image_id, cam_id),
        )

    # Clean up orphaned camera entries
    conn.execute(
        "DELETE FROM cameras WHERE camera_id NOT IN (SELECT DISTINCT camera_id FROM images)"
    )

    conn.commit()
    conn.close()

    print(f"\n  Assigned {len(assignments)} images to {len(new_camera_ids)} camera models")
    print(f"  Rebuilt rigs ({len(new_camera_ids)}), frames, and frame_data for COLMAP 4.x")
    print(f"  Cleaned up orphaned camera entries")


def _pack_params(params):
    """Pack camera parameters as a binary blob (doubles)."""
    import struct
    return struct.pack(f"<{len(params)}d", *params)


def main():
    args = parse_args()

    db_path = Path(args.database)
    config_path = Path(args.config)

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    configs = load_config(str(config_path))
    if not configs:
        print("ERROR: No camera entries in config file", file=sys.stderr)
        sys.exit(1)

    assign_cameras(str(db_path), configs, preview=args.preview, batch=args.batch)


if __name__ == "__main__":
    main()
