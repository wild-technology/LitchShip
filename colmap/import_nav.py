#!/usr/bin/env python3
"""
Import navigation/pose priors into a COLMAP 4.x database.

Writes position priors and gravity vectors (from pitch/roll) into the
pose_priors table. Gravity priors encode the "down" direction in the camera
coordinate frame, enabling gravity-aligned rotation averaging in the
global mapper.

COLMAP 4.x pose_priors schema:
    pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type,
    position (BLOB), position_covariance (BLOB), gravity (BLOB),
    coordinate_system (INTEGER)

Coordinate systems: UNDEFINED=-1, WGS84=0, CARTESIAN=1
For UTM coordinates, use CARTESIAN (1).

Supported input formats:

  ROV nav export (semicolon-delimited, UTM):
    filename;X (East);Y (North);Alt;X Accuracy;Y Accuracy;Alt Accuracy;
    Yaw;Pitch;Roll;Yaw Accuracy;Pitch Accuracy;Roll Accuracy;FocalLength

  Simple GPS (comma-delimited):
    image_name,lat,lon,alt

Usage:
    python3 import_gps_to_colmap.py <database.db> <nav_file>
    python3 import_gps_to_colmap.py <database.db> <nav_file> --preview
    python3 import_gps_to_colmap.py <database.db> <nav_file> --no-gravity

Requires: COLMAP database with images registered (run feature_extractor first).
"""

import argparse
import csv
import math
import os
import sqlite3
import struct
import sys
from pathlib import Path


# ─── Constants ───────────────────────────────────────────────────────────────

# COLMAP CoordinateSystem enum (4.x)
CS_UNDEFINED = -1
CS_WGS84 = 0
CS_CARTESIAN = 1

CS_NAMES = {-1: "UNDEFINED", 0: "WGS84 (lat/lon/alt)", 1: "CARTESIAN (metric)"}

# COLMAP SensorType enum
SENSOR_TYPE_CAMERA = 0


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import nav/pose priors into COLMAP 4.x database"
    )
    parser.add_argument("database", help="Path to COLMAP database.db")
    parser.add_argument("nav_file", help="Nav file with positions and optional orientation")
    parser.add_argument(
        "--delimiter", default="auto",
        help="CSV delimiter: auto, comma, semicolon, tab",
    )
    parser.add_argument(
        "--coordinate-system", type=int, default=-99,
        help="0=WGS84, 1=CARTESIAN (for UTM/metric). Default: auto-detect",
    )
    parser.add_argument(
        "--no-gravity", action="store_true",
        help="Skip gravity vector import even if pitch/roll available",
    )
    parser.add_argument("--preview", action="store_true", help="Preview without writing")
    parser.add_argument("--batch", action="store_true", help="Non-interactive mode")
    parser.add_argument("--quiet", action="store_true", help="Suppress detail output")
    return parser.parse_args()


# ─── Gravity Computation ────────────────────────────────────────────────────

def pitch_roll_to_gravity(pitch_deg, roll_deg):
    """
    Compute gravity (down) direction in COLMAP camera frame from pitch/roll.

    COLMAP camera convention: X-right, Y-down, Z-forward.
    Navigation convention assumed: pitch = degrees below horizontal
    (0=level, 90=straight down), roll = bank angle.

    When camera is level (pitch=0, roll=0): gravity = [0, 1, 0] (Y-down)
    When camera looks straight down (pitch=90): gravity = [0, 0, 1] (Z-forward)
    """
    p = math.radians(pitch_deg)
    r = math.radians(roll_deg)

    gx = -math.sin(r) * math.cos(p)
    gy = math.cos(r) * math.cos(p)
    gz = math.sin(p)

    # Normalize (should already be unit but ensure numerical stability)
    norm = math.sqrt(gx * gx + gy * gy + gz * gz)
    if norm > 0:
        gx, gy, gz = gx / norm, gy / norm, gz / norm

    return gx, gy, gz


# ─── Helpers ─────────────────────────────────────────────────────────────────

def detect_delimiter(first_line):
    for delim, name in [(";", "semicolon"), (",", "comma"), ("\t", "tab")]:
        if delim in first_line:
            return delim, name
    return ",", "comma"


def map_columns(fieldnames):
    col_map = {}
    for col in fieldnames:
        c = col.strip().lower()
        if c in ("filename", "image_name", "image", "name", "file"):
            col_map["name"] = col
        elif c in ("x (east)", "easting", "x_east", "utm_e", "east"):
            col_map["x"] = col
        elif c in ("y (north)", "northing", "y_north", "utm_n", "north"):
            col_map["y"] = col
        elif c in ("lat", "latitude"):
            col_map["lat"] = col
        elif c in ("lon", "lng", "longitude"):
            col_map["lon"] = col
        elif c in ("alt", "altitude", "depth", "z", "elevation"):
            col_map["alt"] = col
        elif c == "yaw":
            col_map["yaw"] = col
        elif c == "pitch":
            col_map["pitch"] = col
        elif c == "roll":
            col_map["roll"] = col
        elif c in ("x accuracy", "x_accuracy"):
            col_map["x_acc"] = col
        elif c in ("y accuracy", "y_accuracy"):
            col_map["y_acc"] = col
        elif c in ("alt accuracy", "alt_accuracy"):
            col_map["alt_acc"] = col
    return col_map


def pack_vec3(x, y, z):
    return struct.pack("<3d", x, y, z)


def pack_mat3x3_diagonal(sx, sy, sz):
    return struct.pack(
        "<9d",
        sx * sx, 0.0, 0.0,
        0.0, sy * sy, 0.0,
        0.0, 0.0, sz * sz,
    )


NAN_VEC3 = pack_vec3(float("nan"), float("nan"), float("nan"))


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_nav_data(nav_file, delimiter_arg):
    with open(nav_file, newline="") as f:
        first_line = f.readline()

    if delimiter_arg == "auto":
        delimiter, delim_name = detect_delimiter(first_line)
    else:
        delimiter = {"comma": ",", "semicolon": ";", "tab": "\t"}.get(
            delimiter_arg, delimiter_arg
        )
        delim_name = delimiter_arg

    print(f"  Delimiter:   {delim_name} ({repr(delimiter)})")

    entries = {}
    with open(nav_file, newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            print("ERROR: File appears empty or has no header", file=sys.stderr)
            sys.exit(1)

        col_map = map_columns(reader.fieldnames)

        if "x" in col_map and "y" in col_map:
            coord_type = "utm"
        elif "lat" in col_map and "lon" in col_map:
            coord_type = "gps"
        else:
            print(f"ERROR: Cannot find position columns.\n  Found: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        if "name" not in col_map:
            print(f"ERROR: Cannot find image name column.\n  Found: {reader.fieldnames}", file=sys.stderr)
            sys.exit(1)

        has_orientation = all(k in col_map for k in ("yaw", "pitch", "roll"))

        for row in reader:
            name = row[col_map["name"]].strip()

            if coord_type == "utm":
                x = float(row[col_map["x"]].strip())
                y = float(row[col_map["y"]].strip())
            else:
                x = float(row[col_map["lon"]].strip())
                y = float(row[col_map["lat"]].strip())

            alt = 0.0
            if "alt" in col_map and row[col_map["alt"]].strip():
                alt = float(row[col_map["alt"]].strip())

            entry = {"x": x, "y": y, "alt": alt}

            if has_orientation:
                entry["yaw"] = float(row[col_map["yaw"]].strip())
                entry["pitch"] = float(row[col_map["pitch"]].strip())
                entry["roll"] = float(row[col_map["roll"]].strip())

            for acc_key in ("x_acc", "y_acc", "alt_acc"):
                if acc_key in col_map and row[col_map[acc_key]].strip():
                    entry[acc_key] = float(row[col_map[acc_key]].strip())

            entries[name] = entry

    return entries, coord_type, has_orientation


def get_db_images(db_path):
    """Get image_id, name, and camera_id from database.
    Returns two dicts for matching: full_name and basename -> (image_id, camera_id).
    COLMAP may store paths with directory prefixes (e.g., from symlinks).
    """
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    conn.close()
    # Index by both full name and basename for flexible matching
    by_full = {}
    by_basename = {}
    for image_id, name, camera_id in rows:
        by_full[name] = (image_id, camera_id)
        basename = os.path.basename(name)
        by_basename[basename] = (image_id, camera_id)
    return by_full, by_basename


# ─── Write Priors ────────────────────────────────────────────────────────────

def write_priors(db_path, db_images, nav_entries, coord_type, has_orientation,
                 coordinate_system, import_gravity=True, nav_filename="",
                 preview=False, batch=False, quiet=False):

    matched = []
    unmatched_nav = []
    unmatched_db = []

    for name, entry in nav_entries.items():
        if name in db_images:
            image_id, camera_id = db_images[name]
            matched.append((name, image_id, camera_id, entry))
        else:
            unmatched_nav.append(name)

    for name in db_images:
        if name not in nav_entries:
            unmatched_db.append(name)

    use_gravity = has_orientation and import_gravity

    # Auto-detect coordinate system
    if coordinate_system == -99:
        if coord_type == "gps":
            coordinate_system = CS_WGS84
        else:
            coordinate_system = CS_CARTESIAN  # UTM/projected → CARTESIAN

    cs_name = CS_NAMES.get(coordinate_system, f"Unknown ({coordinate_system})")

    # Report
    print(f"\n  Coordinate system: {cs_name} ({coordinate_system})")
    print(f"  Gravity priors:   {'pitch/roll → gravity vector' if use_gravity else 'disabled'}")
    print(f"  Nav entries:       {len(nav_entries)}")
    print(f"  Database images:   {len(db_images)}")
    print(f"  Matched:           {len(matched)}")
    if unmatched_nav and not quiet:
        print(f"  Nav without DB:    {len(unmatched_nav)}")
        for name in unmatched_nav[:5]:
            print(f"    - {name}")
        if len(unmatched_nav) > 5:
            print(f"    ... and {len(unmatched_nav) - 5} more")
    if unmatched_db and not quiet:
        print(f"  DB without nav:    {len(unmatched_db)}")
        for name in unmatched_db[:5]:
            print(f"    - {name}")
        if len(unmatched_db) > 5:
            print(f"    ... and {len(unmatched_db) - 5} more")

    if not matched:
        print("\nERROR: No matches between nav file and database", file=sys.stderr)
        sys.exit(1)

    # Statistics
    xs = [e["x"] for _, _, _, e in matched]
    ys = [e["y"] for _, _, _, e in matched]
    alts = [e["alt"] for _, _, _, e in matched]
    print(f"\n  Position range:")
    print(f"    X: {min(xs):.2f} — {max(xs):.2f}  (span: {max(xs)-min(xs):.2f} m)")
    print(f"    Y: {min(ys):.2f} — {max(ys):.2f}  (span: {max(ys)-min(ys):.2f} m)")
    print(f"    Z: {min(alts):.2f} — {max(alts):.2f}  (span: {max(alts)-min(alts):.2f} m)")

    if use_gravity:
        pitches = [e["pitch"] for _, _, _, e in matched]
        rolls = [e["roll"] for _, _, _, e in matched]
        print(f"  Orientation:")
        print(f"    Pitch: {min(pitches):.1f}° — {max(pitches):.1f}°")
        print(f"    Roll:  {min(rolls):.1f}° — {max(rolls):.1f}°")
        # Show sample gravity vector
        gx, gy, gz = pitch_roll_to_gravity(pitches[0], rolls[0])
        print(f"    Sample gravity: ({gx:.4f}, {gy:.4f}, {gz:.4f})")

    has_accuracy = "x_acc" in matched[0][3]
    if has_accuracy:
        x_accs = [e.get("x_acc", 10.0) for _, _, _, e in matched]
        print(f"  Position accuracy: {min(x_accs):.1f} — {max(x_accs):.1f} m")

    if preview:
        print("\n  Preview (first 5):")
        for name, image_id, camera_id, entry in matched[:5]:
            line = f"    {name} (img={image_id}, cam={camera_id}): "
            line += f"x={entry['x']:.2f} y={entry['y']:.2f} z={entry['alt']:.2f}"
            if use_gravity:
                gx, gy, gz = pitch_roll_to_gravity(entry["pitch"], entry["roll"])
                line += f"  g=({gx:.3f},{gy:.3f},{gz:.3f})"
            print(line)
        print(f"\n  Target: pose_priors table (COLMAP 4.x schema)")
        print("  [Preview mode — no changes written]")
        return

    if not batch:
        what = "position + gravity" if use_gravity else "position"
        reply = input(f"\n  Write {len(matched)} {what} priors? [Y/n] ").strip()
        if reply.lower() in ("n", "no"):
            print("  Aborted.")
            return

    # Write
    conn = sqlite3.connect(db_path)

    # Clear existing pose_priors
    conn.execute("DELETE FROM pose_priors")

    for idx, (name, image_id, camera_id, entry) in enumerate(matched):
        position_blob = pack_vec3(entry["x"], entry["y"], entry["alt"])

        if has_accuracy:
            cov_blob = pack_mat3x3_diagonal(
                entry.get("x_acc", 10.0),
                entry.get("y_acc", 10.0),
                entry.get("alt_acc", 1.0),
            )
        else:
            cov_blob = pack_mat3x3_diagonal(10.0, 10.0, 1.0)

        if use_gravity:
            gx, gy, gz = pitch_roll_to_gravity(entry["pitch"], entry["roll"])
            gravity_blob = pack_vec3(gx, gy, gz)
        else:
            gravity_blob = NAN_VEC3

        # corr_data_id = image_id (the "data" for a camera sensor is an image)
        # corr_sensor_id = camera_id
        # corr_sensor_type = 0 (CAMERA)
        conn.execute(
            "INSERT OR REPLACE INTO pose_priors "
            "(pose_prior_id, corr_data_id, corr_sensor_id, corr_sensor_type, "
            " position, position_covariance, gravity, coordinate_system) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (idx + 1, image_id, camera_id, SENSOR_TYPE_CAMERA,
             position_blob, cov_blob, gravity_blob, coordinate_system),
        )

    conn.commit()
    conn.close()

    what = "position + gravity" if use_gravity else "position"
    print(f"\n  Written {len(matched)} {what} priors to pose_priors table")
    print(f"  Coordinate system: {cs_name}")
    if use_gravity:
        print("  Gravity priors enable rotation averaging in global_mapper")
    print()
    print("  Recommended matcher:")
    print("    colmap spatial_matcher --SpatialMatching.max_distance 50")
    print()
    print("  Recommended mapper (uses gravity):")
    print("    colmap global_mapper --database_path ... --image_path ... --output_path ...")


def main():
    args = parse_args()
    db_path = Path(args.database)
    nav_path = Path(args.nav_file)

    if not db_path.exists():
        print(f"ERROR: Database not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    if not nav_path.exists():
        print(f"ERROR: Nav file not found: {nav_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\n  Database: {db_path}")
    print(f"  Nav file: {nav_path}")

    nav_entries, coord_type, has_orientation = load_nav_data(str(nav_path), args.delimiter)
    db_full, db_basename = get_db_images(str(db_path))
    # Merge: prefer full name match, fall back to basename
    db_images = {**db_basename, **db_full}
    write_priors(
        str(db_path), db_images, nav_entries, coord_type, has_orientation,
        coordinate_system=args.coordinate_system,
        import_gravity=not args.no_gravity,
        nav_filename=str(nav_path),
        preview=args.preview, batch=args.batch, quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
