#!/usr/bin/env python3
"""
Generate a PDF benchmark report from a completed COLMAP reconstruction pipeline.

Reads the COLMAP database, sparse model output, log files, and timing data
to produce a comprehensive report with benchmarks, statistics, and
efficiency recommendations.

Usage:
    python3 generate_report.py <output_dir> [--report-path report.pdf]

The output_dir should be the workspace directory produced by colmap_reconstruct.sh,
containing database.db, sparse/0/, and logs/.
"""

import argparse
import glob
import os
import re
import sqlite3
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from fpdf import FPDF
except ImportError:
    print("ERROR: fpdf2 not installed. Run: pip install fpdf2", file=sys.stderr)
    sys.exit(1)


# ─── Data Collection ─────────────────────────────────────────────────────────

def collect_db_stats(db_path):
    """Query COLMAP database for reconstruction statistics."""
    stats = {}
    if not os.path.exists(db_path):
        return stats

    conn = sqlite3.connect(db_path)

    stats["images"] = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    # Camera models
    rows = conn.execute(
        "SELECT model, COUNT(*) FROM cameras GROUP BY model"
    ).fetchall()
    model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                   3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
                   6: "FULL_OPENCV", 8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE"}
    stats["cameras"] = [(model_names.get(m, f"UNKNOWN({m})"), c) for m, c in rows]

    # Features
    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(rows), 0), COALESCE(AVG(rows), 0), "
        "COALESCE(MIN(rows), 0), COALESCE(MAX(rows), 0) FROM keypoints WHERE rows > 0"
    ).fetchone()
    stats["keypoints_images"] = row[0]
    stats["total_features"] = row[1]
    stats["avg_features"] = int(row[2])
    stats["min_features"] = row[3]
    stats["max_features"] = row[4]

    # Matches
    stats["total_matches"] = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    stats["verified_pairs"] = conn.execute(
        "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
    ).fetchone()[0]

    # Pose priors
    stats["pose_priors"] = conn.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]

    # Rigs
    stats["rigs"] = conn.execute("SELECT COUNT(*) FROM rigs").fetchone()[0]

    # Position statistics from pose_priors
    try:
        rows = conn.execute("SELECT position FROM pose_priors WHERE position IS NOT NULL").fetchall()
        if rows:
            positions = []
            for (blob,) in rows:
                if blob and len(blob) >= 24:
                    x, y, z = struct.unpack("<3d", blob[:24])
                    positions.append((x, y, z))
            if positions:
                xs = [p[0] for p in positions]
                ys = [p[1] for p in positions]
                zs = [p[2] for p in positions]
                stats["pos_x_range"] = (min(xs), max(xs))
                stats["pos_y_range"] = (min(ys), max(ys))
                stats["pos_z_range"] = (min(zs), max(zs))
                stats["survey_area_m2"] = (max(xs) - min(xs)) * (max(ys) - min(ys))
    except Exception:
        pass

    conn.close()
    return stats


def collect_model_stats(sparse_dir):
    """Parse sparse model text files for reconstruction statistics."""
    stats = {}
    images_txt = os.path.join(sparse_dir, "images.txt")
    points_txt = os.path.join(sparse_dir, "points3D.txt")
    cameras_txt = os.path.join(sparse_dir, "cameras.txt")

    if os.path.exists(images_txt):
        with open(images_txt) as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        # images.txt has 2 lines per image
        stats["registered_images"] = len(lines) // 2

    if os.path.exists(points_txt):
        errors = []
        point_count = 0
        with open(points_txt) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 8:
                    point_count += 1
                    try:
                        errors.append(float(parts[7]))
                    except (ValueError, IndexError):
                        pass
        stats["total_points"] = point_count
        if errors:
            stats["reproj_error_mean"] = sum(errors) / len(errors)
            stats["reproj_error_median"] = sorted(errors)[len(errors) // 2]
            stats["reproj_error_max"] = max(errors)

    if os.path.exists(cameras_txt):
        camera_details = []
        with open(cameras_txt) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                parts = line.strip().split()
                if len(parts) >= 5:
                    cam_id = parts[0]
                    model = parts[1]
                    width = parts[2]
                    height = parts[3]
                    focal = parts[4]
                    camera_details.append({
                        "id": cam_id, "model": model,
                        "resolution": f"{width}x{height}",
                        "focal_px": float(focal),
                    })
        stats["camera_details"] = camera_details

    return stats


def collect_timing(log_dir, output_dir=None):
    """Collect timing from COLMAP log files.

    COLMAP 4.x writes flat log files like:
        colmap.<host>.<user>.log.INFO.<timestamp>.<pid>
    Each file corresponds to one COLMAP command invocation.
    Stage is identified from content markers.
    """
    timings = []
    if not log_dir or not os.path.isdir(log_dir):
        return timings

    # Stage detection patterns (searched in log content)
    stage_patterns = [
        (r"=== Feature extraction ===", "Feature Extraction"),
        (r"=== Feature matching", "Feature Matching"),
        (r"Running rotation averaging", "Global Mapper"),
        (r"Registering image", "Incremental Mapper"),
        (r"=== Image undistortion", "Image Undistortion"),
        (r"patch_match", "PatchMatch Stereo"),
        (r"stereo_fusion", "Stereo Fusion"),
        (r"Loading descriptors.*Building index", "Vocab Tree Building"),
    ]

    # Collect all INFO log files sorted by creation time
    log_files = sorted(
        [os.path.join(log_dir, f) for f in os.listdir(log_dir)
         if f.endswith(".log") or ".log.INFO." in f],
        key=lambda p: os.path.getmtime(p),
    )

    for fpath in log_files:
        try:
            with open(fpath, errors="replace") as f:
                content = f.read()

            # Find elapsed time
            elapsed_match = re.search(r"Elapsed time: ([\d.]+) \[minutes\]", content)
            if not elapsed_match:
                continue
            minutes = float(elapsed_match.group(1))

            # Determine stage from content
            stage_name = "Unknown Stage"
            for pattern, name in stage_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    stage_name = name
                    break

            timings.append({"stage": stage_name, "minutes": minutes})
        except Exception:
            pass
    return timings


def collect_hardware():
    """Gather hardware information."""
    hw = {}
    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                                 "--format=csv,noheader"], capture_output=True, text=True)
        if result.returncode == 0:
            hw["gpu"] = result.stdout.strip()
    except Exception:
        pass

    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    hw["cpu"] = line.split(":")[1].strip()
                    break
    except Exception:
        pass

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    hw["ram_gb"] = round(kb / 1024 / 1024)
                    break
    except Exception:
        pass

    hw["nproc"] = os.cpu_count()
    return hw


# ─── PDF Report ──────────────────────────────────────────────────────────────

class ReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "COLMAP Reconstruction Pipeline - Benchmark Report", align="R")
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} - Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def section_title(self, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(30, 80, 160)
        self.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 80, 160)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(4)

    def subsection(self, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def kv(self, key, value, bold_value=False):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(80, 80, 80)
        self.cell(70, 6, key)
        style = "B" if bold_value else ""
        self.set_font("Helvetica", style, 10)
        self.set_text_color(30, 30, 30)
        self.cell(0, 6, str(value), new_x="LMARGIN", new_y="NEXT")

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [self.epw / len(headers)] * len(headers)
        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True)
        self.ln()
        # Rows
        self.set_font("Helvetica", "", 9)
        for row in rows:
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6, str(val), border=1)
            self.ln()
        self.ln(3)

    def paragraph(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5, text)
        self.ln(2)


def generate_report(output_dir, report_path):
    db_path = os.path.join(output_dir, "database.db")
    sparse_dir = os.path.join(output_dir, "sparse", "0")
    log_dir = os.path.join(output_dir, "logs")

    db_stats = collect_db_stats(db_path)
    model_stats = collect_model_stats(sparse_dir)
    timings = collect_timing(log_dir, output_dir)
    hw = collect_hardware()

    pdf = ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ─── Title ───────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 60, 140)
    pdf.cell(0, 15, "COLMAP Reconstruction Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 12)
    pdf.set_text_color(100, 100, 100)
    # Dataset name: use the images symlink target or the output dir name
    images_link = os.path.join(output_dir, "images")
    if os.path.islink(images_link):
        dataset_name = os.path.basename(os.path.dirname(os.path.realpath(images_link)))
    else:
        dataset_name = os.path.basename(output_dir)
    pdf.cell(0, 8, f"Dataset: {dataset_name}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    # ─── Hardware ────────────────────────────────────────────────────────
    pdf.section_title("1. Hardware Configuration")
    pdf.kv("GPU", hw.get("gpu", "N/A"))
    pdf.kv("CPU", hw.get("cpu", "N/A"))
    pdf.kv("Threads", str(hw.get("nproc", "N/A")))
    pdf.kv("RAM", f"{hw.get('ram_gb', 'N/A')} GB")
    pdf.kv("COLMAP Version", "4.1.0.dev0 (with CUDA, FAISS, ONNX)")
    pdf.ln(5)

    # ─── Dataset Overview ────────────────────────────────────────────────
    pdf.section_title("2. Dataset Overview")
    pdf.kv("Total Images", f"{db_stats.get('images', 'N/A'):,}")
    pdf.kv("Pose Priors (GPS+Gravity)", f"{db_stats.get('pose_priors', 0):,}")
    pdf.kv("Rigs", str(db_stats.get("rigs", "N/A")))

    if "pos_x_range" in db_stats:
        x_span = db_stats["pos_x_range"][1] - db_stats["pos_x_range"][0]
        y_span = db_stats["pos_y_range"][1] - db_stats["pos_y_range"][0]
        z_span = db_stats["pos_z_range"][1] - db_stats["pos_z_range"][0]
        pdf.kv("Survey Extent (X)", f"{x_span:.1f} m")
        pdf.kv("Survey Extent (Y)", f"{y_span:.1f} m")
        pdf.kv("Depth Range", f"{db_stats['pos_z_range'][0]:.1f} to {db_stats['pos_z_range'][1]:.1f} m")
        area = db_stats.get("survey_area_m2", 0)
        if area > 0:
            pdf.kv("Survey Area", f"{area:.0f} sq.m ({area/10000:.2f} hectares)")

    pdf.ln(3)

    # Camera table
    pdf.subsection("Camera Models")
    if db_stats.get("cameras"):
        pdf.table(
            ["Model", "Count"],
            [(m, str(c)) for m, c in db_stats["cameras"]],
            col_widths=[120, 60],
        )

    if model_stats.get("camera_details"):
        pdf.table(
            ["ID", "Model", "Resolution", "Focal (px)"],
            [(c["id"], c["model"], c["resolution"], f"{c['focal_px']:.1f}")
             for c in model_stats["camera_details"]],
            col_widths=[30, 60, 50, 50],
        )

    # ─── Feature Extraction ──────────────────────────────────────────────
    pdf.section_title("3. Feature Extraction")
    pdf.kv("Images with Features", f"{db_stats.get('keypoints_images', 0):,}")
    pdf.kv("Total Features", f"{db_stats.get('total_features', 0):,}")
    pdf.kv("Average per Image", f"{db_stats.get('avg_features', 0):,}")
    pdf.kv("Min / Max per Image", f"{db_stats.get('min_features', 0):,} / {db_stats.get('max_features', 0):,}")
    feat_time = next((t["minutes"] for t in timings if "extraction" in t["stage"].lower() or "feature" in t["stage"].lower()), None)
    if feat_time:
        pdf.kv("Extraction Time", f"{feat_time:.1f} minutes")
        if db_stats.get("images", 0) > 0:
            pdf.kv("Speed", f"{db_stats['images'] / feat_time:.0f} images/min")
    pdf.ln(3)

    # ─── Matching ────────────────────────────────────────────────────────
    pdf.section_title("4. Feature Matching")
    pdf.kv("Total Match Pairs", f"{db_stats.get('total_matches', 0):,}")
    pdf.kv("Verified Pairs (geometric)", f"{db_stats.get('verified_pairs', 0):,}")
    if db_stats.get("total_matches", 0) > 0:
        ratio = db_stats.get("verified_pairs", 0) / db_stats["total_matches"] * 100
        pdf.kv("Verification Rate", f"{ratio:.1f}%")
    match_time = next((t["minutes"] for t in timings if "match" in t["stage"].lower()), None)
    if match_time:
        pdf.kv("Matching Time", f"{match_time:.1f} minutes")
    pdf.ln(3)

    # ─── Reconstruction ─────────────────────────────────────────────────
    pdf.section_title("5. Sparse Reconstruction")
    reg = model_stats.get("registered_images", "N/A")
    total = db_stats.get("images", 0)
    pdf.kv("Registered Images", f"{reg}", bold_value=True)
    if isinstance(reg, int) and total > 0:
        pdf.kv("Registration Rate", f"{reg/total*100:.1f}%")
    pdf.kv("3D Points", f"{model_stats.get('total_points', 'N/A'):,}" if isinstance(model_stats.get("total_points"), int) else "N/A")

    if "reproj_error_mean" in model_stats:
        pdf.kv("Reprojection Error (mean)", f"{model_stats['reproj_error_mean']:.3f} px")
        pdf.kv("Reprojection Error (median)", f"{model_stats['reproj_error_median']:.3f} px")
        pdf.kv("Reprojection Error (max)", f"{model_stats['reproj_error_max']:.3f} px")

    mapper_time = next((t["minutes"] for t in timings if "mapper" in t["stage"].lower() or "reconstruction" in t["stage"].lower()), None)
    if mapper_time:
        pdf.kv("Mapper Time", f"{mapper_time:.1f} minutes")
    pdf.ln(3)

    # ─── Timing Summary ─────────────────────────────────────────────────
    pdf.section_title("6. Timing Benchmarks")
    if timings:
        total_min = sum(t["minutes"] for t in timings)
        rows = [(t["stage"], f"{t['minutes']:.1f} min") for t in timings]
        rows.append(("TOTAL", f"{total_min:.1f} min ({total_min/60:.1f} hours)"))
        pdf.table(["Stage", "Duration"], rows, col_widths=[120, 70])

    # ─── Efficiency Analysis ─────────────────────────────────────────────
    pdf.add_page()
    pdf.section_title("7. Efficiency Analysis & Recommendations")

    recommendations = []

    # Feature extraction speed
    if feat_time and db_stats.get("images", 0) > 0:
        speed = db_stats["images"] / feat_time
        if speed < 50:
            recommendations.append(
                "Feature extraction is slower than expected. Verify GPU SIFT is being used "
                "(not CPU Covariant SIFT from domain_size_pooling). Check logs for 'SIFT GPU' vs 'Covariant SIFT CPU'."
            )
        else:
            recommendations.append(
                f"Feature extraction speed ({speed:.0f} img/min) is healthy for GPU SIFT on RTX 5090."
            )

    # Registration rate
    if isinstance(reg, int) and total > 0:
        rate = reg / total * 100
        if rate < 80:
            recommendations.append(
                f"Registration rate ({rate:.1f}%) is below 80%. Possible causes: insufficient matching overlap, "
                "poor GPS priors causing spatial matcher to miss pairs, or challenging underwater conditions. "
                "Consider: increasing --SpatialMatching.max_distance, using exhaustive matching on subsets, "
                "or trying sequential matching with loop detection."
            )
        elif rate < 95:
            recommendations.append(
                f"Registration rate ({rate:.1f}%) is good but some images are unregistered. "
                "These may be from areas with poor texture, heavy backscatter, or gaps in coverage."
            )
        else:
            recommendations.append(
                f"Registration rate ({rate:.1f}%) is excellent."
            )

    # Reprojection error
    if "reproj_error_mean" in model_stats:
        err = model_stats["reproj_error_mean"]
        if err > 2.0:
            recommendations.append(
                f"Mean reprojection error ({err:.3f} px) is high. This may indicate "
                "camera model inadequacy, poor feature matching, or moving objects. "
                "Consider: refining camera models, filtering outlier points, or using OPENCV_FISHEYE "
                "for wide-angle cameras."
            )
        elif err > 1.0:
            recommendations.append(
                f"Mean reprojection error ({err:.3f} px) is acceptable for underwater imagery."
            )
        else:
            recommendations.append(
                f"Mean reprojection error ({err:.3f} px) is excellent."
            )

    # Matching efficiency
    if db_stats.get("verified_pairs", 0) > 0 and db_stats.get("total_matches", 0) > 0:
        vrate = db_stats["verified_pairs"] / db_stats["total_matches"] * 100
        if vrate < 30:
            recommendations.append(
                f"Match verification rate ({vrate:.1f}%) is low. Many candidate pairs fail geometric "
                "verification. This is common for underwater imagery with backscatter. Consider: "
                "lowering --SiftMatching.max_ratio to 0.8, or using ALIKED+LightGlue features "
                "(--features ALIKED) which are more robust to illumination changes."
            )

    # Disk usage
    db_size_gb = os.path.getsize(db_path) / 1073741824 if os.path.exists(db_path) else 0
    if db_size_gb > 100:
        recommendations.append(
            f"Database size ({db_size_gb:.1f} GB) is large. For future runs, consider "
            "--max-features 8192 (COLMAP default) if 16K features per image isn't needed."
        )

    # Global mapper vs incremental
    recommendations.append(
        "Using global_mapper with gravity priors from pitch/roll. This constrains rotation "
        "averaging and should produce better camera orientations than incremental mapper "
        "for this near-nadir (looking-down) survey geometry."
    )

    # Future improvements
    recommendations.append(
        "Consider ALIKED+LightGlue (--features ALIKED) for future runs. Learned features "
        "are significantly more robust to low contrast and illumination variation than SIFT "
        "in underwater environments. Available in this COLMAP 4.1 build."
    )

    for i, rec in enumerate(recommendations, 1):
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.cell(8, 6, f"{i}.")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(50, 50, 50)
        pdf.multi_cell(0, 5, rec)
        pdf.ln(3)

    # ─── Output ──────────────────────────────────────────────────────────
    pdf.output(report_path)
    print(f"\n  Report generated: {report_path}")
    print(f"  Size: {os.path.getsize(report_path) / 1024:.0f} KB")


def main():
    parser = argparse.ArgumentParser(description="Generate COLMAP reconstruction benchmark report")
    parser.add_argument("output_dir", help="COLMAP output workspace directory")
    parser.add_argument("--report-path", default=None, help="Output PDF path (default: <output_dir>/report.pdf)")
    args = parser.parse_args()

    output_dir = args.output_dir
    report_path = args.report_path or os.path.join(output_dir, "report.pdf")

    if not os.path.isdir(output_dir):
        print(f"ERROR: Output directory not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    generate_report(output_dir, report_path)


if __name__ == "__main__":
    main()
