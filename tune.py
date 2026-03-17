#!/usr/bin/env python3
"""
tune.py — Terminal UI for LichtFeld Studio parameter tuning

Curses-based parameter tuner with live VRAM estimation, runtime estimates,
and one-key training launch. Optimized for 30GB usable VRAM (RTX 5090).

Usage:
    python3 tune.py                          # interactive mode
    python3 tune.py /path/to/dataset         # pre-fill dataset path
    python3 tune.py --preset max-quality     # start with max-quality preset
"""

import curses
import os
import re
import subprocess
import sys
import tempfile
import time

# ─── VRAM Model ──────────────────────────────────────────────────────────────
# Empirical model for gaussian splatting CUDA training VRAM usage.
# Calibrated against RTX 5090 (32 GB) benchmarks.

VRAM_TOTAL_GB = 32.0       # Physical VRAM
VRAM_OVERHEAD_GB = 2.0     # CUDA context + driver
VRAM_USABLE_GB = VRAM_TOTAL_GB - VRAM_OVERHEAD_GB  # 30 GB

# Per-gaussian costs during training (bytes)
GAUSSIAN_PARAMS_BYTES = 248       # position(12) + SH(192) + opacity(4) + scale(12) + rotation(16) + misc(12)
GAUSSIAN_OPTIMIZER_BYTES = 496    # Adam: 2x param size for first/second moment
GAUSSIAN_SORT_BYTES = 16          # tile sort keys + indices
BYTES_PER_GAUSSIAN = GAUSSIAN_PARAMS_BYTES + GAUSSIAN_OPTIMIZER_BYTES + GAUSSIAN_SORT_BYTES  # ~760

# Render buffer per pixel (forward + backward pass)
RENDER_BYTES_PER_PIXEL = 48  # color(12) + depth(4) + alpha(4) + gradients(28)

# Fixed training overhead (temporary buffers, cuBLAS workspace, etc.)
TRAINING_OVERHEAD_GB = 1.5


def estimate_vram_gb(max_cap, width, height, resize_factor):
    """Estimate peak VRAM usage in GB during training."""
    rw = width / resize_factor
    rh = height / resize_factor
    render_gb = rw * rh * RENDER_BYTES_PER_PIXEL / (1024**3)
    gaussian_gb = max_cap * BYTES_PER_GAUSSIAN / (1024**3)
    return VRAM_OVERHEAD_GB + TRAINING_OVERHEAD_GB + render_gb + gaussian_gb


def max_gaussians_for_vram(width, height, resize_factor, target_gb=VRAM_USABLE_GB):
    """Calculate maximum gaussians that fit in target VRAM."""
    rw = width / resize_factor
    rh = height / resize_factor
    render_gb = rw * rh * RENDER_BYTES_PER_PIXEL / (1024**3)
    available = target_gb - VRAM_OVERHEAD_GB - TRAINING_OVERHEAD_GB - render_gb
    if available <= 0:
        return 0
    return int(available * (1024**3) / BYTES_PER_GAUSSIAN)


def estimate_runtime_minutes(iterations, max_cap, width, height, resize_factor):
    """Approximate RTX 5090 training time in minutes.

    Empirical power-law fit calibrated against benchmark data:
      2M gaussians, half-res 4032x3024, 30K iters -> ~35 min
      3M gaussians, full-res 4032x3024, 30K iters -> ~90 min
    Accurate to roughly 1.5x.
    """
    pixels_M = (width / resize_factor) * (height / resize_factor) / 1e6
    cap_M = max_cap / 1e6
    ms_per_iter = 3 + 22 * (cap_M ** 0.8) * (pixels_M ** 0.5)
    return iterations * ms_per_iter / 60000


def format_runtime(minutes):
    """Human-readable runtime string."""
    if minutes < 1:
        return f"~{max(1, int(minutes * 60))}s"
    if minutes < 60:
        return f"~{int(minutes)} min"
    h = int(minutes // 60)
    m = int(minutes % 60)
    if m == 0:
        return f"~{h}h"
    return f"~{h}h {m}m"


# ─── Dataset Detection ───────────────────────────────────────────────────────

def detect_dataset(path):
    """Detect dataset properties: image count, resolution, point count."""
    info = {
        "path": path,
        "images": 0,
        "width": 4032,
        "height": 3024,
        "points": 0,
        "valid": False,
    }
    if not path or not os.path.isdir(path):
        return info

    # Count images
    img_dir = os.path.join(path, "images")
    if not os.path.isdir(img_dir):
        img_dir = path
    exts = (".jpg", ".jpeg", ".png")
    images = [f for f in os.listdir(img_dir) if f.lower().endswith(exts)]
    info["images"] = len(images)

    # Detect resolution from first image
    if images:
        sample = os.path.join(img_dir, images[0])
        try:
            result = subprocess.run(
                ["file", sample], capture_output=True, text=True, timeout=5
            )
            m = re.search(r"(\d{3,5})\s*x\s*(\d{3,5})", result.stdout)
            if m:
                info["width"] = int(m.group(1))
                info["height"] = int(m.group(2))
        except Exception:
            pass

    # Count points
    pts_file = os.path.join(path, "sparse", "0", "points3D.txt")
    if os.path.isfile(pts_file):
        try:
            with open(pts_file) as f:
                count = sum(1 for line in f if not line.startswith("#"))
            info["points"] = count
        except Exception:
            pass

    info["valid"] = info["images"] > 0
    return info


# ─── Presets ─────────────────────────────────────────────────────────────────

PRESET_ORDER = ["preview", "medium", "high", "max-quality"]

PRESET_LABELS = {
    "preview":     "PREVIEW",
    "medium":      "MEDIUM",
    "high":        "HIGH",
    "max-quality": "MAX",
}

PRESET_DESCRIPTIONS = {
    "preview":     "Fast check — does the pipeline work?",
    "medium":      "Production-ready, good quality/speed balance",
    "high":        "Best quality at half resolution",
    "max-quality": "Full resolution, max detail for this scene",
}


def scene_cap(n_pts, n_img, multiplier):
    """Estimate a sensible max-cap for a scene.

    ADC grows gaussians from the initial COLMAP point cloud.  Final counts
    typically land between 1x and 3x the initial point count; going much
    higher just wastes VRAM on capacity the optimizer won't fill.

    When no point count is available we fall back to an image-based
    heuristic (images x 300), but still apply the same multiplier so the
    estimate stays grounded.
    """
    base = n_pts if n_pts > 0 else n_img * 300
    return int(base * multiplier)


def make_presets(dataset_info):
    """Generate presets scaled to dataset size and 30 GB VRAM.

    Every preset caps max_cap at the *lesser* of what the scene likely
    needs (derived from actual point count) and what fits in VRAM.
    This avoids pre-allocating 37M gaussians for a 4M-point scene.
    """
    w = dataset_info["width"]
    h = dataset_info["height"]
    n_img = max(dataset_info["images"], 1)
    n_pts = dataset_info["points"]

    def capped(multiplier, resize, vram_fraction):
        """Scene-aware cap: min(scene need, VRAM budget), rounded to 100K."""
        need = scene_cap(n_pts, n_img, multiplier)
        vram_limit = max_gaussians_for_vram(w, h, resize, VRAM_USABLE_GB * vram_fraction)
        raw = min(need, vram_limit)
        return max(100000, (raw // 100000) * 100000)

    common = {
        "strategy": "adc",
        "min_opacity": 0.1,
        "prune_ratio": 0.6,
        "enable_sparsity": True,
        "enable_mip": True,
    }

    presets = {}

    presets["preview"] = {
        **common,
        "iterations": 3000,
        "max_cap": capped(1.0, 4, 0.80),
        "resize_factor": 4,
    }

    presets["medium"] = {
        **common,
        "iterations": 7000,
        "max_cap": capped(1.5, 2, 0.85),
        "resize_factor": 2,
    }

    presets["high"] = {
        **common,
        "iterations": 30000,
        "max_cap": capped(2.0, 2, 0.90),
        "resize_factor": 2,
    }

    presets["max-quality"] = {
        **common,
        "iterations": 30000,
        "max_cap": capped(3.0, 1, 0.92),
        "resize_factor": 1,
    }

    return presets


# ─── Parameter Definitions ───────────────────────────────────────────────────
# (key, label, type, min, max, step, default, unit, description)

TRAINING_PARAMS = [
    ("strategy",        "Strategy",       "choice", ["adc", "mcmc"], None, None, "adc",  "",
     "Densification method. ADC recommended — built-in pruning, no bloom."),
    ("iterations",      "Iterations",     "int",    1000, 100000, 1000, 30000, "",
     "Training steps. 3K=preview, 7K=fast, 30K=full convergence."),
    ("max_cap",         "Max gaussians",  "int",    100000, 50000000, 100000, 2000000, "",
     "VRAM ceiling for splat count. Sized relative to COLMAP points."),
    ("resize_factor",   "Resize factor",  "int",    1, 8, 1, 2, "x",
     "Image downscale. 1=full res, 2=half (4x fewer pixels), 4=quarter."),
    ("min_opacity",     "Min opacity",    "float",  0.0, 1.0, 0.01, 0.1, "",
     "Cull splats below this during training. 0.05=gentle, 0.15=aggressive."),
    ("prune_ratio",     "Prune ratio",    "float",  0.0, 1.0, 0.05, 0.6, "",
     "Fraction of low-contribution splats pruned per cycle. 0.4-0.8."),
    ("enable_sparsity", "Sparsity",       "bool",   None, None, None, True, "",
     "Sparsity regularization — penalizes redundant overlapping splats."),
    ("enable_mip",      "MIP filtering",  "bool",   None, None, None, True, "",
     "Anti-aliasing for multi-scale views. Always recommended."),
]

POST_PARAMS = [
    ("sor_passes",      "SOR passes",     "int",    0, 10, 1, 3, "",
     "Outlier removal iterations. Each pass tightens. 0=skip, 2-4=typical."),
    ("sor_std",         "SOR std factor",  "float",  0.5, 5.0, 0.1, 2.0, "s",
     "Distance threshold in std devs. 1.5=tight/aggressive, 2.5=loose."),
    ("sor_k",           "SOR neighbors",  "int",    5, 100, 5, 20, "",
     "K-nearest neighbors for density estimation. 10-30 typical."),
    ("opacity_clean",   "Opacity floor",  "float",  0.0, 0.5, 0.01, 0.05, "",
     "Post-training: remove near-invisible floaters below this opacity."),
]

ALL_PARAMS = TRAINING_PARAMS + POST_PARAMS


# ─── TUI ─────────────────────────────────────────────────────────────────────

class TunerApp:
    def __init__(self, stdscr, initial_dataset="", initial_preset=None):
        self.scr = stdscr
        self.cursor = 0          # index into ALL_PARAMS
        self.active_preset = 2   # index into PRESET_ORDER (default: "high")
        self.editing_path = False
        self.path_buffer = initial_dataset
        self.message = ""
        self.message_time = 0
        self.scroll_offset = 0   # for scrolling params if terminal is short

        # Initialize values from defaults
        self.values = {}
        for p in ALL_PARAMS:
            self.values[p[0]] = p[6]
        self.values["dataset"] = initial_dataset

        # Detect dataset
        self.dataset_info = detect_dataset(initial_dataset)

        # Apply preset
        preset_name = initial_preset or "high"
        if preset_name in PRESET_ORDER:
            self.active_preset = PRESET_ORDER.index(preset_name)
        self._apply_preset(PRESET_ORDER[self.active_preset])

        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)     # green
        curses.init_pair(2, curses.COLOR_YELLOW, -1)     # yellow
        curses.init_pair(3, curses.COLOR_RED, -1)        # red
        curses.init_pair(4, curses.COLOR_CYAN, -1)       # cyan / section headers
        curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLUE)   # title bar
        curses.init_pair(6, curses.COLOR_BLACK, curses.COLOR_GREEN)  # key hints
        curses.init_pair(7, curses.COLOR_BLACK, curses.COLOR_CYAN)   # selected preset
        curses.init_pair(8, curses.COLOR_WHITE, curses.COLOR_MAGENTA)  # runtime badge

    def _apply_preset(self, name):
        presets = make_presets(self.dataset_info)
        if name in presets:
            for k, v in presets[name].items():
                if k in self.values:
                    self.values[k] = v
            if name in PRESET_ORDER:
                self.active_preset = PRESET_ORDER.index(name)
            self.message = f"Preset: {PRESET_LABELS.get(name, name)}"
            self.message_time = time.time()

    def _set_message(self, msg):
        self.message = msg
        self.message_time = time.time()

    def _vram_estimate(self):
        return estimate_vram_gb(
            self.values["max_cap"],
            self.dataset_info["width"],
            self.dataset_info["height"],
            self.values["resize_factor"],
        )

    def _runtime_estimate(self):
        return estimate_runtime_minutes(
            self.values["iterations"],
            self.values["max_cap"],
            self.dataset_info["width"],
            self.dataset_info["height"],
            self.values["resize_factor"],
        )

    def _vram_max_gaussians(self):
        return max_gaussians_for_vram(
            self.dataset_info["width"],
            self.dataset_info["height"],
            self.values["resize_factor"],
            VRAM_USABLE_GB * 0.92,
        )

    def _preset_vram(self, name):
        presets = make_presets(self.dataset_info)
        p = presets[name]
        return estimate_vram_gb(
            p["max_cap"],
            self.dataset_info["width"],
            self.dataset_info["height"],
            p["resize_factor"],
        )

    def _preset_runtime(self, name):
        presets = make_presets(self.dataset_info)
        p = presets[name]
        return estimate_runtime_minutes(
            p["iterations"],
            p["max_cap"],
            self.dataset_info["width"],
            self.dataset_info["height"],
            p["resize_factor"],
        )

    def _build_command(self):
        ds = self.values["dataset"]
        name = os.path.basename(ds.rstrip("/")) if ds else "scene"
        out = os.path.expanduser(f"~/output/{name}_tuned")

        parts = [
            "~/gaussian-splatting-cuda/build/LichtFeld-Studio",
            f"-d {ds}" if ds else "-d <DATASET>",
            f"-o {out}",
            f"--strategy {self.values['strategy']}",
            f"--max-cap {self.values['max_cap']}",
            f"-i {self.values['iterations']}",
            f"-r {self.values['resize_factor']}",
            f"--min-opacity {self.values['min_opacity']}",
        ]
        if self.values["enable_sparsity"]:
            parts.append("--enable-sparsity")
            parts.append(f"--prune-ratio {self.values['prune_ratio']}")
        if self.values["enable_mip"]:
            parts.append("--enable-mip")
        parts.append("--headless")
        return " \\\n    ".join(parts)

    def _build_clean_command(self):
        parts = ["python3 clean_splat.py <output.ply> <cleaned.ply>"]
        if self.values["opacity_clean"] > 0:
            parts.append(f"--opacity-min {self.values['opacity_clean']}")
        parts.append(f"-k {self.values['sor_k']}")
        parts.append(f"-s {self.values['sor_std']}")
        parts.append(f"-p {self.values['sor_passes']}")
        return " ".join(parts)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _safe_addstr(self, row, col, text, attr=0):
        """addstr that silently clips at screen edges."""
        h, w = self.scr.getmaxyx()
        if row < 0 or row >= h or col >= w:
            return
        text = text[:max(0, w - col - 1)]
        if text:
            try:
                self.scr.addstr(row, col, text, attr)
            except curses.error:
                pass

    def draw(self):
        self.scr.erase()
        h, w = self.scr.getmaxyx()
        sa = self._safe_addstr

        # ── Title bar ──
        sa(0, 0, " " * w, curses.color_pair(5))
        title = "LichtFeld Studio — Parameter Tuner"
        sa(0, 2, title, curses.color_pair(5) | curses.A_BOLD)
        gpu_tag = "RTX 5090 · 32 GB"
        sa(0, max(0, w - len(gpu_tag) - 2), gpu_tag, curses.color_pair(5))

        row = 2

        # ── Dataset line ──
        ds = self.values["dataset"]
        if self.editing_path:
            sa(row, 2, "Dataset: ", curses.color_pair(4) | curses.A_BOLD)
            display = self.path_buffer + "▌"
            sa(row, 11, display[:w - 13], curses.A_UNDERLINE)
            row += 1
        elif ds:
            sa(row, 2, "Dataset: ", curses.color_pair(4))
            sa(row, 11, ds[:w - 13])
            row += 1
            if self.dataset_info["valid"]:
                info = self.dataset_info
                detail = (f"{info['images']:,} images · "
                          f"{info['width']}x{info['height']} · "
                          f"{info['points']:,} COLMAP points")
                sa(row, 11, detail, curses.A_DIM)
                row += 1
        else:
            sa(row, 2, "Dataset: ", curses.color_pair(4))
            sa(row, 11, "press Enter to set path", curses.A_DIM)
            row += 1

        # ── VRAM gauge + runtime ──
        row += 1
        vram_est = self._vram_estimate()
        vram_pct = min(vram_est / VRAM_TOTAL_GB, 1.0)
        bar_w = min(36, w - 38)

        if vram_pct > 0.95:
            vram_color = curses.color_pair(3) | curses.A_BOLD
            vram_tag = "DANGER"
        elif vram_pct > 0.85:
            vram_color = curses.color_pair(2) | curses.A_BOLD
            vram_tag = "TIGHT"
        elif vram_pct > 0.70:
            vram_color = curses.color_pair(2)
            vram_tag = "OK"
        else:
            vram_color = curses.color_pair(1)
            vram_tag = "SAFE"

        filled = int(vram_pct * bar_w)
        sa(row, 2, "VRAM ", curses.color_pair(4))
        sa(row, 7, "[")
        for i in range(bar_w):
            ch = "#" if i < filled else "-"
            attr = vram_color if i < filled else curses.A_DIM
            sa(row, 8 + i, ch, attr)
        right = f"] {vram_est:.1f}/{VRAM_TOTAL_GB:.0f} GB  {vram_tag}"
        sa(row, 8 + bar_w, right, vram_color)

        runtime = self._runtime_estimate()
        rt_str = format_runtime(runtime)
        rt_full = f"  ETA {rt_str}"
        sa(row, 8 + bar_w + len(right) + 1, rt_full, curses.color_pair(8) | curses.A_BOLD)
        row += 2

        # ── Presets panel ──
        sa(row, 2, "--- Presets ", curses.color_pair(4))
        sa(row, 14, "pick one, then fine-tune below ", curses.A_DIM)
        sa(row, 45, "-" * max(0, w - 47), curses.color_pair(4))
        row += 1

        presets = make_presets(self.dataset_info)
        for idx, pname in enumerate(PRESET_ORDER):
            p = presets[pname]
            label = PRESET_LABELS[pname]
            desc = PRESET_DESCRIPTIONS[pname]
            is_active = (idx == self.active_preset)

            # Key badge
            key_str = f" {idx + 1} "
            sa(row, 2, key_str, curses.color_pair(6))

            # Preset name
            name_str = f" {label:<8}"
            if is_active:
                sa(row, 6, name_str, curses.color_pair(7) | curses.A_BOLD)
            else:
                sa(row, 6, name_str, curses.A_DIM)

            # Stats columns
            col = 15
            iters_s = f"{p['iterations'] // 1000}K"
            sa(row, col, f"{iters_s:>4} iter", curses.A_BOLD if is_active else curses.A_DIM)
            col += 10

            rf_s = f"-r{p['resize_factor']}"
            sa(row, col, rf_s, curses.A_BOLD if is_active else curses.A_DIM)
            col += 5

            cap_s = f"{p['max_cap']/1e6:.1f}M" if p['max_cap'] >= 1e6 else f"{p['max_cap']//1000}K"
            sa(row, col, f"{cap_s:>6} pts", curses.A_BOLD if is_active else curses.A_DIM)
            col += 11

            # Runtime
            rt = estimate_runtime_minutes(
                p["iterations"], p["max_cap"],
                self.dataset_info["width"], self.dataset_info["height"],
                p["resize_factor"],
            )
            rt_s = format_runtime(rt)
            sa(row, col, f"{rt_s:>8}", curses.color_pair(2) if is_active else curses.A_DIM)
            col += 10

            # VRAM
            vram = estimate_vram_gb(
                p["max_cap"],
                self.dataset_info["width"], self.dataset_info["height"],
                p["resize_factor"],
            )
            vram_s = f"{vram:.1f} GB"
            sa(row, col, f"{vram_s:>7}", curses.A_DIM)
            col += 9

            # Description
            sa(row, col, desc[:w - col - 1], curses.A_DIM)

            row += 1

        row += 1

        # ── Parameter area ──
        # Calculate how many rows we have for params + command
        # Reserve: 1 message + 1 help bar + 1 blank = 3 bottom rows
        # Plus ~4 rows for command preview
        avail_param_rows = h - row - 7
        total_param_display = len(ALL_PARAMS) + 2  # +2 for section headers

        # Adjust scroll
        # Map cursor to display row (accounting for section header)
        cursor_display = self.cursor
        if self.cursor >= len(TRAINING_PARAMS):
            cursor_display += 1  # account for post-processing header

        if cursor_display < self.scroll_offset:
            self.scroll_offset = cursor_display
        elif cursor_display >= self.scroll_offset + avail_param_rows:
            self.scroll_offset = cursor_display - avail_param_rows + 1
        self.scroll_offset = max(0, min(self.scroll_offset, total_param_display - avail_param_rows))

        # Build display list: [(param_index_or_None, section_label_or_None)]
        display_items = []
        display_items.append((None, "--- Training "))
        for i in range(len(TRAINING_PARAMS)):
            display_items.append((i, None))
        display_items.append((None, "--- Post-Processing (clean_splat.py) "))
        for i in range(len(TRAINING_PARAMS), len(ALL_PARAMS)):
            display_items.append((i, None))

        visible = display_items[self.scroll_offset:self.scroll_offset + avail_param_rows]

        for display_row, (param_idx, section_label) in enumerate(visible):
            r = row + display_row
            if r >= h - 4:
                break

            if section_label is not None:
                # Section header
                sa(r, 2, section_label, curses.color_pair(4))
                pad_start = 2 + len(section_label)
                sa(r, pad_start, "-" * max(0, w - pad_start - 2), curses.color_pair(4))
                continue

            p = ALL_PARAMS[param_idx]
            key, label, ptype = p[0], p[1], p[2]
            desc = p[8]
            val = self.values[key]
            is_selected = (param_idx == self.cursor)

            # Cursor indicator
            if is_selected:
                sa(r, 1, ">", curses.color_pair(1) | curses.A_BOLD)

            # Label (fixed width)
            label_w = 16
            sa(r, 3, f"{label:<{label_w}}", curses.A_BOLD if is_selected else 0)

            # Value
            val_col = 3 + label_w + 1
            attr = curses.A_BOLD if is_selected else 0
            val_end = val_col  # track where value display ends

            if ptype == "bool":
                display = " ON " if val else " OFF"
                color = curses.color_pair(1) if val else curses.color_pair(3)
                sa(r, val_col, "<", attr)
                sa(r, val_col + 1, display, attr | color)
                sa(r, val_col + 1 + len(display), " >", attr)
                val_end = val_col + len(display) + 3
            elif ptype == "choice":
                display = f" {val} "
                sa(r, val_col, "<", attr)
                sa(r, val_col + 1, display, attr)
                sa(r, val_col + 1 + len(display), ">", attr)
                val_end = val_col + len(display) + 2
                if key == "strategy" and val == "mcmc":
                    sa(r, val_end + 1, "!! no --min-opacity!", curses.color_pair(3))
                    val_end += 21
            elif ptype == "int":
                unit = p[7]
                val_s = f"{val:>12,}"
                sa(r, val_col, "<", attr)
                sa(r, val_col + 1, val_s, attr)
                sa(r, val_col + 1 + len(val_s), f" >{unit}", attr)
                val_end = val_col + len(val_s) + 3 + len(unit)
            elif ptype == "float":
                unit = p[7]
                val_s = f"{val:>8.2f}"
                sa(r, val_col, "<", attr)
                sa(r, val_col + 1, val_s, attr)
                sa(r, val_col + 1 + len(val_s), f" >{unit}", attr)
                val_end = val_col + len(val_s) + 3 + len(unit)

            # Description (dim, right of value)
            desc_col = max(val_end + 2, 42)
            if desc_col < w - 10:
                sa(r, desc_col, desc[:w - desc_col - 1], curses.A_DIM)

        row += len(visible) + 1

        # ── Command preview ──
        if row < h - 4:
            sa(row, 2, "--- Command ", curses.color_pair(4))
            sa(row, 14, "-" * max(0, w - 16), curses.color_pair(4))
            row += 1
            cmd = self._build_command()
            for line in cmd.split("\n"):
                if row < h - 3:
                    sa(row, 4, line[:w - 6], curses.A_DIM)
                    row += 1
            if row < h - 3:
                clean = self._build_clean_command()
                sa(row, 4, clean[:w - 6], curses.A_DIM)
                row += 1

        # ── Message ──
        if self.message and time.time() - self.message_time < 4:
            sa(h - 3, 2, self.message[:w - 4], curses.color_pair(2) | curses.A_BOLD)

        # ── Help bar ──
        help_row = h - 1
        helps = [
            ("1-4", "preset"),
            ("^/v", "select"),
            ("</>", "adjust"),
            ("[/]", "x10"),
            ("Enter", "edit path"),
            ("m", "fill VRAM"),
            ("r", "RUN"),
            ("c", "copy"),
            ("q", "quit"),
        ]
        col = 0
        for key_str, desc_str in helps:
            needed = len(key_str) + len(desc_str) + 4
            if col + needed > w:
                break
            sa(help_row, col, f" {key_str} ", curses.color_pair(6))
            sa(help_row, col + len(key_str) + 2, f" {desc_str}", curses.A_DIM)
            col += needed + 1

        self.scr.refresh()

    # ── Input handling ───────────────────────────────────────────────────────

    def adjust(self, direction):
        """Adjust current parameter by direction (-1 or +1)."""
        p = ALL_PARAMS[self.cursor]
        key, ptype = p[0], p[2]

        if ptype == "bool":
            self.values[key] = not self.values[key]
        elif ptype == "choice":
            choices = p[3]
            idx = choices.index(self.values[key])
            self.values[key] = choices[(idx + direction) % len(choices)]
            if key == "strategy" and self.values[key] == "mcmc":
                self._set_message("WARNING: MCMC + --min-opacity crashes! Auto-disabling.")
                self.values["min_opacity"] = 0.0
        elif ptype == "int":
            step = p[5]
            val = self.values[key] + direction * step
            self.values[key] = max(p[3], min(p[4], val))
        elif ptype == "float":
            step = p[5]
            val = round(self.values[key] + direction * step, 4)
            self.values[key] = max(p[3], min(p[4], val))

    def big_adjust(self, direction):
        """Adjust by 10x step."""
        p = ALL_PARAMS[self.cursor]
        key, ptype = p[0], p[2]
        if ptype == "int":
            step = p[5] * 10
            val = self.values[key] + direction * step
            self.values[key] = max(p[3], min(p[4], val))
        elif ptype == "float":
            step = p[5] * 10
            val = round(self.values[key] + direction * step, 4)
            self.values[key] = max(p[3], min(p[4], val))

    def set_max_gaussians(self):
        """Set max_cap to the maximum that fits in VRAM."""
        max_g = self._vram_max_gaussians()
        max_g = (max_g // 100000) * 100000
        self.values["max_cap"] = max(100000, max_g)
        self._set_message(f"Max cap set to {max_g:,} (92% VRAM fill)")

    def launch_training(self):
        """Generate runnable scripts in output directory."""
        ds = self.values["dataset"]
        if not ds or not os.path.isdir(ds):
            self._set_message("ERROR: Set a valid dataset path first!")
            return

        name = os.path.basename(ds.rstrip("/"))
        out = os.path.expanduser(f"~/output/{name}_tuned_{int(time.time())}")
        os.makedirs(out, exist_ok=True)

        cmd = self._build_command().replace("\\\n    ", " ")
        cmd = cmd.replace(f"~/output/{name}_tuned", out)
        log_file = os.path.join(out, "training.log")

        vram = self._vram_estimate()
        runtime = format_runtime(self._runtime_estimate())

        with open(os.path.join(out, "command.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Generated by tune.py at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# VRAM estimate: {vram:.1f} GB | Runtime estimate: {runtime}\n\n")
            f.write(cmd.replace("    ", "") + f" 2>&1 | tee {log_file}\n")
        os.chmod(os.path.join(out, "command.sh"), 0o755)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(out, "clean.sh"), "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# Post-processing for {name}\n")
            f.write(f"PLY=$(find {out} -name '*.ply' -not -name '*_cleaned*' | sort | tail -1)\n")
            f.write(f'[ -z "$PLY" ] && echo "No PLY found" && exit 1\n')
            f.write(f'CLEANED="${{PLY%.ply}}_cleaned.ply"\n')
            f.write(f"python3 {script_dir}/clean_splat.py \"$PLY\" \"$CLEANED\" "
                    f"--opacity-min {self.values['opacity_clean']} "
                    f"-k {self.values['sor_k']} "
                    f"-s {self.values['sor_std']} "
                    f"-p {self.values['sor_passes']}\n")
            f.write(f'echo "Cleaned: $CLEANED"\n')
        os.chmod(os.path.join(out, "clean.sh"), 0o755)

        self._set_message(f"Saved: {out}/command.sh  (run it to train)")
        return out

    def copy_command(self):
        """Copy command to clipboard via xclip/xsel/wl-copy."""
        cmd = self._build_command().replace("\\\n    ", " ")
        for clipcmd in ["xclip -selection clipboard", "xsel --clipboard", "wl-copy"]:
            try:
                proc = subprocess.Popen(
                    clipcmd.split(), stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
                )
                proc.communicate(cmd.encode())
                if proc.returncode == 0:
                    self._set_message("Command copied to clipboard!")
                    return
            except FileNotFoundError:
                continue
        fallback_path = os.path.join(tempfile.gettempdir(), "lichtfeld_cmd.sh")
        with open(fallback_path, "w") as f:
            f.write(cmd + "\n")
        self._set_message(f"Saved to {fallback_path} (no clipboard tool found)")

    def run(self):
        self.scr.timeout(100)

        while True:
            self.draw()

            try:
                key = self.scr.getch()
            except curses.error:
                continue

            if key == -1:
                continue

            # Path editing mode
            if self.editing_path:
                if key == 27:  # Escape
                    self.editing_path = False
                    curses.curs_set(0)
                elif key in (curses.KEY_ENTER, 10, 13):
                    self.values["dataset"] = self.path_buffer
                    self.dataset_info = detect_dataset(self.path_buffer)
                    self.editing_path = False
                    curses.curs_set(0)
                    if self.dataset_info["valid"]:
                        self._set_message(f"Detected {self.dataset_info['images']:,} images")
                        # Re-apply current preset with new dataset info
                        self._apply_preset(PRESET_ORDER[self.active_preset])
                    else:
                        self._set_message("No images found at that path")
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    self.path_buffer = self.path_buffer[:-1]
                elif key == 9:  # Tab
                    self._tab_complete()
                elif 32 <= key < 127:
                    self.path_buffer += chr(key)
                continue

            # Normal mode
            if key in (ord("q"), ord("Q"), 27):
                break
            elif key == curses.KEY_UP:
                self.cursor = max(0, self.cursor - 1)
            elif key == curses.KEY_DOWN:
                self.cursor = min(len(ALL_PARAMS) - 1, self.cursor + 1)
            elif key == curses.KEY_LEFT:
                self.adjust(-1)
            elif key == curses.KEY_RIGHT:
                self.adjust(1)
            elif key == curses.KEY_SLEFT:
                self.big_adjust(-10)
            elif key == curses.KEY_SRIGHT:
                self.big_adjust(10)
            elif key in (curses.KEY_ENTER, 10, 13):
                p = ALL_PARAMS[self.cursor]
                if p[2] == "bool":
                    self.adjust(1)
                else:
                    # Enter on any param opens dataset path editor
                    self.editing_path = True
                    self.path_buffer = self.values["dataset"]
                    curses.curs_set(1)
            elif key == ord("["):
                self.big_adjust(-1)
            elif key == ord("]"):
                self.big_adjust(1)

            # Presets
            elif key == ord("1"):
                self._apply_preset("preview")
            elif key == ord("2"):
                self._apply_preset("medium")
            elif key == ord("3"):
                self._apply_preset("high")
            elif key == ord("4"):
                self._apply_preset("max-quality")

            # Actions
            elif key in (ord("m"), ord("M")):
                self.set_max_gaussians()
            elif key in (ord("r"), ord("R")):
                self.launch_training()
            elif key in (ord("c"), ord("C")):
                self.copy_command()

    def _tab_complete(self):
        """Simple path tab completion."""
        path = self.path_buffer
        if path.startswith("~"):
            path = os.path.expanduser(path)
        dirname = os.path.dirname(path) or "."
        basename = os.path.basename(path)
        try:
            entries = os.listdir(dirname)
            matches = sorted(e for e in entries if e.startswith(basename))
            if len(matches) == 1:
                completed = os.path.join(dirname, matches[0])
                if os.path.isdir(completed):
                    completed += "/"
                self.path_buffer = completed
            elif len(matches) > 1:
                prefix = os.path.commonprefix(matches)
                if prefix:
                    self.path_buffer = os.path.join(dirname, prefix)
        except (OSError, PermissionError):
            pass


def main():
    dataset = ""
    preset = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--preset" and i + 1 < len(args):
            preset = args[i + 1]
            i += 2
        elif args[i].startswith("--preset="):
            preset = args[i].split("=", 1)[1]
            i += 1
        elif not args[i].startswith("-"):
            dataset = args[i]
            i += 1
        else:
            i += 1

    def app(stdscr):
        TunerApp(stdscr, initial_dataset=dataset, initial_preset=preset).run()

    curses.wrapper(app)


if __name__ == "__main__":
    main()
