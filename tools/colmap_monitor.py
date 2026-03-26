#!/usr/bin/env python3
"""
COLMAP Reconstruction Pipeline — Real-Time Monitoring Dashboard

A self-contained web server that provides live visualization of COLMAP
pipeline progress by polling the SQLite database and reading log files.

Auto-launched by colmap_reconstruct.sh. Serves at http://localhost:8080.

Usage:
    python3 colmap_monitor.py --db /path/to/database.db --log-dir /path/to/logs
    python3 colmap_monitor.py --db /path/to/database.db --log-dir /path/to/logs --port 8080

Requires: Python 3.6+ (stdlib only, no external packages)
"""

import argparse
import glob
import json
import os
import sqlite3
import struct
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# ─── Global State (updated by poller thread) ────────────────────────────────

state = {
    "stage": "INITIALIZING",
    "start_time": time.time(),
    "image_count_total": 0,
    "images": 0,
    "cameras": [],
    "keypoints": 0,
    "total_features": 0,
    "matches": 0,
    "verified_pairs": 0,
    "pose_priors": 0,
    "positions": [],
    "log_lines": [],
    "error": None,
}
state_lock = threading.Lock()

# Config (set from CLI args)
config = {
    "db_path": "",
    "log_dir": "",
    "stage_file": "",
    "image_count": 0,
    "port": 8080,
}


# ─── Database Poller ─────────────────────────────────────────────────────────

def poll_database():
    """Background thread: polls COLMAP database every 3 seconds."""
    while True:
        try:
            db_path = config["db_path"]
            if not os.path.exists(db_path):
                time.sleep(3)
                continue

            conn = sqlite3.connect(db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=30000")
            conn.execute("PRAGMA query_only=ON")

            data = {}

            # Image count
            try:
                data["images"] = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            except Exception:
                data["images"] = 0

            # Camera stats
            try:
                rows = conn.execute(
                    "SELECT model, COUNT(*) as cnt FROM cameras GROUP BY model"
                ).fetchall()
                model_names = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 2: "SIMPLE_RADIAL",
                               3: "RADIAL", 4: "OPENCV", 5: "OPENCV_FISHEYE",
                               6: "FULL_OPENCV", 8: "SIMPLE_RADIAL_FISHEYE", 9: "RADIAL_FISHEYE"}
                data["cameras"] = [
                    {"model": model_names.get(m, f"UNKNOWN({m})"), "count": c}
                    for m, c in rows
                ]
            except Exception:
                data["cameras"] = []

            # Keypoints
            try:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(rows), 0) FROM keypoints WHERE rows > 0"
                ).fetchone()
                data["keypoints"] = row[0]
                data["total_features"] = row[1]
            except Exception:
                data["keypoints"] = 0
                data["total_features"] = 0

            # Matches
            try:
                data["matches"] = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
            except Exception:
                data["matches"] = 0

            # Verified pairs
            try:
                data["verified_pairs"] = conn.execute(
                    "SELECT COUNT(*) FROM two_view_geometries WHERE rows > 0"
                ).fetchone()[0]
            except Exception:
                data["verified_pairs"] = 0

            # Pose priors
            try:
                data["pose_priors"] = conn.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
            except Exception:
                data["pose_priors"] = 0

            # Position scatter data (subsample evenly for large datasets)
            try:
                total_priors = conn.execute("SELECT COUNT(*) FROM pose_priors").fetchone()[0]
                # Sample evenly: take every Nth row to get ~5000 points
                step = max(1, total_priors // 5000)
                rows = conn.execute(
                    "SELECT p.position, i.name FROM pose_priors p "
                    "JOIN images i ON p.corr_data_id = i.image_id "
                    "WHERE p.pose_prior_id % ? = 0",
                    (step,),
                ).fetchall()
                positions = []
                for pos_blob, name in rows:
                    if pos_blob and len(pos_blob) >= 24:
                        x, y, z = struct.unpack("<3d", pos_blob[:24])
                        # Detect camera from filename
                        if "HERC" in name:
                            cam = "HERC"
                        elif "camlower" in name:
                            cam = "camlower"
                        elif "cammid" in name:
                            cam = "cammid"
                        elif "camupper" in name:
                            cam = "camupper"
                        else:
                            cam = name.split("_")[0] if "_" in name else "unknown"
                        positions.append({"x": x, "y": y, "z": z, "camera": cam})
                data["positions"] = positions
            except Exception:
                data["positions"] = []

            conn.close()

            # Read stage file
            stage = "UNKNOWN"
            sf = config["stage_file"]
            if sf and os.path.exists(sf):
                try:
                    with open(sf) as f:
                        stage = f.read().strip() or "UNKNOWN"
                except Exception:
                    pass
            data["stage"] = stage

            # Read log tail
            log_lines = []
            log_dir = config["log_dir"]
            if log_dir and os.path.isdir(log_dir):
                # COLMAP writes log files inside --log_path directory
                # Find the most recently modified log file (any depth)
                all_logs = []
                for root, dirs, files in os.walk(log_dir):
                    for f in files:
                        fp = os.path.join(root, f)
                        try:
                            all_logs.append((os.path.getmtime(fp), fp))
                        except OSError:
                            pass
                all_logs.sort(reverse=True)
                if all_logs:
                    try:
                        with open(all_logs[0][1], errors="replace") as f:
                            lines = f.readlines()
                            log_lines = [l.rstrip() for l in lines[-100:]]
                    except Exception:
                        pass
            data["log_lines"] = log_lines

            # Update global state
            with state_lock:
                state.update(data)
                state["image_count_total"] = config["image_count"] or data["images"]
                state["error"] = None

        except Exception as e:
            with state_lock:
                state["error"] = str(e)

        time.sleep(3)


# ─── HTTP Handler ────────────────────────────────────────────────────────────

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress request logging

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/status":
            self._serve_json()
        elif self.path == "/api/positions":
            with state_lock:
                positions = state.get("positions", [])
            self._json_response(positions)
        elif self.path == "/api/log":
            with state_lock:
                lines = state.get("log_lines", [])
            self._json_response(lines)
        else:
            self.send_error(404)

    def _serve_json(self):
        with state_lock:
            data = {
                "stage": state["stage"],
                "elapsed_seconds": int(time.time() - state["start_time"]),
                "image_count_total": state["image_count_total"],
                "images": state["images"],
                "cameras": state["cameras"],
                "keypoints": state["keypoints"],
                "total_features": state["total_features"],
                "matches": state["matches"],
                "verified_pairs": state["verified_pairs"],
                "pose_priors": state["pose_priors"],
                "error": state["error"],
            }
        self._json_response(data)

    def _json_response(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _serve_html(self):
        body = DASHBOARD_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


# ─── Dashboard HTML ──────────────────────────────────────────────────────────

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>COLMAP Pipeline Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background:#0d1117; color:#c9d1d9; }
  .header { background:#161b22; padding:16px 24px; border-bottom:1px solid #30363d; display:flex; justify-content:space-between; align-items:center; }
  .header h1 { font-size:18px; color:#58a6ff; }
  .header .stage { font-size:14px; padding:4px 12px; border-radius:12px; font-weight:600; }
  .stage-FEATURE_EXTRACTION { background:#1f6feb33; color:#58a6ff; }
  .stage-MATCHING { background:#da363333; color:#f85149; }
  .stage-RECONSTRUCTION { background:#238636; color:#3fb950; }
  .stage-COMPLETE { background:#238636; color:#3fb950; }
  .stage-CAMERA_ASSIGNMENT, .stage-INITIALIZING { background:#30363d; color:#8b949e; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; padding:16px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }
  .card h2 { font-size:13px; color:#8b949e; text-transform:uppercase; letter-spacing:0.5px; margin-bottom:12px; }
  .stat { font-size:28px; font-weight:700; color:#f0f6fc; }
  .stat-row { display:flex; gap:24px; flex-wrap:wrap; }
  .stat-item { text-align:center; }
  .stat-label { font-size:11px; color:#8b949e; margin-top:2px; }
  .full { grid-column: 1 / -1; }
  .progress-bar { height:6px; background:#21262d; border-radius:3px; margin-top:8px; overflow:hidden; }
  .progress-fill { height:100%; border-radius:3px; transition:width 0.5s; }
  .camera-table { width:100%; border-collapse:collapse; font-size:13px; }
  .camera-table td { padding:4px 8px; border-bottom:1px solid #21262d; }
  .camera-table td:last-child { text-align:right; font-weight:600; color:#f0f6fc; }
  #logbox { background:#0d1117; border:1px solid #30363d; border-radius:4px; padding:8px; font-family:'Cascadia Code',monospace; font-size:11px; height:200px; overflow-y:auto; white-space:pre-wrap; word-break:break-all; color:#8b949e; }
  canvas { max-height:250px; }
  .elapsed { color:#8b949e; font-size:14px; }
</style>
</head>
<body>

<div class="header">
  <h1>COLMAP Pipeline Monitor</h1>
  <div>
    <span id="stage" class="stage stage-INITIALIZING">INITIALIZING</span>
    <span class="elapsed" id="elapsed"></span>
  </div>
</div>

<div class="grid">
  <!-- Stats Row -->
  <div class="card full">
    <div class="stat-row" id="stats">
      <div class="stat-item"><div class="stat" id="s-images">0</div><div class="stat-label">Images</div></div>
      <div class="stat-item"><div class="stat" id="s-features">0</div><div class="stat-label">Features</div></div>
      <div class="stat-item"><div class="stat" id="s-matches">0</div><div class="stat-label">Match Pairs</div></div>
      <div class="stat-item"><div class="stat" id="s-verified">0</div><div class="stat-label">Verified</div></div>
      <div class="stat-item"><div class="stat" id="s-priors">0</div><div class="stat-label">Pose Priors</div></div>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progress" style="width:0%;background:#58a6ff;"></div></div>
  </div>

  <!-- Camera Stats -->
  <div class="card">
    <h2>Camera Models</h2>
    <table class="camera-table" id="cam-table"><tbody></tbody></table>
  </div>

  <!-- Position Scatter -->
  <div class="card">
    <h2>Image Positions (XY)</h2>
    <canvas id="posChart"></canvas>
  </div>

  <!-- Log Tail -->
  <div class="card full">
    <h2>Log Output</h2>
    <div id="logbox"></div>
  </div>
</div>

<script>
const fmt = n => n >= 1e6 ? (n/1e6).toFixed(1)+'M' : n >= 1e3 ? (n/1e3).toFixed(1)+'K' : n.toString();
const fmtTime = s => {const h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sec=s%60; return h>0?h+'h '+m+'m':m>0?m+'m '+sec+'s':sec+'s';};

// Position scatter chart
const posCtx = document.getElementById('posChart').getContext('2d');
const camColors = {camlower:'#58a6ff', cammid:'#f85149', camupper:'#3fb950', HERC:'#d2a8ff', unknown:'#8b949e'};
const posChart = new Chart(posCtx, {
  type:'scatter',
  data:{datasets:[]},
  options:{
    responsive:true, maintainAspectRatio:false,
    plugins:{legend:{labels:{color:'#8b949e',font:{size:10}}}},
    scales:{
      x:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#21262d'},title:{display:true,text:'Easting (m)',color:'#8b949e'}},
      y:{ticks:{color:'#8b949e',font:{size:10}},grid:{color:'#21262d'},title:{display:true,text:'Northing (m)',color:'#8b949e'}}
    }
  }
});

function updatePositions(positions) {
  const groups = {};
  positions.forEach(p => {
    if (!groups[p.camera]) groups[p.camera] = [];
    groups[p.camera].push({x:p.x, y:p.y});
  });
  posChart.data.datasets = Object.entries(groups).map(([cam,pts]) => ({
    label:cam, data:pts, pointRadius:2, pointBackgroundColor:camColors[cam]||'#8b949e',
  }));
  posChart.update('none');
}

const stageOrder = ['INITIALIZING','FEATURE_EXTRACTION','CAMERA_ASSIGNMENT','MATCHING','RECONSTRUCTION','DENSE_RECONSTRUCTION','COMPLETE'];
function stageProgress(stage) {
  const idx = stageOrder.indexOf(stage);
  return idx < 0 ? 0 : Math.round((idx / (stageOrder.length-1)) * 100);
}

async function refresh() {
  try {
    const [statusRes, posRes, logRes] = await Promise.all([
      fetch('/api/status'), fetch('/api/positions'), fetch('/api/log')
    ]);
    const s = await statusRes.json();
    const positions = await posRes.json();
    const lines = await logRes.json();

    // Stage
    const el = document.getElementById('stage');
    el.textContent = s.stage;
    el.className = 'stage stage-' + s.stage;
    document.getElementById('elapsed').textContent = fmtTime(s.elapsed_seconds);

    // Stats
    document.getElementById('s-images').textContent = fmt(s.images);
    document.getElementById('s-features').textContent = fmt(s.total_features);
    document.getElementById('s-matches').textContent = fmt(s.matches);
    document.getElementById('s-verified').textContent = fmt(s.verified_pairs);
    document.getElementById('s-priors').textContent = fmt(s.pose_priors);

    // Progress
    const pct = stageProgress(s.stage);
    const bar = document.getElementById('progress');
    bar.style.width = pct + '%';
    bar.style.background = s.stage === 'COMPLETE' ? '#238636' : '#58a6ff';

    // Camera table
    const tbody = document.querySelector('#cam-table tbody');
    tbody.innerHTML = s.cameras.map(c => '<tr><td>'+c.model+'</td><td>'+c.count+'</td></tr>').join('');

    // Positions
    if (positions.length > 0) updatePositions(positions);

    // Log
    const logbox = document.getElementById('logbox');
    const atBottom = logbox.scrollHeight - logbox.scrollTop <= logbox.clientHeight + 50;
    logbox.textContent = lines.join('\\n');
    if (atBottom) logbox.scrollTop = logbox.scrollHeight;

  } catch(e) { console.error('Refresh error:', e); }
}

setInterval(refresh, 3000);
refresh();
</script>
</body>
</html>"""


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="COLMAP Pipeline Monitor Dashboard")
    parser.add_argument("--db", required=True, help="Path to COLMAP database.db")
    parser.add_argument("--log-dir", default="", help="Directory containing COLMAP log files")
    parser.add_argument("--stage-file", default="", help="Path to .colmap_stage file")
    parser.add_argument("--image-count", type=int, default=0, help="Total expected image count")
    parser.add_argument("--port", type=int, default=8080, help="HTTP server port")
    args = parser.parse_args()

    config["db_path"] = args.db
    config["log_dir"] = args.log_dir
    config["stage_file"] = args.stage_file
    config["image_count"] = args.image_count
    config["port"] = args.port

    # Start poller thread
    poller = threading.Thread(target=poll_database, daemon=True)
    poller.start()

    # Start HTTP server
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"  Dashboard: http://localhost:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
