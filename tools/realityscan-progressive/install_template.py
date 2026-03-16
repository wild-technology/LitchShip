#!/usr/bin/env python3
"""
install_template.py — Install the LitchShip quality report template into RealityScan.

Copies quality_report.json.tpl into RealityScan's Reports/ directory and registers
it in Report.xml so it appears in the export dialog.

Usage (run on Windows where RealityScan is installed):
    python install_template.py
    python install_template.py --rs-path "C:\\Program Files\\Epic Games\\RealityScan_2.1"
    python install_template.py --dry-run
"""

import argparse
import glob
import os
import shutil
import sys
from datetime import datetime

TEMPLATE_GUID = "{8F3A1B2C-4D5E-6F78-9A0B-C1D2E3F4A5B6}"
TEMPLATE_FILENAME = "quality_report.json.tpl"
TEMPLATE_DESC = "Quality Report (JSON) for LitchShip"

FORMAT_ENTRY = f'''  <format id="{TEMPLATE_GUID}" mask="*.json" desc="{TEMPLATE_DESC}" writer="RealityScan.Export.ReportWriter">
    <hint>Per-image quality metrics (tie points, coverage, reprojection error) for progressive Gaussian splatting training</hint>
    <body>$Include("Reports\\{TEMPLATE_FILENAME}")</body>
  </format>'''

SEARCH_ROOTS = [
    os.environ.get("ProgramFiles", r"C:\Program Files"),
    os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"),
    r"D:\Program Files",
]

# Exact paths to try first (fastest)
DEFAULT_PATHS = [
    os.path.join(root, "Epic Games", "RealityScan")
    for root in SEARCH_ROOTS
] + [
    r"C:\Program Files\Capturing Reality\RealityCapture",
]

# Glob patterns for versioned installs (e.g. RealityScan_2.1, RealityCapture_1.4)
GLOB_PATTERNS = [
    os.path.join(root, "Epic Games", "RealityScan*")
    for root in SEARCH_ROOTS
] + [
    os.path.join(root, "Capturing Reality", "RealityCapture*")
    for root in SEARCH_ROOTS
]


def find_rs_install() -> str:
    """Auto-detect RealityScan installation directory.

    Checks exact paths first, then expands glob patterns to catch versioned
    installs like RealityScan_2.1 or RealityCapture_1.4.
    """
    # Exact matches first
    for path in DEFAULT_PATHS:
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "Report.xml")):
            return path

    # Glob for versioned directories
    for pattern in GLOB_PATTERNS:
        for path in sorted(glob.glob(pattern), reverse=True):  # newest version first
            if os.path.isdir(path) and os.path.isfile(os.path.join(path, "Report.xml")):
                return path

    return ""


def get_template_source() -> str:
    """Find our template file relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(script_dir, "templates", TEMPLATE_FILENAME)
    if not os.path.isfile(template_path):
        print(f"ERROR: Template not found at {template_path}", file=sys.stderr)
        sys.exit(1)
    return template_path


def guid_in_report_xml(report_xml_path: str) -> bool:
    """Check if our GUID is already registered in Report.xml."""
    with open(report_xml_path, "r", encoding="utf-8") as f:
        content = f.read()
    return TEMPLATE_GUID in content


def install_to_report_xml(report_xml_path: str, dry_run: bool = False) -> bool:
    """Add our format entry to Report.xml, backing up first."""
    with open(report_xml_path, "r", encoding="utf-8") as f:
        content = f.read()

    if TEMPLATE_GUID in content:
        print(f"  Already registered in Report.xml (GUID {TEMPLATE_GUID})")
        return False

    # Find the closing </formats> tag and insert before it
    close_tag = "</formats>"
    if close_tag not in content:
        print(f"ERROR: Could not find {close_tag} in Report.xml", file=sys.stderr)
        print("  You may need to add the entry manually. See the template file for the XML snippet.", file=sys.stderr)
        sys.exit(1)

    new_content = content.replace(close_tag, FORMAT_ENTRY + "\n" + close_tag)

    if dry_run:
        print(f"  [DRY RUN] Would add to Report.xml:")
        print(f"  {FORMAT_ENTRY[:80]}...")
        return True

    # Backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = report_xml_path + f".backup_{timestamp}"
    shutil.copy2(report_xml_path, backup_path)
    print(f"  Backed up Report.xml to {backup_path}")

    with open(report_xml_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    print(f"  Registered template in Report.xml")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Install LitchShip quality report template into RealityScan"
    )
    parser.add_argument("--rs-path", type=str, default=None,
                        help="Path to RealityScan installation directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without making changes")
    args = parser.parse_args()

    # Find RealityScan
    rs_path = args.rs_path or find_rs_install()
    if not rs_path or not os.path.isdir(rs_path):
        searched = DEFAULT_PATHS + GLOB_PATTERNS
        print("ERROR: RealityScan installation not found.", file=sys.stderr)
        print("  Searched:", file=sys.stderr)
        for p in searched:
            print(f"    {p}", file=sys.stderr)
        print(f'\n  Use --rs-path to specify the installation directory.', file=sys.stderr)
        print(f'  NOTE: Quote paths with spaces:', file=sys.stderr)
        print(f'    python install_template.py --rs-path "C:\\Program Files\\Epic Games\\RealityScan_2.1"', file=sys.stderr)
        sys.exit(1)

    print(f"RealityScan found at: {rs_path}")

    reports_dir = os.path.join(rs_path, "Reports")
    report_xml = os.path.join(rs_path, "Report.xml")

    if not os.path.isdir(reports_dir):
        if args.dry_run:
            print(f"  [DRY RUN] Would create {reports_dir}")
        else:
            os.makedirs(reports_dir, exist_ok=True)
            print(f"  Created {reports_dir}")

    # Copy template
    template_src = get_template_source()
    template_dst = os.path.join(reports_dir, TEMPLATE_FILENAME)

    if args.dry_run:
        print(f"  [DRY RUN] Would copy:")
        print(f"    {template_src}")
        print(f"    -> {template_dst}")
    else:
        shutil.copy2(template_src, template_dst)
        print(f"  Copied template to {template_dst}")

    # Register in Report.xml
    if not os.path.isfile(report_xml):
        print(f"\n  WARNING: Report.xml not found at {report_xml}", file=sys.stderr)
        print(f"  You'll need to add the format entry manually.", file=sys.stderr)
        print(f"  See the template file comments for the XML snippet.", file=sys.stderr)
    else:
        install_to_report_xml(report_xml, dry_run=args.dry_run)

    # Print export instructions
    print(f"\n{'=' * 60}")
    print(f"  Installation {'would be ' if args.dry_run else ''}complete!")
    print(f"{'=' * 60}")
    print(f"\n  To export the quality report:")
    print(f"")
    print(f"  GUI:")
    print(f"    ALIGNMENT tab -> Export -> Report")
    print(f"    Select \"{TEMPLATE_DESC}\"")
    print(f"    Save as quality_report.json")
    print(f"")
    print(f"  CLI:")
    print(f'    RealityScan.exe -load project.rsproj \\')
    print(f'      -exportReport "C:\\output\\quality_report.json" "{TEMPLATE_GUID}"')
    print(f"")
    print(f"  Post-process (fixes JSON formatting):")
    print(f"    python fix_report_json.py quality_report.json")
    print(f"")
    print(f"  Then copy quality_report.json to your dataset directory")
    print(f"  alongside the COLMAP files before running analyze_colmap.py.")


if __name__ == "__main__":
    main()
