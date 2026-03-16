#!/usr/bin/env python3
"""
fix_report_json.py — Post-process RealityScan quality report JSON.

RealityScan's template engine may produce invalid JSON due to:
  - COMMA_PLACEHOLDER tokens (used to avoid $If argument parsing conflicts)
  - Trailing commas in arrays/objects
  - HTML comment blocks (<!-- ... -->) appended from the template

This script cleans up all of these issues and validates the result.

Usage:
    python fix_report_json.py quality_report.json
    python fix_report_json.py quality_report.json --output cleaned.json
    python fix_report_json.py quality_report.json --in-place
"""

import argparse
import json
import re
import sys
from pathlib import Path


def clean_report_json(raw: str) -> str:
    """Clean raw RealityScan report output into valid JSON."""
    text = raw

    # Strip HTML comments (template documentation block)
    text = re.sub(r"<!--[\s\S]*?-->", "", text)

    # Replace COMMA_PLACEHOLDER with actual commas
    text = text.replace("COMMA_PLACEHOLDER", ",")

    # Remove trailing commas before } or ]
    # Handles: {"key": "val",} and [1, 2, 3,]
    text = re.sub(r",\s*([}\]])", r"\1", text)

    # Remove any leading/trailing whitespace
    text = text.strip()

    return text


def validate_json(text: str) -> dict:
    """Parse JSON and return the parsed object. Raises on failure."""
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser(
        description="Post-process RealityScan quality report JSON"
    )
    parser.add_argument("input", type=Path, help="Path to raw quality_report.json from RealityScan")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path (default: prints to stdout)")
    parser.add_argument("--in-place", action="store_true",
                        help="Overwrite the input file with cleaned JSON")
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"ERROR: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    raw = args.input.read_text(encoding="utf-8")
    cleaned = clean_report_json(raw)

    try:
        data = validate_json(cleaned)
    except json.JSONDecodeError as e:
        print(f"ERROR: JSON still invalid after cleanup: {e}", file=sys.stderr)
        print(f"  Line {e.lineno}, column {e.colno}", file=sys.stderr)
        # Write the partially cleaned output so user can debug
        debug_path = args.input.with_suffix(".debug.json")
        debug_path.write_text(cleaned, encoding="utf-8")
        print(f"  Partially cleaned output written to {debug_path}", file=sys.stderr)
        sys.exit(1)

    # Pretty-print the valid JSON
    output_text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    # Summary
    cam_count = len(data.get("cameras", []))
    component = data.get("component", {})
    print(f"  Cleaned: {cam_count} cameras, "
          f"medianError={component.get('medianError', 'N/A')}, "
          f"avgTrackLength={component.get('avgTrackLength', 'N/A')}",
          file=sys.stderr)

    if args.in_place:
        args.input.write_text(output_text, encoding="utf-8")
        print(f"  Written to {args.input}", file=sys.stderr)
    elif args.output:
        args.output.write_text(output_text, encoding="utf-8")
        print(f"  Written to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(output_text)


if __name__ == "__main__":
    main()
