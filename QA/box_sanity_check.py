"""
box_sanity_check.py

Automated sanity checker for MTG OCR training data.
Validates every .box file against its .gt.txt counterpart.

Checks performed:
  1. Missing files        — .box or .gt.txt exists without its counterpart
  2. Line count mismatch  — number of box entries != number of characters in gt.txt
  3. Zero coordinates     — any box with 0 0 0 0 (degenerate projection)
  4. Oversized boxes      — box dimensions exceed image resolution
  5. Overlapping boxes    — two boxes overlap by more than OVERLAP_THRESHOLD

Usage:
    python box_sanity_check.py --dataset U:/TesseractTraining/BatchRenders
    python box_sanity_check.py --dataset U:/TesseractTraining/BatchRenders --overlap 0.3
"""

import os
import argparse
from PIL import Image

# -----------------------------
# Config
# -----------------------------
OVERLAP_THRESHOLD = 0.5  # flag if two boxes overlap by more than this fraction


# -----------------------------
# Helpers
# -----------------------------

def parse_box_file(box_path):
    """
    Parse a .box file into a list of (char, x1, y1, x2, y2) tuples.
    Returns (entries, errors) where errors is a list of string descriptions.
    """
    entries = []
    errors = []
    with open(box_path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 5:
                errors.append(f"  Line {lineno}: malformed — '{line.strip()}'")
                continue
            ch = parts[0]
            try:
                x1, y1, x2, y2 = map(int, parts[1:5])
            except ValueError:
                errors.append(f"  Line {lineno}: non-integer coordinates — '{line.strip()}'")
                continue
            entries.append((ch, x1, y1, x2, y2))
    return entries, errors


def read_gt_text(gt_path):
    """Read ground truth text, stripping trailing newline."""
    with open(gt_path, "r", encoding="utf-8") as f:
        return f.read().rstrip("\n")


def get_image_size(png_path):
    """Return (width, height) of image, or None if unreadable."""
    try:
        with Image.open(png_path) as img:
            return img.size
    except Exception:
        return None


def compute_overlap_fraction(a, b):
    """
    Compute what fraction of the smaller box is overlapped by the larger.
    a, b are (x1, y1, x2, y2) tuples.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    smaller_area = min(area_a, area_b)

    return intersection / smaller_area


# -----------------------------
# Main checker
# -----------------------------

def check_dataset(dataset_dir, overlap_threshold):
    png_files  = {os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.lower().endswith(".png")}
    box_files  = {os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.lower().endswith(".box")}
    gt_files = {f[:-len(".gt.txt")] for f in os.listdir(dataset_dir) if f.lower().endswith(".gt.txt")}

    all_bases = png_files | box_files | gt_files
    all_bases = sorted(all_bases)

    total        = 0
    passed       = 0
    failed       = 0
    all_issues   = []

    for base in all_bases:
        issues = []

        png_path = os.path.join(dataset_dir, base + ".png")
        box_path = os.path.join(dataset_dir, base + ".box")
        gt_path  = os.path.join(dataset_dir, base + ".gt.txt")

        has_png = base in png_files
        has_box = base in box_files
        has_gt  = base in gt_files

        # --- Check 1: missing files ---
        if not has_box:
            issues.append("  MISSING: .box file")
        if not has_gt:
            issues.append("  MISSING: .gt.txt file")
        if not has_png:
            issues.append("  MISSING: .png file")

        if not has_box or not has_gt:
            failed += 1
            all_issues.append((base, issues))
            continue

        # --- Parse files ---
        entries, parse_errors = parse_box_file(box_path)
        issues.extend(parse_errors)

        gt_text = read_gt_text(gt_path)
        gt_chars = list(gt_text)

        img_size = get_image_size(png_path) if has_png else None
        img_w = img_size[0] if img_size else None
        img_h = img_size[1] if img_size else None

        # --- Check 2: line count mismatch ---
        if len(entries) != len(gt_chars):
            issues.append(
                f"  LINE COUNT MISMATCH: .box has {len(entries)} entries, "
                f".gt.txt has {len(gt_chars)} chars  |  gt='{gt_text}'"
            )

        for i, (ch, x1, y1, x2, y2) in enumerate(entries):
            # --- Check 3: zero coordinates ---
            if x1 == 0 and y1 == 0 and x2 == 0 and y2 == 0:
                issues.append(f"  ZERO COORDS: entry {i} char='{ch}'")

            # --- Check 4: oversized boxes ---
            if img_w and img_h:
                if x2 > img_w or y2 > img_h:
                    issues.append(
                        f"  OVERSIZED BOX: entry {i} char='{ch}' "
                        f"({x1},{y1},{x2},{y2}) exceeds image ({img_w}x{img_h})"
                    )
                if x1 < 0 or y1 < 0:
                    issues.append(
                        f"  NEGATIVE COORDS: entry {i} char='{ch}' ({x1},{y1},{x2},{y2})"
                    )

            # --- Check 5: zero-size boxes ---
            if x1 >= x2 or y1 >= y2:
                issues.append(
                    f"  DEGENERATE BOX: entry {i} char='{ch}' "
                    f"({x1},{y1},{x2},{y2}) has zero or negative area"
                )

        # --- Check 5: overlapping boxes ---
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                _, x1a, y1a, x2a, y2a = entries[i]
                _, x1b, y1b, x2b, y2b = entries[j]
                overlap = compute_overlap_fraction(
                    (x1a, y1a, x2a, y2a),
                    (x1b, y1b, x2b, y2b)
                )
                if overlap > overlap_threshold:
                    issues.append(
                        f"  OVERLAP: entries {i}('{entries[i][0]}') and "
                        f"{j}('{entries[j][0]}') overlap {overlap:.1%}"
                    )

        total += 1
        if issues:
            failed += 1
            all_issues.append((base, issues))
        else:
            passed += 1

    # --- Report ---
    print("\n" + "=" * 60)
    print(f"  MTG OCR Dataset Sanity Check")
    print(f"  Directory: {dataset_dir}")
    print("=" * 60)
    print(f"  Total samples checked : {total}")
    print(f"  Passed                : {passed}")
    print(f"  Failed                : {failed}")
    print("=" * 60)

    if all_issues:
        print(f"\n  Issues found in {len(all_issues)} file(s):\n")
        for base, issues in all_issues:
            print(f"  [{base}]")
            for issue in issues:
                print(issue)
            print()
    else:
        print("\n  All files passed. You're good to go :)")

    print("=" * 60 + "\n")
    return failed == 0


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check MTG OCR training data.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to folder containing .png, .gt.txt, and .box files"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=OVERLAP_THRESHOLD,
        help=f"Overlap fraction threshold to flag boxes (default: {OVERLAP_THRESHOLD})"
    )
    args = parser.parse_args()
    check_dataset(args.dataset, args.overlap)
