"""
box_overlay_visual.py

Visual QA tool for MTG OCR training data.
Draws .box file rectangles onto their corresponding .png renders
and saves the result to an output folder for manual spot checking.

Each output image shows:
  - The original render
  - A coloured bounding box per character
  - The character label above each box

Usage:
    # Check all files in dataset (saves overlays to ./overlay_output/)
    python box_overlay_visual.py --dataset U:/TesseractTraining/BatchRenders

    # Check a random sample of 50 files
    python box_overlay_visual.py --dataset U:/TesseractTraining/BatchRenders --sample 50

    # Specify custom output directory
    python box_overlay_visual.py --dataset U:/TesseractTraining/BatchRenders --output U:/TesseractTraining/QA
"""

import os
import argparse
import random
from PIL import Image, ImageDraw, ImageFont

# -----------------------------
# Config
# -----------------------------
BOX_COLOR       = (255, 80, 80)     # red boxes
TEXT_COLOR      = (255, 220, 0)     # yellow labels
BOX_THICKNESS   = 2
FONT_SIZE       = 14


# -----------------------------
# Helpers
# -----------------------------

def parse_box_file(box_path):
    entries = []
    with open(box_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            ch = parts[0]
            try:
                x1, y1, x2, y2 = map(int, parts[1:5])
                entries.append((ch, x1, y1, x2, y2))
            except ValueError:
                continue
    return entries


def get_label_font(size):
    """Try to load a basic font, fall back to PIL default if unavailable."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def draw_overlay(png_path, box_path, out_path, font):
    """Draw box overlays onto the image and save to out_path."""
    entries = parse_box_file(box_path)

    with Image.open(png_path).convert("RGB") as img:
        img_w, img_h = img.size
        draw = ImageDraw.Draw(img)

        for i, (ch, x1, y1, x2, y2) in enumerate(entries):
            # Skip degenerate boxes instead of crashing
            if x1 >= x2 or y1 >= y2:
                continue
            # Draw rectangle
            for t in range(BOX_THICKNESS):
                draw.rectangle(
                    [x1 - t, y1 - t, x2 + t, y2 + t],
                    outline=BOX_COLOR
                )

            # Draw character label above box
            label_x = x1
            label_y = max(0, y1 - FONT_SIZE - 2)
            draw.text((label_x, label_y), ch, fill=TEXT_COLOR, font=font)

        # Draw entry count in corner
        summary = f"{len(entries)} boxes"
        draw.text((4, 4), summary, fill=(0, 255, 0), font=font)

        img.save(out_path)


# -----------------------------
# Main
# -----------------------------

def run_overlay(dataset_dir, output_dir, sample_size):
    os.makedirs(output_dir, exist_ok=True)

    # Find all bases that have both .png and .box
    png_bases = {os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.lower().endswith(".png")}
    box_bases = {os.path.splitext(f)[0] for f in os.listdir(dataset_dir) if f.lower().endswith(".box")}
    valid_bases = sorted(png_bases & box_bases)

    if not valid_bases:
        print("No matching .png + .box pairs found. Exiting.")
        return

    # Apply sample if requested
    if sample_size and sample_size < len(valid_bases):
        valid_bases = random.sample(valid_bases, sample_size)
        valid_bases.sort()
        print(f"Sampling {sample_size} files from {len(png_bases & box_bases)} available.")
    else:
        print(f"Processing all {len(valid_bases)} files.")

    font = get_label_font(FONT_SIZE)
    failed = []

    for i, base in enumerate(valid_bases, start=1):
        png_path = os.path.join(dataset_dir, base + ".png")
        box_path = os.path.join(dataset_dir, base + ".box")
        out_path = os.path.join(output_dir, base + "_overlay.png")

        print(f"  [{i}/{len(valid_bases)}] {base}", end="\r")

        try:
            draw_overlay(png_path, box_path, out_path, font)
        except Exception as e:
            failed.append((base, str(e)))

    print()
    print("\n" + "=" * 60)
    print(f"  MTG OCR Visual QA")
    print(f"  Input  : {dataset_dir}")
    print(f"  Output : {output_dir}")
    print("=" * 60)
    print(f"  Processed : {len(valid_bases)}")
    print(f"  Failed    : {len(failed)}")
    if failed:
        print("\n  Failures:")
        for base, err in failed:
            print(f"    [{base}] {err}")
    else:
        print(f"\n  All overlays saved to {output_dir}")
        print("  Open the folder and eyeball them :)")
    print("=" * 60 + "\n")


# -----------------------------
# Entry point
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visual QA overlay for MTG OCR training data.")
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to folder containing .png and .box files"
    )
    parser.add_argument(
        "--output",
        default="overlay_output",
        help="Path to save overlay images (default: ./overlay_output/)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of random files to sample (default: all)"
    )
    args = parser.parse_args()
    run_overlay(args.dataset, args.output, args.sample)
