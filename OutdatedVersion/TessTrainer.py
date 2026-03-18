import os
import subprocess

# -----------------------------
# Configuration
# -----------------------------
dataset_dir = r"U:\TesseractTraining\BatchRenders"  # folder with PNG + .gt.txt + .box
output_dir = r"U:\TesseractTraining\TrFiles"
traineddata_name = "mtg_ocr"  # final traineddata filename
batch_size = 1000  # number of files per batch
start_index = 0    # where to start in this run (0-based)
tesseract_cmd = "tesseract"  # make sure this is in PATH

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Gather all image files
# -----------------------------
image_files = [f for f in os.listdir(dataset_dir) if f.lower().endswith(".png")]
image_files.sort()  # sort alphabetically for consistency

# Apply batch slicing
image_files = image_files[start_index:start_index + batch_size]

if not image_files:
    print("No files in this batch. Exiting.")
    exit()

# -----------------------------
# Training loop with progress + resume
# -----------------------------
tr_files = []
total = len(image_files)

for i, img_file in enumerate(image_files, start=1):
    base, _ = os.path.splitext(img_file)
    png_path = os.path.join(dataset_dir, img_file)
    box_path = os.path.join(dataset_dir, base + ".box")
    tr_output = os.path.join(output_dir, base + ".tr")

    # Progress line
    print(f"Training {i}/{total} files", end="\r")

    # Skip if .tr already exists (resume feature)
    if os.path.exists(tr_output):
        tr_files.append(tr_output)
        continue

    if not os.path.exists(box_path):
        print(f"\nSkipping {img_file}: no .box file found.")
        continue

    subprocess.run([
        tesseract_cmd,
        png_path,
        tr_output.replace(".tr", ""),  # output basename (no extension)
        "nobatch",
        "box.train"
    ], check=True)

    tr_files.append(tr_output)

print(f"\n✅ Finished training {len(tr_files)}/{total} files")

# -----------------------------
# Combine all .tr files into one
# -----------------------------
combined_tr_path = os.path.join(output_dir, "training_data.tr")
with open(combined_tr_path, "wb") as outfile:
    for tr_file in tr_files:
        with open(tr_file, "rb") as infile:
            outfile.write(infile.read())

print(f"📦 Combined {len(tr_files)} .tr files into {combined_tr_path}")

# -----------------------------
# Next training step
# -----------------------------
print("\n👉 You can now continue with Tesseract training on this batch:")
print(f"tesseract {combined_tr_path} {traineddata_name} --training")
