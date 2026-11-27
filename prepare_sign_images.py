"""Build the ISL reference images (A.png to Z.png) from your own ISL dataset.

Usage:
    python prepare_sign_images.py PATH_TO_DATASET

Expected dataset layout: one folder per letter, folder name = the letter
(case-insensitive), e.g. PATH_TO_DATASET/A/img001.jpg. Folders nested one or
two levels deeper (e.g. PATH/train/A/...) are found automatically.

For each letter the script samples up to 40 images, keeps the sharpest one,
fits it onto a square light-grey canvas and saves it to assets/isl/<LETTER>.png.
Use images from the project's ISL dataset only, so the references match
the signs the model was trained on.
"""
import argparse
import os
import string
from pathlib import Path

import numpy as np
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OUT = Path(__file__).parent / "assets" / "isl"


def sharpness(img):
    gray = np.asarray(img.convert("L"), dtype=float)
    gy, gx = np.gradient(gray)
    return float(np.mean(np.hypot(gx, gy)))


def collect(root):
    found = {}
    for folder, _, files in os.walk(root):
        name = Path(folder).name.strip().upper()
        if name in string.ascii_uppercase and len(name) == 1:
            imgs = [Path(folder) / f for f in sorted(files) if Path(f).suffix.lower() in EXTS]
            if imgs:
                found.setdefault(name, []).extend(imgs)
    return found


def best_image(paths, sample=40):
    step = max(1, len(paths) // sample)
    best, best_score = None, -1.0
    for p in paths[::step][:sample]:
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        score = sharpness(img)
        if score > best_score:
            best, best_score = img, score
    return best


def square(img, size):
    img.thumbnail((size, size))
    canvas = Image.new("RGB", (size, size), (243, 244, 247))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--size", type=int, default=400)
    args = ap.parse_args()

    found = collect(args.dataset)
    OUT.mkdir(parents=True, exist_ok=True)
    done = []
    for letter in string.ascii_uppercase:
        if letter not in found:
            continue
        img = best_image(found[letter])
        if img is None:
            continue
        square(img, args.size).save(OUT / f"{letter}.png")
        done.append(letter)
    missing = [c for c in string.ascii_uppercase if c not in done]
    print(f"Saved {len(done)} of 26 images to {OUT}")
    if missing:
        print("Missing letters:", " ".join(missing))
        print("Check that the folders are named with the letter (A, B, C ...).")


if __name__ == "__main__":
    main()
