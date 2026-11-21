"""Test the trained model on new images from the command line.

For every image it prints the detected hands and the winner, and saves a copy
with the boxes drawn on it.

Examples:
    python predict.py photo.jpg
    python predict.py my_new_photos/              # every image in a folder
    python predict.py my_new_photos/ --conf 0.4 --out results/
"""

import argparse
import csv
from pathlib import Path

import cv2

from rps import DEFAULT_WEIGHTS, load_model, play

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(description="Run the Rock Paper Scissors model on images")
    p.add_argument("source", help="an image file or a folder of images")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="path to the trained best.pt")
    p.add_argument("--conf", type=float, default=0.5, help="confidence threshold (default 0.5)")
    p.add_argument("--out", default="predictions", help="folder for annotated images and results.csv")
    return p.parse_args()


def collect_images(source):
    source = Path(source)
    if source.is_dir():
        return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_TYPES)
    if source.is_file():
        return [source]
    raise FileNotFoundError(f"Not found: {source}")


def main():
    args = parse_args()
    model = load_model(args.weights)
    images = collect_images(args.source)
    if not images:
        raise SystemExit(f"No images found in {args.source}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for path in images:
        annotated, rnd = play(model, str(path), conf=args.conf)
        cv2.imwrite(str(out_dir / f"{path.stem}_pred.jpg"), annotated)

        hands = ", ".join(f"{h.gesture} ({h.confidence:.2f})" for h in rnd.hands) or "nothing detected"
        print(f"{path.name}: {hands}  ->  {rnd.verdict}")
        rows.append([path.name, len(rnd.hands), hands, rnd.verdict])

    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "hands_found", "detections", "verdict"])
        writer.writerows(rows)

    print(f"\n{len(images)} image(s) done. Annotated images and results.csv saved to {out_dir}/")


if __name__ == "__main__":
    main()
