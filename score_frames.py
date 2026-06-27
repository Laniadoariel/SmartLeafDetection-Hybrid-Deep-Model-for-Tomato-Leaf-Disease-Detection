"""Score extracted frames for how clearly leaves/vegetation are visible.

Combines two signals:
  - Sharpness: variance of the Laplacian (higher = less blur).
  - Vegetation: fraction of pixels that are green in HSV space.

Frames are ranked by a combined score and the top-N are copied to a folder.
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


def green_ratio(bgr: np.ndarray) -> float:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Green hue range in OpenCV (H: 0-179). ~35-85 covers green leaves.
    lower = np.array([25, 40, 40])
    upper = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    return float(np.count_nonzero(mask)) / mask.size


def sharpness(bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score frames for leaf visibility")
    parser.add_argument("--frames-dir", default="video_frames")
    parser.add_argument("--out-dir", default="best_leaf_frames")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--min-green", type=float, default=0.05,
                        help="Skip frames with green ratio below this (default 0.05)")
    args = parser.parse_args()

    frame_paths = sorted(Path(args.frames_dir).glob("*.jpg"))
    if not frame_paths:
        print(f"No frames found in {args.frames_dir}")
        return

    rows = []
    for fp in frame_paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        g = green_ratio(img)
        s = sharpness(img)
        rows.append((fp.name, g, s))

    if not rows:
        print("No readable frames.")
        return

    # Normalize sharpness to 0-1 across the set so it is comparable to green ratio.
    max_s = max(r[2] for r in rows) or 1.0
    scored = []
    for name, g, s in rows:
        s_norm = s / max_s
        # Combined score: vegetation presence weighted with sharpness.
        combined = (0.6 * g) + (0.4 * s_norm)
        scored.append((name, g, s, s_norm, combined))

    # Write full CSV report.
    with open("frame_scores.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "green_ratio", "sharpness", "sharpness_norm", "combined"])
        for r in sorted(scored, key=lambda x: x[4], reverse=True):
            w.writerow([r[0], f"{r[1]:.4f}", f"{r[2]:.1f}", f"{r[3]:.4f}", f"{r[4]:.4f}"])

    # Filter by minimum green, then take top-N by combined score.
    candidates = [r for r in scored if r[1] >= args.min_green]
    candidates.sort(key=lambda x: x[4], reverse=True)
    top = candidates[: args.top_n]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, g, s, s_norm, combined in top:
        shutil.copy(Path(args.frames_dir) / name, out / name)

    print(f"Scored {len(scored)} frames. {len(candidates)} passed min-green={args.min_green}.")
    print(f"Top {len(top)} frames copied to {out}/\n")
    print(f"{'frame':<22}{'green':>8}{'sharp':>10}{'score':>8}")
    for name, g, s, s_norm, combined in top:
        print(f"{name:<22}{g:>8.3f}{s:>10.1f}{combined:>8.3f}")


if __name__ == "__main__":
    main()
