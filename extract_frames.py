"""Extract frames from a video at a fixed interval using OpenCV."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def extract(video_path: str, out_dir: str, every_n: int) -> list[str]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {video_path}\n  fps={fps:.2f} total_frames={total}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            fp = str(Path(out_dir) / f"frame_{idx:06d}.jpg")
            cv2.imwrite(fp, frame)
            saved.append(fp)
        idx += 1
    cap.release()
    print(f"Saved {len(saved)} frames to {out_dir}/")
    return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", default="video_frames")
    parser.add_argument("--every-n", type=int, default=15,
                        help="Save 1 of every N frames (default: 15)")
    args = parser.parse_args()
    extract(args.video, args.out_dir, args.every_n)
