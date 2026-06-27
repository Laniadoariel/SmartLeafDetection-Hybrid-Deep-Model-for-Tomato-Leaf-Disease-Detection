#!/usr/bin/env python3
"""Repair YOLO checkpoints whose ``train_args`` contain keys that a newer
Ultralytics no longer accepts (e.g. ``cls_pw``).

Why this exists
---------------
A checkpoint trained with an older Ultralytics stores the full training config
under ``ckpt["train_args"]``. On ``resume=True`` (or ``model.val``), the newer
Ultralytics rebuilds its config via ``get_cfg`` which *strictly* validates keys
and raises ``SyntaxError: '<key>' is not a valid YOLO argument`` for anything it
no longer knows. The training loop tolerates the extra key, but the validator
construction does not, so resume crashes right after the optimizer is built.

This script removes any key from a checkpoint's ``train_args`` that is not in
the installed Ultralytics' ``DEFAULT_CFG_DICT`` (keeping a small allowlist of
runtime keys), after writing a ``.bak`` backup. The model weights themselves are
untouched.

Usage:
    python training/leaf_detection/fix_resume_checkpoint.py \
        runs/leaves/leaf_improved/weights/last.pt \
        runs/leaves/leaf_improved/weights/best.pt
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

# Runtime keys Ultralytics writes into train_args but that are not part of the
# user-facing DEFAULT_CFG_DICT; they are safe to keep (or are ignored on load).
ALLOWLIST = {"save_dir"}


def clean_checkpoint(path: Path) -> None:
    import torch
    from ultralytics.utils import DEFAULT_CFG_DICT

    if not path.exists():
        print(f"  SKIP (missing): {path}")
        return

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    train_args = ckpt.get("train_args")
    if not isinstance(train_args, dict):
        print(f"  SKIP (no dict train_args): {path}")
        return

    stale = [k for k in train_args
             if k not in DEFAULT_CFG_DICT and k not in ALLOWLIST]
    if not stale:
        print(f"  OK (no stale keys): {path}")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)
        print(f"  Backed up -> {backup}")
    else:
        print(f"  Backup already exists -> {backup} (not overwriting)")

    for k in stale:
        train_args.pop(k, None)
    ckpt["train_args"] = train_args
    torch.save(ckpt, path)
    print(f"  Removed stale keys {stale} and saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+", help="Paths to .pt checkpoints")
    args = parser.parse_args()

    print("Repairing checkpoint train_args for the installed Ultralytics:")
    for c in args.checkpoints:
        clean_checkpoint(Path(c))
    print("Done.")


if __name__ == "__main__":
    main()
