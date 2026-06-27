#!/usr/bin/env python3
"""Generate ``CLASS_MAPPING_REPORT.md`` — a human-reviewable table of every
source label across the project's disease datasets and the canonical class it
maps to, plus any labels that are unmapped.

This makes the canonical-taxonomy decisions auditable for the final-project
documentation. It reads each dataset's real ``data.yaml`` ``names`` so the
report always matches what ``build_classification_dataset.py`` actually does.

Usage:
    python training/disease_classification/dump_class_mapping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "training" / "disease_classification"))

from build_classification_dataset import DATASET_CONFIGS, _load_names  # noqa: E402
from class_mapping import CANONICAL_CLASSES, _normalize, map_label  # noqa: E402


def write_mapping_report(out_path: Path, datasets: list[str] | None = None) -> Path:
    datasets = datasets or list(DATASET_CONFIGS)
    rows: list[tuple[str, str, str, str]] = []   # dataset, original, normalized, canonical/ignored
    unmapped: list[tuple[str, str]] = []          # dataset, original

    for ds in datasets:
        names_yaml = PROJECT_ROOT / DATASET_CONFIGS[ds]["names_yaml"]
        if not names_yaml.exists():
            continue
        names = _load_names(names_yaml)
        items = sorted(names.items()) if isinstance(names, dict) else list(enumerate(names))
        for _idx, name in items:
            try:
                canon = map_label(name)
                rows.append((ds, str(name), _normalize(name),
                             canon if canon is not None else "(ignored — non-tomato)"))
            except KeyError:
                unmapped.append((ds, str(name)))

    lines = ["# CLASS_MAPPING_REPORT", "",
             "Auto-generated from each dataset's `data.yaml`. Shows how every source",
             "label maps into the canonical tomato-leaf disease taxonomy.", "",
             f"**Canonical classes ({len(CANONICAL_CLASSES)}):** "
             + ", ".join(f"`{c}`" for c in CANONICAL_CLASSES), "",
             "| Dataset | Original Label | Normalized | Canonical Label |",
             "|---|---|---|---|"]
    for ds, original, norm, canon in rows:
        lines.append(f"| {ds} | {original} | {norm} | {canon} |")

    lines += ["", "## UNMAPPED LABELS", ""]
    if unmapped:
        lines.append("These labels are not recognized and are **skipped** by the builder "
                     "(reported, never silently dropped). Add them to `class_mapping._RAW_LABELS` "
                     "if they should contribute crops:")
        lines.append("")
        for ds, original in unmapped:
            lines.append(f"- `{original}`  (dataset: {ds})")
    else:
        lines.append("_None — every source label in the configured datasets is mapped._")

    # Summary counts per canonical class.
    counts: dict[str, int] = {}
    for _ds, _o, _n, canon in rows:
        counts[canon] = counts.get(canon, 0) + 1
    lines += ["", "## Source-label count per canonical class", "",
              "| Canonical Label | # source labels mapped to it |", "|---|---|"]
    for canon in sorted(counts, key=lambda c: (-counts[c], c)):
        lines.append(f"| {canon} | {counts[canon]} |")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    out = (PROJECT_ROOT / "training" / "disease_classification" / "reports"
           / "CLASS_MAPPING_REPORT.md")
    path = write_mapping_report(out)
    print(f"Wrote {path}")
    print("Review the UNMAPPED LABELS section to confirm nothing important is dropped.")


if __name__ == "__main__":
    main()
