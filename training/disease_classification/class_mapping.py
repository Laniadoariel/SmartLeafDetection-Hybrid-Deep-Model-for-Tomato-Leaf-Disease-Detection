"""Canonical tomato-leaf disease taxonomy + cross-dataset label mapping.

The project's disease datasets each use their own naming convention and class
count (tomato_diseases/merged: 7 classes, tomato_6k: 5, PlantDoc: 30 of which 9
are tomato). To train a single classifier we unify every source label into one
canonical, snake_case taxonomy.

This module is the single source of truth for that mapping. The dataset builder
and the evaluation/comparison scripts all import from here so the mapping is
never duplicated.
"""

from __future__ import annotations

# Canonical class list (stable order). Folder names in the generated dataset
# and the entries in classes.json use exactly these strings.
CANONICAL_CLASSES: list[str] = [
    "healthy",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "target_spot",
    "spider_mites",
    "mosaic_virus",
    "yellow_leaf_curl_virus",
    "powdery_mildew",
    "black_spot",
]

# Exact source-label -> canonical-class mapping. Keys are written in readable
# form; they are normalized via ``_normalize`` at construction (below), so it
# does not matter whether a key uses spaces or underscores. A value of ``None``
# means "ignore this label" (e.g. non-tomato PlantDoc classes).
_RAW_LABELS: dict[str, str | None] = {
    # --- tomato_diseases (Roboflow/Sylhet) & merged_diseases (7 classes) ---
    "Bacterial Spot": "bacterial_spot",
    "Early_Blight": "early_blight",
    "Healthy": "healthy",
    "Late_blight": "late_blight",
    "Leaf Mold": "leaf_mold",
    "Target_Spot": "target_spot",
    "black spot": "black_spot",
    # --- tomato_6k (YOLOv5, 5 classes) ---
    "bacterial_spot": "bacterial_spot",
    "early_blight": "early_blight",
    "late_blight": "late_blight",
    "powdery_mildew": "powdery_mildew",
    # --- PlantDoc tomato classes (9 of 30; rest map to None) ---
    "Tomato Early blight leaf": "early_blight",
    "Tomato Septoria leaf spot": "septoria_leaf_spot",
    "Tomato leaf": "healthy",
    "Tomato leaf bacterial spot": "bacterial_spot",
    "Tomato leaf late blight": "late_blight",
    "Tomato leaf mosaic virus": "mosaic_virus",
    "Tomato leaf yellow virus": "yellow_leaf_curl_virus",
    "Tomato mold leaf": "leaf_mold",
    "Tomato two spotted spider mites leaf": "spider_mites",
    # --- PlantDoc non-tomato classes: explicitly ignored ---
    "Apple Scab Leaf": None,
    "Apple leaf": None,
    "Apple rust leaf": None,
    "Bell_pepper leaf": None,
    "Bell_pepper leaf spot": None,
    "Blueberry leaf": None,
    "Cherry leaf": None,
    "Corn Gray leaf spot": None,
    "Corn leaf blight": None,
    "Corn rust leaf": None,
    "Peach leaf": None,
    "Potato leaf": None,
    "Potato leaf early blight": None,
    "Potato leaf late blight": None,
    "Raspberry leaf": None,
    "Soyabean leaf": None,
    "Soybean leaf": None,
    "Squash Powdery mildew leaf": None,
    "Strawberry leaf": None,
    "grape leaf": None,
    "grape leaf black rot": None,
}


def _normalize(label: str) -> str:
    """Normalize a raw label for case/whitespace/underscore-insensitive lookup."""
    return " ".join(str(label).strip().lower().replace("_", " ").split())


# Lookup table keyed by the NORMALIZED label form (built once).
_RAW_TO_CANONICAL: dict[str, str | None] = {_normalize(k): v for k, v in _RAW_LABELS.items()}


def map_label(raw_label: str) -> str | None:
    """Map a raw dataset label to a canonical class.

    Returns the canonical class string, or ``None`` if the label is explicitly
    ignored. Raises ``KeyError`` for labels not present in the mapping.
    """
    key = _normalize(raw_label)
    if key not in _RAW_TO_CANONICAL:
        raise KeyError(f"Unmapped source label: {raw_label!r} (normalized {key!r})")
    return _RAW_TO_CANONICAL[key]


# Sentinel returned (instead of raising) for labels we don't recognize, so a
# long dataset build reports them rather than crashing on the first one.
UNMAPPED = "__UNMAPPED__"


def build_id_to_canonical(names: list[str] | dict[int, str]) -> dict[int, str | None]:
    """Build a class-id -> canonical mapping for one dataset's ``names``.

    ``names`` is the YOLO ``data.yaml`` names entry (a list indexed by class id,
    or a ``{id: name}`` dict). Unknown labels map to :data:`UNMAPPED` (the caller
    records and skips them) rather than raising.
    """
    if isinstance(names, dict):
        items = sorted(names.items())
    else:
        items = list(enumerate(names))
    out: dict[int, str | None] = {}
    for idx, name in items:
        try:
            out[int(idx)] = map_label(name)
        except KeyError:
            out[int(idx)] = UNMAPPED
    return out
