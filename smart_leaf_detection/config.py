"""Pipeline configuration with validation and file loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from smart_leaf_detection.errors import ConfigValidationError

_DEFAULT_CLASS_NAMES: list[str] = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
]

_VALID_OUTPUT_FORMATS = {"json", "csv"}


@dataclass
class PipelineConfig:
    """Central configuration for the SmartLeafDetection pipeline.

    All parameters have documented defaults except ``video_path`` which is required.
    Validation runs automatically on construction via ``__post_init__``.
    """

    # Frame extraction
    video_path: str = ""
    extraction_rate: int | None = None
    target_fps: float | None = None

    # Plant detection
    plant_weights_path: str = "yolo11_plants.pt"
    plant_confidence_threshold: float = 0.25

    # Plant tracking
    bytetrack_config_path: str | None = None

    # ROI cropping
    roi_padding: float = 0.1

    # Leaf detection
    # Trained leaf detector produced by training/leaf_detection/. Falls back to
    # the legacy name if the new artifact is absent (see LeafDetector loading).
    leaf_weights_path: str = "weights/leaf_best.pt"
    leaf_confidence_threshold: float = 0.25

    # Disease classification
    classifier_weights_path: str = "resnet50_tomato.pt"
    class_names: list[str] = field(default_factory=lambda: list(_DEFAULT_CLASS_NAMES))
    # "auto" resolves to CUDA -> MPS (Apple Silicon) -> CPU at runtime so the
    # same config works on Windows/Linux (NVIDIA) and macOS without edits.
    device: str = "auto"

    # Temporal aggregation
    aggregation_window: int = 30
    aggregation_confidence_threshold: float = 0.6
    aggregation_majority_ratio: float = 0.6
    dual_threshold_enabled: bool = True

    # Plant status
    top_k_diseases: int = 3

    # GPS
    srt_path: str | None = None

    # Export
    output_format: str = "json"
    severity_enabled: bool = False
    output_path: str = "disease_report.json"

    # Performance
    target_processing_fps: float = 20.0

    def __post_init__(self) -> None:
        self.validate()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Validate all configuration parameters.

        Raises ``ConfigValidationError`` with the parameter name and a
        human-readable constraint description when a value is invalid.
        """
        # aggregation_window must be in [15, 60]
        if not (15 <= self.aggregation_window <= 60):
            raise ConfigValidationError(
                "aggregation_window",
                "must be between 15 and 60 inclusive",
            )

        # Confidence thresholds must be non-negative
        if self.plant_confidence_threshold < 0:
            raise ConfigValidationError(
                "plant_confidence_threshold",
                "must be non-negative",
            )
        if self.leaf_confidence_threshold < 0:
            raise ConfigValidationError(
                "leaf_confidence_threshold",
                "must be non-negative",
            )
        if self.aggregation_confidence_threshold < 0:
            raise ConfigValidationError(
                "aggregation_confidence_threshold",
                "must be non-negative",
            )
        if self.aggregation_majority_ratio < 0:
            raise ConfigValidationError(
                "aggregation_majority_ratio",
                "must be non-negative",
            )

        # class_names must be non-empty
        if not self.class_names:
            raise ConfigValidationError(
                "class_names",
                "must be a non-empty list",
            )

        # output_format must be json or csv
        if self.output_format not in _VALID_OUTPUT_FORMATS:
            raise ConfigValidationError(
                "output_format",
                f"must be one of {sorted(_VALID_OUTPUT_FORMATS)}",
            )

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        """Load a ``PipelineConfig`` from a YAML or JSON file.

        The file extension determines the parser (``.yaml`` / ``.yml`` for
        YAML, ``.json`` for JSON).  Missing keys use the documented defaults.

        Raises ``ConfigValidationError`` if the file cannot be read or parsed.
        """
        path = Path(path)

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigValidationError(
                "config_file",
                f"cannot read config file '{path}': {exc}",
            ) from exc

        suffix = path.suffix.lower()
        try:
            if suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(raw) or {}
            elif suffix == ".json":
                data = json.loads(raw)
            else:
                raise ConfigValidationError(
                    "config_file",
                    f"unsupported config file format '{suffix}'; use .yaml, .yml, or .json",
                )
        except (yaml.YAMLError, json.JSONDecodeError) as exc:
            raise ConfigValidationError(
                "config_file",
                f"failed to parse config file '{path}': {exc}",
            ) from exc

        if not isinstance(data, dict):
            raise ConfigValidationError(
                "config_file",
                "config file must contain a mapping at the top level",
            )

        return cls(**data)
