"""Tests for PipelineConfig: defaults, validation, and file loading."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from smart_leaf_detection.config import PipelineConfig, _DEFAULT_CLASS_NAMES
from smart_leaf_detection.errors import ConfigValidationError


# ------------------------------------------------------------------
# Defaults
# ------------------------------------------------------------------


class TestDefaults:
    """Verify documented default values are applied."""

    def test_default_values(self) -> None:
        cfg = PipelineConfig(video_path="video.mp4")
        assert cfg.video_path == "video.mp4"
        assert cfg.extraction_rate is None
        assert cfg.target_fps is None
        assert cfg.plant_weights_path == "yolo11_plants.pt"
        assert cfg.plant_confidence_threshold == 0.25
        assert cfg.bytetrack_config_path is None
        assert cfg.roi_padding == 0.1
        assert cfg.leaf_weights_path == "weights/leaf_best.pt"
        assert cfg.leaf_confidence_threshold == 0.25
        assert cfg.classifier_weights_path == "resnet50_tomato.pt"
        assert cfg.class_names == _DEFAULT_CLASS_NAMES
        # Default is "auto" (resolves CUDA -> MPS -> CPU at runtime) so the
        # same config is portable across macOS / Windows / Linux.
        assert cfg.device == "auto"
        assert cfg.aggregation_window == 30
        assert cfg.aggregation_confidence_threshold == 0.6
        assert cfg.aggregation_majority_ratio == 0.6
        assert cfg.dual_threshold_enabled is True
        assert cfg.top_k_diseases == 3
        assert cfg.srt_path is None
        assert cfg.output_format == "json"
        assert cfg.severity_enabled is False
        assert cfg.output_path == "disease_report.json"
        assert cfg.target_processing_fps == 20.0

    def test_class_names_default_is_independent_copy(self) -> None:
        cfg1 = PipelineConfig(video_path="a.mp4")
        cfg2 = PipelineConfig(video_path="b.mp4")
        cfg1.class_names.append("extra")
        assert "extra" not in cfg2.class_names


# ------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------


class TestValidation:
    """Verify ConfigValidationError is raised for invalid values."""

    def test_aggregation_window_too_low(self) -> None:
        with pytest.raises(ConfigValidationError, match="aggregation_window"):
            PipelineConfig(video_path="v.mp4", aggregation_window=14)

    def test_aggregation_window_too_high(self) -> None:
        with pytest.raises(ConfigValidationError, match="aggregation_window"):
            PipelineConfig(video_path="v.mp4", aggregation_window=61)

    def test_aggregation_window_boundary_low(self) -> None:
        cfg = PipelineConfig(video_path="v.mp4", aggregation_window=15)
        assert cfg.aggregation_window == 15

    def test_aggregation_window_boundary_high(self) -> None:
        cfg = PipelineConfig(video_path="v.mp4", aggregation_window=60)
        assert cfg.aggregation_window == 60

    def test_negative_plant_confidence_threshold(self) -> None:
        with pytest.raises(ConfigValidationError, match="plant_confidence_threshold"):
            PipelineConfig(video_path="v.mp4", plant_confidence_threshold=-0.1)

    def test_negative_leaf_confidence_threshold(self) -> None:
        with pytest.raises(ConfigValidationError, match="leaf_confidence_threshold"):
            PipelineConfig(video_path="v.mp4", leaf_confidence_threshold=-0.01)

    def test_negative_aggregation_confidence_threshold(self) -> None:
        with pytest.raises(ConfigValidationError, match="aggregation_confidence_threshold"):
            PipelineConfig(video_path="v.mp4", aggregation_confidence_threshold=-1.0)

    def test_negative_aggregation_majority_ratio(self) -> None:
        with pytest.raises(ConfigValidationError, match="aggregation_majority_ratio"):
            PipelineConfig(video_path="v.mp4", aggregation_majority_ratio=-0.5)

    def test_empty_class_names(self) -> None:
        with pytest.raises(ConfigValidationError, match="class_names"):
            PipelineConfig(video_path="v.mp4", class_names=[])

    def test_invalid_output_format(self) -> None:
        with pytest.raises(ConfigValidationError, match="output_format"):
            PipelineConfig(video_path="v.mp4", output_format="xml")

    def test_valid_output_format_csv(self) -> None:
        cfg = PipelineConfig(video_path="v.mp4", output_format="csv")
        assert cfg.output_format == "csv"

    def test_zero_confidence_thresholds_are_valid(self) -> None:
        cfg = PipelineConfig(
            video_path="v.mp4",
            plant_confidence_threshold=0.0,
            leaf_confidence_threshold=0.0,
            aggregation_confidence_threshold=0.0,
            aggregation_majority_ratio=0.0,
        )
        assert cfg.plant_confidence_threshold == 0.0


# ------------------------------------------------------------------
# File loading
# ------------------------------------------------------------------


class TestFromFile:
    """Verify loading from YAML and JSON config files."""

    def test_load_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(textwrap.dedent("""\
            video_path: drone.mp4
            aggregation_window: 20
            output_format: csv
        """))
        cfg = PipelineConfig.from_file(cfg_file)
        assert cfg.video_path == "drone.mp4"
        assert cfg.aggregation_window == 20
        assert cfg.output_format == "csv"
        # defaults still applied
        assert cfg.plant_confidence_threshold == 0.25

    def test_load_json(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.json"
        data = {"video_path": "drone.mp4", "aggregation_window": 45}
        cfg_file.write_text(json.dumps(data))
        cfg = PipelineConfig.from_file(cfg_file)
        assert cfg.video_path == "drone.mp4"
        assert cfg.aggregation_window == 45

    def test_load_yml_extension(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yml"
        cfg_file.write_text("video_path: v.mp4\n")
        cfg = PipelineConfig.from_file(cfg_file)
        assert cfg.video_path == "v.mp4"

    def test_load_invalid_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.yaml"
        cfg_file.write_text(": : :\n  - [invalid")
        with pytest.raises(ConfigValidationError, match="config_file"):
            PipelineConfig.from_file(cfg_file)

    def test_load_invalid_json(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "bad.json"
        cfg_file.write_text("{not valid json")
        with pytest.raises(ConfigValidationError, match="config_file"):
            PipelineConfig.from_file(cfg_file)

    def test_load_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigValidationError, match="config_file"):
            PipelineConfig.from_file(tmp_path / "nope.yaml")

    def test_load_unsupported_extension(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.toml"
        cfg_file.write_text("video_path = 'v.mp4'")
        with pytest.raises(ConfigValidationError, match="unsupported"):
            PipelineConfig.from_file(cfg_file)

    def test_load_yaml_with_invalid_values_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("video_path: v.mp4\naggregation_window: 100\n")
        with pytest.raises(ConfigValidationError, match="aggregation_window"):
            PipelineConfig.from_file(cfg_file)

    def test_load_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = PipelineConfig.from_file(cfg_file)
        assert cfg.aggregation_window == 30

    def test_load_yaml_non_mapping_raises(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "list.yaml"
        cfg_file.write_text("- item1\n- item2\n")
        with pytest.raises(ConfigValidationError, match="mapping"):
            PipelineConfig.from_file(cfg_file)
