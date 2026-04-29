"""Error hierarchy for the SmartLeafDetection pipeline."""

from __future__ import annotations


class PipelineError(Exception):
    """Base error for all pipeline failures."""

    pass


class FrameExtractionError(PipelineError):
    """Raised when video file is unreadable or corrupted.

    Attributes:
        file_path: Path to the problematic video file.
        failure_description: Description of what went wrong.
    """

    def __init__(self, file_path: str, failure_description: str) -> None:
        self.file_path = file_path
        self.failure_description = failure_description
        super().__init__(f"Frame extraction failed for '{file_path}': {failure_description}")


class ModelLoadError(PipelineError):
    """Raised when a model weight file (YOLO or ResNet50) is missing or fails to load.

    Attributes:
        weights_path: Path to the problematic weights file.
        failure_description: Description of what went wrong.
    """

    def __init__(self, weights_path: str, failure_description: str) -> None:
        self.weights_path = weights_path
        self.failure_description = failure_description
        super().__init__(f"Model load failed for '{weights_path}': {failure_description}")


class SRTParseError(PipelineError):
    """Raised when an SRT file is malformed or unreadable.

    Attributes:
        file_path: Path to the problematic SRT file.
        failure_description: Description of what went wrong.
    """

    def __init__(self, file_path: str, failure_description: str) -> None:
        self.file_path = file_path
        self.failure_description = failure_description
        super().__init__(f"SRT parse failed for '{file_path}': {failure_description}")


class ConfigValidationError(PipelineError):
    """Raised when a configuration parameter is invalid.

    Attributes:
        parameter_name: Name of the invalid parameter.
        constraint_description: Description of the violated constraint.
    """

    def __init__(self, parameter_name: str, constraint_description: str) -> None:
        self.parameter_name = parameter_name
        self.constraint_description = constraint_description
        super().__init__(
            f"Invalid config parameter '{parameter_name}': {constraint_description}"
        )
