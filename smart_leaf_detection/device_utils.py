"""Cross-platform compute-device resolution.

The project runs on macOS (Apple Silicon -> Metal/``mps``), Windows/Linux with
an NVIDIA GPU (``cuda``), or any machine on CPU. Hard-coding a single backend
(e.g. ``cuda`` or ``mps``) breaks the code on the other platforms, so all device
selection goes through :func:`resolve_torch_device`.

Priority for ``"auto"`` is: CUDA (fastest, NVIDIA) -> MPS (Apple Silicon) -> CPU.
An explicit request (``"cuda"``/``"mps"``) is honoured when that backend is
available and silently falls back to CPU when it is not, so the same command
works unchanged on Mac and Windows.
"""

from __future__ import annotations


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _mps_available() -> bool:
    try:
        import torch

        backend = getattr(torch.backends, "mps", None)
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def resolve_torch_device(prefer: str | None = "auto") -> str:
    """Return a valid ``torch`` device string for the current machine.

    Args:
        prefer: ``"auto"`` (default) picks the best available backend.
            ``"cuda"``/``"mps"`` are honoured if available, else fall back to
            CPU. ``"cpu"`` or an explicit index like ``"0"`` are returned as-is.

    Returns:
        One of ``"cuda"``, ``"mps"``, ``"cpu"``, or the explicit value passed in.
    """
    choice = (prefer or "auto").strip().lower()

    if choice in ("", "auto"):
        if _cuda_available():
            return "cuda"
        if _mps_available():
            return "mps"
        return "cpu"

    if choice == "cuda" and not _cuda_available():
        return "cpu"
    if choice == "mps" and not _mps_available():
        return "cpu"

    return choice


def resolve_ultralytics_device(prefer: str | None = None) -> str | None:
    """Resolve a device for Ultralytics ``train``/``val``/``predict`` calls.

    Ultralytics auto-selects the best backend when ``device=None``; we preserve
    that for ``None``/``"auto"`` so the framework handles platform differences.
    An explicit value is normalised through :func:`resolve_torch_device` so a
    request for an unavailable backend degrades to CPU instead of crashing.
    """
    if prefer is None or str(prefer).strip().lower() in ("", "auto"):
        return None
    return resolve_torch_device(prefer)
