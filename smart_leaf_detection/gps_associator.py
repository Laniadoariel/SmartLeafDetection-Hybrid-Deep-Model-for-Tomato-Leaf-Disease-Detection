"""GPS association from drone SRT telemetry files.

Parses DJI-style SRT subtitle files recorded alongside drone video to extract
per-frame GPS coordinates, altitude, and camera metadata. Maps subtitle entries
to frame indices (subtitle 1 → frame 0, subtitle 2 → frame 1, etc.).

GPS Accuracy Note
-----------------
The GPS coordinates provided are **approximate frame-level** positions derived
from the drone's onboard GNSS receiver as recorded in the SRT telemetry. They
represent the drone's location at the time each subtitle block was captured,
NOT a precise georeferenced position of individual plants on the ground.
Typical accuracy is ±2–5 m depending on satellite conditions and drone altitude.
For precise plant-level geolocation, post-processing with ground control points
(GCPs) or RTK-corrected GPS would be required.
"""

from __future__ import annotations

import re
from pathlib import Path

from smart_leaf_detection.errors import SRTParseError
from smart_leaf_detection.models import GPSCoordinate

# Regex patterns for extracting GPS data from DJI SRT blocks.
# Matches patterns like: [latitude: 32.123456] [longitude: 34.987654] [altitude: 50.5]
_LAT_RE = re.compile(r"\[latitude\s*:\s*([+-]?\d+(?:\.\d+)?)\]", re.IGNORECASE)
_LON_RE = re.compile(r"\[longitude\s*:\s*([+-]?\d+(?:\.\d+)?)\]", re.IGNORECASE)
_ALT_RE = re.compile(r"\[altitude\s*:\s*([+-]?\d+(?:\.\d+)?)\]", re.IGNORECASE)

# Matches the subtitle index line (e.g. "1", "2", …)
_INDEX_RE = re.compile(r"^\s*(\d+)\s*$")

# Matches the SRT timestamp line (e.g. "00:00:00,000 --> 00:00:01,000")
_TIMESTAMP_RE = re.compile(
    r"^\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[,.]\d{3}\s*$"
)


class GPSAssociator:
    """Parses SRT files and associates GPS coordinates with frame indices.

    If *srt_path* is ``None``, GPS data is unavailable and all lookups return
    ``None``.  If the SRT file is malformed or unreadable, an
    :class:`~smart_leaf_detection.errors.SRTParseError` is raised during
    construction.

    GPS coordinates are **approximate frame-level** positions from the drone's
    GNSS receiver — not precise georeferenced plant locations.  See the module
    docstring for details on expected accuracy.
    """

    def __init__(self, srt_path: str | None = None) -> None:
        """Parse the SRT file on construction.

        Parameters
        ----------
        srt_path:
            Path to a DJI-style SRT file.  ``None`` means GPS is unavailable.
        """
        self._gps_data: dict[int, GPSCoordinate] = {}
        self._available: bool = srt_path is not None

        if srt_path is not None:
            self._parse(srt_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_gps_for_frame(self, frame_index: int) -> GPSCoordinate | None:
        """Return the GPS coordinate for *frame_index*, or ``None``.

        Returns ``None`` when:
        - No SRT file was provided (GPS unavailable).
        - The requested *frame_index* has no corresponding SRT entry.
        """
        if not self._available:
            return None
        return self._gps_data.get(frame_index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse(self, srt_path: str) -> None:
        """Read and parse the SRT file, populating ``self._gps_data``."""
        path = Path(srt_path)

        # --- Read file contents -------------------------------------------
        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raise SRTParseError(srt_path, "File not found")
        except PermissionError:
            raise SRTParseError(srt_path, "Permission denied")
        except OSError as exc:
            raise SRTParseError(srt_path, f"Unable to read file: {exc}")

        if not content.strip():
            raise SRTParseError(srt_path, "File is empty")

        # --- Split into subtitle blocks -----------------------------------
        blocks = self._split_blocks(content)

        if not blocks:
            raise SRTParseError(
                srt_path, "No valid subtitle blocks found in SRT file"
            )

        # --- Extract GPS from each block ----------------------------------
        parsed_any = False
        for block in blocks:
            result = self._parse_block(block)
            if result is not None:
                subtitle_index, coord = result
                # Subtitle indices are 1-based; frame indices are 0-based.
                frame_index = subtitle_index - 1
                self._gps_data[frame_index] = coord
                parsed_any = True

        if not parsed_any:
            raise SRTParseError(
                srt_path,
                "SRT file contains subtitle blocks but no GPS coordinates could be extracted",
            )

    @staticmethod
    def _split_blocks(content: str) -> list[str]:
        """Split raw SRT content into individual subtitle blocks.

        Blocks are separated by one or more blank lines.
        """
        # Normalise line endings and split on blank-line boundaries.
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        raw_blocks = re.split(r"\n\s*\n", content.strip())
        return [b.strip() for b in raw_blocks if b.strip()]

    @staticmethod
    def _parse_block(block: str) -> tuple[int, GPSCoordinate] | None:
        """Extract subtitle index and GPS coordinate from a single block.

        Returns ``None`` if the block does not contain valid GPS data.
        """
        lines = block.strip().splitlines()
        if not lines:
            return None

        # --- Find subtitle index ------------------------------------------
        subtitle_index: int | None = None
        for line in lines:
            m = _INDEX_RE.match(line)
            if m:
                subtitle_index = int(m.group(1))
                break

        if subtitle_index is None or subtitle_index < 1:
            return None

        # --- Combine remaining text for regex search ----------------------
        text = "\n".join(lines)

        lat_match = _LAT_RE.search(text)
        lon_match = _LON_RE.search(text)

        if not lat_match or not lon_match:
            # Block exists but has no GPS — skip silently (some SRT blocks
            # may contain only camera settings without coordinates).
            return None

        latitude = float(lat_match.group(1))
        longitude = float(lon_match.group(1))

        alt_match = _ALT_RE.search(text)
        altitude = float(alt_match.group(1)) if alt_match else None

        return subtitle_index, GPSCoordinate(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
        )
