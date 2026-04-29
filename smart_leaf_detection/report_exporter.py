"""Report exporter for the SmartLeafDetection pipeline."""

from __future__ import annotations

import csv
import json
from typing import Any

from smart_leaf_detection.models import DiseaseRecord, GPSCoordinate


class ReportExporter:
    """Generates the final disease report in JSON or CSV format.

    Exports one DiseaseRecord per diseased PlantID. Healthy plants
    produce zero records. Each record includes flight_id, plant_id,
    GPS coordinates (12-digit precision when available), disease_labels,
    and evidence_metrics. An optional severity field is included when
    severity estimation is enabled.
    """

    def __init__(self, output_format: str = "json", severity_enabled: bool = False) -> None:
        """
        Args:
            output_format: Output format — ``"json"`` or ``"csv"``.
            severity_enabled: When True, the ``severity`` field is
                included in every exported record.
        """
        if output_format not in ("json", "csv"):
            raise ValueError(f"Unsupported output format: {output_format!r}. Use 'json' or 'csv'.")
        self.output_format = output_format
        self.severity_enabled = severity_enabled

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, records: list[DiseaseRecord], output_path: str) -> str:
        """Write *records* to *output_path* and return the path written."""
        if self.output_format == "json":
            self._export_json(records, output_path)
        else:
            self._export_csv(records, output_path)
        return output_path

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_to_dict(self, record: DiseaseRecord) -> dict[str, Any]:
        """Convert a single DiseaseRecord to a plain dict for serialisation."""
        d: dict[str, Any] = {
            "flight_id": record.flight_id,
            "plant_id": record.plant_id,
            "gps": self._gps_to_dict(record.gps),
            "disease_labels": record.disease_labels,
            "evidence_metrics": record.evidence_metrics,
        }
        if self.severity_enabled:
            d["severity"] = record.severity
        return d

    @staticmethod
    def _gps_to_dict(gps: GPSCoordinate | None) -> dict[str, str | None] | None:
        """Serialise a GPSCoordinate with 12-digit precision, or None."""
        if gps is None:
            return None
        result: dict[str, str | None] = {
            "latitude": f"{gps.latitude:.12f}",
            "longitude": f"{gps.longitude:.12f}",
        }
        result["altitude"] = f"{gps.altitude:.12f}" if gps.altitude is not None else None
        return result

    # ------------------------------------------------------------------
    # Format-specific writers
    # ------------------------------------------------------------------

    def _export_json(self, records: list[DiseaseRecord], output_path: str) -> None:
        payload = [self._record_to_dict(r) for r in records]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def _export_csv(self, records: list[DiseaseRecord], output_path: str) -> None:
        fieldnames = [
            "flight_id",
            "plant_id",
            "gps_latitude",
            "gps_longitude",
            "gps_altitude",
            "disease_labels",
            "evidence_metrics",
        ]
        if self.severity_enabled:
            fieldnames.append("severity")

        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for record in records:
                row = self._flatten_record(record)
                writer.writerow(row)

    def _flatten_record(self, record: DiseaseRecord) -> dict[str, Any]:
        """Flatten a DiseaseRecord into a dict suitable for CSV output."""
        gps = record.gps
        row: dict[str, Any] = {
            "flight_id": record.flight_id,
            "plant_id": record.plant_id,
            "gps_latitude": f"{gps.latitude:.12f}" if gps else "",
            "gps_longitude": f"{gps.longitude:.12f}" if gps else "",
            "gps_altitude": f"{gps.altitude:.12f}" if gps and gps.altitude is not None else "",
            "disease_labels": ";".join(record.disease_labels),
            "evidence_metrics": json.dumps(record.evidence_metrics),
        }
        if self.severity_enabled:
            row["severity"] = record.severity if record.severity is not None else ""
        return row
