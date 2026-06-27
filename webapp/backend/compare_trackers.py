#!/usr/bin/env python3
"""Read-only A/B comparison of leaf-tracking quality between two flights.

Compares tracking stability between a ByteTrack run and a BoT-SORT run so we
can decide, with evidence, whether BoT-SORT should become the default tracker.

IMPORTANT — experimental protocol
----------------------------------
Production drops leaves seen in fewer than MIN_TRACK_LEN (=2) frames BEFORE
they are written to the DB. To see the TRUE tracking quality (including the
one-view "singletons" that fragmentation produces), run BOTH comparison
analyses with MIN_TRACK_LEN=1 so nothing is hidden:

    # baseline
    FRAME_STRIDE_SEC=0.3 LEAF_TRACKER=bytetrack MIN_TRACK_LEN=1 \\
        uvicorn app.main:app --port 8000        # upload the video, let it finish

    # candidate
    FRAME_STRIDE_SEC=0.3 LEAF_TRACKER=botsort  MIN_TRACK_LEN=1 \\
        uvicorn app.main:app --port 8000        # upload the SAME video again

Then compare (most-recent flight = botsort, the one before = bytetrack):

    python compare_trackers.py

Or pass explicit flight IDs:

    python compare_trackers.py --bytetrack-flight <ID> --botsort-flight <ID>

This script only runs SELECT queries; it never modifies the database.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import sqlite3
import statistics
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DB = BACKEND_DIR / "smartleaf.db"
REPORTS_DIR = BACKEND_DIR / "reports"

# Production keeps only leaves observed in >= this many frames (mirror worker default).
PRODUCTION_MIN_TRACK_LEN = 2


def _connect(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    # Read-only connection via URI so we can never accidentally write.
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _two_most_recent_flights(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT id FROM flights WHERE status = 'completed' "
        "ORDER BY created_at DESC LIMIT 2"
    ).fetchall()
    return [r["id"] for r in rows]


def _flight_meta(conn: sqlite3.Connection, flight_id: str) -> dict | None:
    row = conn.execute(
        "SELECT id, video_filename, status, created_at, "
        "total_video_frames, relevant_frames, total_detections, total_plants "
        "FROM flights WHERE id = ?",
        (flight_id,),
    ).fetchone()
    return dict(row) if row else None


def _views_distribution(conn: sqlite3.Connection, flight_id: str) -> list[int]:
    rows = conn.execute(
        "SELECT views_total FROM plant_results WHERE flight_id = ?",
        (flight_id,),
    ).fetchall()
    return [int(r["views_total"]) for r in rows]


def _metrics(meta: dict, views: list[int]) -> dict:
    tracked_leaves = len(views)
    one_view = sum(1 for v in views if v == 1)
    multi_view = sum(1 for v in views if v >= 2)
    final_cards = sum(1 for v in views if v >= PRODUCTION_MIN_TRACK_LEN)
    # Total observations that became part of tracked leaves. When the run used
    # MIN_TRACK_LEN=1 this equals every detection that the tracker grouped.
    detections_in_leaves = sum(views)
    return {
        "flight_id": meta["id"],
        "video": meta.get("video_filename", ""),
        "raw_detections_db": meta.get("total_detections", 0),
        "detections_in_leaves": detections_in_leaves,
        "relevant_frames": meta.get("relevant_frames", 0),
        "tracked_leaves": tracked_leaves,
        "stable_leaves_2plus": multi_view,
        "leaves_1_view": one_view,
        "avg_views": round(statistics.mean(views), 2) if views else 0.0,
        "median_views": round(statistics.median(views), 1) if views else 0.0,
        "max_views": max(views) if views else 0,
        "final_result_cards": final_cards,
    }


_ROWS = [
    ("Raw detections (flights.total_detections)", "raw_detections_db"),
    ("Detections grouped into leaves (sum views)", "detections_in_leaves"),
    ("Relevant frames (>=1 leaf)", "relevant_frames"),
    ("Tracked leaves (all)", "tracked_leaves"),
    ("Stable tracked leaves (>=2 views)", "stable_leaves_2plus"),
    ("Leaves with only 1 view (singletons)", "leaves_1_view"),
    ("Average views per leaf", "avg_views"),
    ("Median views per leaf", "median_views"),
    ("Max views on a single leaf", "max_views"),
    (f"Final result cards (>= {PRODUCTION_MIN_TRACK_LEN} views)", "final_result_cards"),
]


def _print_table(bt: dict, bs: dict) -> None:
    label_w = 44
    col_w = 16
    print()
    print("Leaf tracking comparison  (FRAME_STRIDE_SEC=0.3)")
    print("=" * (label_w + col_w * 2 + 4))
    print(f"{'Metric':<{label_w}}{'ByteTrack':>{col_w}}{'BoT-SORT':>{col_w}}")
    print("-" * (label_w + col_w * 2 + 4))
    for label, key in _ROWS:
        bt_v, bs_v = bt[key], bs[key]
        print(f"{label:<{label_w}}{str(bt_v):>{col_w}}{str(bs_v):>{col_w}}")
    print("-" * (label_w + col_w * 2 + 4))
    print(f"ByteTrack flight: {bt['flight_id']}  ({bt['video']})")
    print(f"BoT-SORT  flight: {bs['flight_id']}  ({bs['video']})")
    print()


def _verdict(bt: dict, bs: dict) -> str:
    better_stable = bs["stable_leaves_2plus"] > bt["stable_leaves_2plus"]
    better_avg = bs["avg_views"] > bt["avg_views"]
    if better_stable and better_avg:
        return (
            "VERDICT: BoT-SORT improves BOTH stable tracked leaves and average "
            "views per leaf.\n"
            "  -> Recommend making it the default: set _DEFAULT_TRACKER = \"botsort\"\n"
            "     in webapp/backend/app/worker.py (or run with LEAF_TRACKER=botsort)."
        )
    if better_stable or better_avg:
        return (
            "VERDICT: BoT-SORT improves only ONE of the two key metrics "
            f"(stable_leaves better={better_stable}, avg_views better={better_avg}).\n"
            "  -> Mixed result. Inspect the table before changing the default; "
            "consider re-running to rule out variance."
        )
    return (
        "VERDICT: BoT-SORT does NOT improve stable tracked leaves or average "
        "views per leaf.\n"
        "  -> Keep ByteTrack as the default (_DEFAULT_TRACKER stays \"bytetrack\")."
    )


def _write_csv(bt: dict, bs: dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = REPORTS_DIR / f"tracker_comparison_{ts}.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "bytetrack", "botsort"])
        for label, key in _ROWS:
            w.writerow([label, bt[key], bs[key]])
        w.writerow(["bytetrack_flight_id", bt["flight_id"], ""])
        w.writerow(["botsort_flight_id", "", bs["flight_id"]])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to smartleaf.db")
    ap.add_argument("--bytetrack-flight", help="Flight ID of the ByteTrack run")
    ap.add_argument("--botsort-flight", help="Flight ID of the BoT-SORT run")
    ap.add_argument("--no-csv", action="store_true", help="Do not write a CSV report")
    args = ap.parse_args()

    conn = _connect(args.db)
    try:
        bt_id, bs_id = args.bytetrack_flight, args.botsort_flight
        if not (bt_id and bs_id):
            recent = _two_most_recent_flights(conn)
            if len(recent) < 2:
                raise SystemExit(
                    "Need two completed flights to compare. Run the baseline and "
                    "candidate analyses first, or pass --bytetrack-flight/--botsort-flight."
                )
            # Convention: most recent run is the BoT-SORT candidate, the one
            # before it is the ByteTrack baseline.
            bs_id = bs_id or recent[0]
            bt_id = bt_id or recent[1]
            print("No flight IDs given; using the two most recent completed flights:")
            print(f"  ByteTrack (older):   {bt_id}")
            print(f"  BoT-SORT  (newest):  {bs_id}")
            print("  (override with --bytetrack-flight / --botsort-flight if wrong)")

        bt_meta, bs_meta = _flight_meta(conn, bt_id), _flight_meta(conn, bs_id)
        if not bt_meta:
            raise SystemExit(f"ByteTrack flight not found: {bt_id}")
        if not bs_meta:
            raise SystemExit(f"BoT-SORT flight not found: {bs_id}")

        bt = _metrics(bt_meta, _views_distribution(conn, bt_id))
        bs = _metrics(bs_meta, _views_distribution(conn, bs_id))
    finally:
        conn.close()

    _print_table(bt, bs)
    print(_verdict(bt, bs))

    if not args.no_csv:
        out = _write_csv(bt, bs)
        print(f"\nSaved comparison CSV: {out}")


if __name__ == "__main__":
    main()
