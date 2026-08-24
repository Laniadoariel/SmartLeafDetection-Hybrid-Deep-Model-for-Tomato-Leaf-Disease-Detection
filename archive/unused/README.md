# archive/unused

Temporary holding place for files that appear to be **no longer used by the
production application**, kept here (not deleted) pending review.

Nothing in this folder is imported or opened by the web application
(`webapp/`) or by the training pipelines. Files were verified as unreferenced
before being moved here.

| File | Reason it was moved here |
|------|--------------------------|
| `demo.html` | Legacy static demo page that documents the **old** plant-detection → plant-tracking pipeline (ByteTrack), which the production web app does not run. Superseded by the current leaf-centric BoT-SORT pipeline. Not imported by any code. |
| `frame_scores.csv` | Stale generated output. `score_frames.py` regenerates this file in the current working directory on each run, so the checked-in copy is not needed. |
| `video_disease_report.json` | Stale generated output from an old CLI run. Not referenced by the web app; report paths in code default to fresh output files. |

## Deletion policy

Do **not** permanently delete these files without explicit approval. If, after
review, they are confirmed unnecessary, they can be removed in a follow-up
commit. Until then they are preserved here so nothing is lost.
