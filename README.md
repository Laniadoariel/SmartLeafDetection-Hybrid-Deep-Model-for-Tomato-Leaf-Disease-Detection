# 🌿 SmartLeafDetection

**Hybrid Deep Learning System for Drone-Based Tomato Leaf Disease Detection**

A complete end-to-end system that processes drone video of tomato fields to detect and classify leaf diseases using **YOLOv11** (leaf localization + tracking) and a **dedicated ResNet50 image classifier** (selected production model) for disease prediction.

## Features

- **Video Processing**: Upload drone video → frame extraction → disease analysis
- **Leaf Detection**: YOLOv11 leaf detector (localization + **BoT-SORT** tracking with Global Motion Compensation for the moving drone camera; ByteTrack available as a fallback via `LEAF_TRACKER`)
- **Disease Classification**: dedicated **ResNet50** image classifier (selected production model) trained on leaf crops; pluggable backend (`classifier` or legacy `yolo`); EfficientNetV2-S / MobileNetV3 also benchmarked and kept for experimentation
- **Full Pipeline**: Leaf detection → tracking → crop → normalization → disease classifier → temporal aggregation per leaf → final prediction
- **Web Application**: React + FastAPI dashboard with upload, investigation, and results tabs
- **GPS Support**: Associates results with drone SRT telemetry data

## Prerequisites

The project runs on **macOS, Windows, and Linux**. Install these once:

| Tool | Version | Notes |
|------|---------|-------|
| **Python** | 3.10 – 3.12 | 3.11 recommended. `python --version` |
| **Node.js + npm** | 18+ | Only needed for the web frontend. `node --version` |
| **Git** | any recent | to clone the repo |
| **PyTorch** | 2.2+ | installed via `requirements.txt`; GPU builds need extra steps (below) |

**Compute backend is auto-detected** — no code changes between machines. At
runtime the code picks **CUDA** (NVIDIA GPU on Windows/Linux) → **MPS** (Apple
Silicon) → **CPU**, in that order
(`smart_leaf_detection/device_utils.py`). CPU works everywhere but is slower.

> **NVIDIA GPU (CUDA) users:** plain `pip install torch` gives a CPU build on
> Windows/Linux. For a CUDA build, install torch from the official selector at
> <https://pytorch.org/get-started/locally/> — e.g. for CUDA 12.1:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.
> On macOS the default wheel already includes Apple-Silicon (MPS) support.

## Quick Start

Helper scripts under `scripts/` wrap the per-OS differences (virtual-env
activation, `PYTHONPATH` separators). Use them, or run the raw commands below.

### 1. Setup (creates venv + installs everything)

**macOS / Linux**
```bash
bash scripts/setup.sh
source venv/bin/activate
```

**Windows (PowerShell)**
```powershell
.\scripts\setup.ps1
.\venv\Scripts\Activate.ps1
```

<details>
<summary>Raw commands (any OS, without the helper script)</summary>

```bash
# macOS / Linux
python3 -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r webapp/backend/requirements.txt
```
```powershell
# Windows PowerShell
python -m venv venv ; .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -r webapp\backend\requirements.txt
```
</details>

### 2. Test on sample images
```bash
# Download/train YOLO weights first (see Model Weights section below)
python run_on_frames.py \
  --frames 1_original.jpeg 2_original.jpeg 3_original.jpeg \
  --disease-weights weights/best.pt \
  --confidence 0.3
```
(On Windows use `^` line-continuations or put it on one line.)

### 3. Run the web application

**macOS / Linux** — two terminals:
```bash
# Terminal 1 — backend
bash scripts/start_backend.sh 8000
# Terminal 2 — frontend
bash scripts/start_frontend.sh
```

**Windows (PowerShell)** — two terminals:
```powershell
# Terminal 1 — backend
.\scripts\start_backend.ps1 -Port 8000
# Terminal 2 — frontend
.\scripts\start_frontend.ps1
```

<details>
<summary>Raw commands (without the helper scripts)</summary>

```bash
# macOS / Linux — backend (PYTHONPATH uses ':' separators)
source venv/bin/activate
cd webapp/backend
PYTHONPATH=".:../.." LEAF_CONF=0.3 python -m uvicorn app.main:app --port 8000 --reload
```
```powershell
# Windows PowerShell — backend (PYTHONPATH uses ';' separators)
.\venv\Scripts\Activate.ps1
cd webapp\backend
$env:PYTHONPATH = "..;..\.." ; $env:LEAF_CONF = "0.3"
python -m uvicorn app.main:app --port 8000 --reload
```
```bash
# Frontend (any OS)
cd webapp/frontend
npm install
npm run dev
```
</details>

Open http://localhost:3000

## Project Structure

```
SmartLeafDetection/
├── smart_leaf_detection/       # Core ML pipeline
│   ├── pipeline.py             # Main orchestrator
│   ├── frame_extractor.py      # Video → frames
│   ├── plant_detector.py       # YOLOv11 plant detection
│   ├── plant_tracker.py        # ByteTrack plant tracking
│   ├── leaf_detector.py        # YOLOv11 leaf detection
│   ├── leaf_tracker.py         # ByteTrack leaf tracking
│   ├── roi_cropper.py          # ROI extraction with padding
│   ├── leaf_normalizer.py      # 224×224 ImageNet normalization
│   ├── disease_classifier.py   # ResNet50 classifier (legacy CLI helper)
│   ├── leaf_disease_classifier.py  # dedicated image classifier (webapp inference)
│   ├── temporal_aggregator.py  # Sliding window aggregation
│   ├── plant_status_engine.py  # Plant health inference
│   ├── gps_associator.py       # SRT telemetry parsing
│   ├── report_exporter.py      # JSON/CSV export
│   ├── config.py               # Pipeline configuration
│   ├── models.py               # Data models
│   └── errors.py               # Error hierarchy
├── training/                   # Training scripts
│   ├── prepare_cvat_and_train.py  # Main training (merges all datasets)
│   ├── train_stable.py         # Crash-proof training (val=False)
│   ├── train_resnet50.py       # ResNet50 fine-tuning (legacy)
│   ├── train_yolo_leaves.py    # Leaf detection training
│   ├── disease_classification/ # dedicated disease classifier pipeline
│   │   ├── class_mapping.py            # canonical taxonomy + label mapping
│   │   ├── build_classification_dataset.py  # crop YOLO boxes → ImageFolder
│   │   ├── model_factory.py            # torchvision arch factory
│   │   ├── train_classifier.py         # transfer-learning trainer
│   │   └── evaluate_classifier.py      # test metrics + YOLO comparison
│   └── extract_annotations.py  # CVAT annotation extraction
├── tests/                      # Test suite
│   ├── test_end_to_end.py      # Full pipeline integration tests
│   ├── test_pipeline_config.py # Configuration tests
│   └── test_leaf_detector.py   # Detector tests
├── webapp/                     # Web application
│   ├── backend/                # FastAPI + SQLAlchemy
│   └── frontend/               # React + TypeScript
├── run_on_frames.py            # CLI tool for static images
├── requirements.txt            # Python dependencies
└── demo.html                   # Static demo page
```

## Pipeline Flow

```
Drone Video → Frame Extraction → Plant Detection (YOLOv11)
→ Plant Tracking (ByteTrack) → ROI Cropping
→ Leaf Detection (YOLOv11) → Leaf Tracking (ByteTrack)
→ Normalization (224×224) → Disease Classification (dedicated ResNet50 classifier)
→ Temporal Aggregation → Plant Status Inference
→ GPS Association → Report Export
```

## Model Weights

The trained model weights are too large for GitHub. Download or train them:

### Option A: Train from scratch
```bash
# Prepare datasets (Roboflow + CVAT + PlantDoc + Tomato-6K)
python training/prepare_cvat_and_train.py

# Or use crash-proof training
python training/train_stable.py
```

### Option B: Place weights manually
Place your trained `best.pt` or `last.pt` in:
```
weights/best.pt
```

## Datasets Used

1. **Roboflow Tomato Diseases** (645 images, 7 classes)
2. **CVAT Late Blight** (101 images, polyline annotations)
3. **PlantDoc** (2,500+ field images, multi-leaf scenes)
4. **Tomato-6K** (5,139 train images, 5 classes)

## Running Tests

```bash
# activate the venv first (macOS/Linux: source venv/bin/activate
#                          Windows:     .\venv\Scripts\Activate.ps1)
python -m pytest tests/ -v
```

## Cross-platform notes (macOS / Windows / Linux)

This project is designed to run identically on all three. Key points:

- **Compute device is auto-selected** (CUDA → MPS → CPU) by
  `smart_leaf_detection/device_utils.py`; no per-OS edits. Override anywhere a
  `--device` flag exists, or via `PipelineConfig(device=...)` using
  `"auto"`/`"cuda"`/`"mps"`/`"cpu"`.
- **Use the `scripts/` helpers** (`.sh` for macOS/Linux, `.ps1` for Windows) so
  you never have to remember venv-activation paths or `PYTHONPATH` separators
  (`:` on Unix vs `;` on Windows).
- **Stored file paths are OS-agnostic** — the web backend writes forward-slash
  relative paths so served URLs work the same on Windows and macOS.
- **Frontend** (React + Vite via npm) is already cross-platform.

### Troubleshooting

| Symptom | Platform | Fix |
|---------|----------|-----|
| `running scripts is disabled on this system` | Windows | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then re-run the `.ps1`. |
| Training/inference is very slow | Windows/Linux | You're on the CPU torch build. Install the CUDA build from <https://pytorch.org/get-started/locally/>. |
| `torch.cuda.is_available()` is `False` with an NVIDIA GPU | Windows/Linux | Installed the CPU wheel; reinstall torch with the matching `--index-url cuXXX`. |
| `'cls_pw' is not a valid YOLO argument` on `--resume` | any | Old checkpoint + newer Ultralytics — run `fix_resume_checkpoint.py` (see bug #11). |
| `ModuleNotFoundError: smart_leaf_detection` when starting backend | any | `PYTHONPATH` not set — use the `scripts/start_backend.*` helper. |

## Technology Stack

- **Detection**: YOLOv11 (Ultralytics) — leaf localization + tracking
- **Classification**: dedicated **ResNet50** image classifier (production); EfficientNet-B0 / EfficientNetV2-S / ConvNeXt-Tiny / MobileNetV3 selectable via `--arch` and benchmarked (PyTorch/torchvision)
- **Tracking**: ByteTrack-style IOU tracker
- **Backend**: FastAPI + SQLAlchemy + SQLite
- **Frontend**: React + TypeScript + Vite
- **CV**: OpenCV

## Authors

Noy Ariel — Technion Final Project

---

# 🧪 Disease Classification — Dedicated Image Classifier

This section documents the **dedicated disease-classification pipeline** that
replaces the old "run the YOLO disease detector on each leaf crop" approach. The
YOLO leaf detector now does **only localization + tracking**; the disease
prediction comes from a purpose-trained image classifier.

New pipeline:

```
Video → Frame Extraction → YOLO Leaf Detection → Leaf Tracking (ByteTrack)
→ Leaf Crop Extraction → Normalization (224×224, ImageNet)
→ Disease Image Classifier → Temporal Aggregation (per tracked leaf)
→ Final Disease Prediction
```

## Why a dedicated classifier (not the YOLO detector)

The YOLO disease model was trained to find disease-bbox patterns in whole
scenes. Run on a tightly-cropped single leaf it often finds *no* box and the
leaf silently defaulted to "healthy", biasing results. A classifier trained
specifically on single-leaf crops always returns a class with calibrated
probabilities, which is exactly what the per-leaf temporal aggregation needs.

## Selected production model: ResNet50

A controlled benchmark (identical dataset, split, seed, augmentation, optimizer,
schedule, and early-stopping on validation Macro-F1) compared three
architectures on the held-out **test** split:

| Model | Accuracy | Macro-F1 | Avg inference | Params | Size |
|-------|----------|----------|---------------|--------|------|
| EfficientNetV2-S | 0.953 | 0.861 | 31.8 ms | 20.2 M | 77.9 MB |
| **ResNet50 (selected)** | 0.942 | 0.849 | 14.0 ms | 23.5 M | 90.1 MB |
| MobileNetV3-Large | 0.930 | 0.826 | 13.3 ms | 4.2 M | 16.3 MB |

**EfficientNetV2-S achieved slightly higher benchmark results** (≈1% higher
accuracy and ~0.012 higher Macro-F1). **ResNet50 was nonetheless selected as the
production model** because:

- The performance gap is small and within run-to-run variance for a dataset of
  this size, so it does not justify changing the project's architecture.
- This is an academic final-year engineering project whose research,
  documentation, thesis, presentation, and experimental evaluation are all built
  around ResNet50. **Consistency across the implementation, the documentation,
  and the written/defended work is more valuable here than the marginal metric
  gain** offered by EfficientNetV2-S.
- ResNet50 is a well-understood, widely-cited baseline — appropriate and
  defensible for a thesis.

EfficientNetV2-S and MobileNetV3-Large checkpoints are **kept for future
experimentation** (`weights/benchmark/`) but are not used by default. The
training script still supports `--arch {efficientnet_v2_s, efficientnet_b0,
resnet50, convnext_tiny, mobilenet_v3_large}`; the **default arch is now
`resnet50`** so the production model is reproducible with a plain training run.

> Cross-architecture note: all three models showed the **same** weak spot —
> `target_spot` (recall ≈ 0.27) confused with `septoria_leaf_spot` /
> `powdery_mildew`. Because the failure is identical across architectures, the
> system is **data-limited, not architecture-limited**; improving those classes'
> data will help more than swapping backbones (see the benchmark report's
> failure analysis).

## Dataset generation (from existing YOLO labels)

Every disease dataset in the project is YOLO **detection** format. The builder
crops each labelled box into an `ImageFolder` classification dataset, unifying
class names and preventing train/val/test leakage.

```bash
python training/disease_classification/build_classification_dataset.py
# options: --datasets tomato_diseases merged_diseases plantdoc tomato_6k
#          --padding 0.08 --min-size 24 --min-samples-per-class 25
#          --split-mode existing   (default: honor each dataset's train/valid/test folders)
#          --split-mode resplit --val-frac 0.1 --test-frac 0.1   (one fixed 80/10/10 source-level split)
```

Either split mode guarantees the **test set never shares an image with
training** (assignment is at the source-image level). Use `existing` to reuse
the datasets' own splits; use `resplit` if you want a single clean 80/10/10
split (handy because `merged_diseases` ships no test folder and `PlantDoc` ships
no validation folder).

Output: `datasets/leaf_clf/{train,validation,test}/<class>/*.jpg` plus
`training/disease_classification/reports/dataset_report.json` (total/per-class
counts, imbalance ratio, discarded crops + reasons, avg crop size) and
`class_distribution.png`.

**Leakage prevention:** each *source image id* (Roboflow `.rf.<hash>` suffix
stripped) is assigned to exactly one split using priority **test > validation >
train**, scanned across all datasets before any cropping. So augmented copies
and cross-dataset duplicates of the same photo never straddle splits.

### Canonical class mapping (Step 2)

All source labels are unified into one snake_case taxonomy in
`training/disease_classification/class_mapping.py` (single source of truth):

| Canonical class | Mapped from (examples) |
|---|---|
| `healthy` | `Healthy`, `healthy`, PlantDoc `Tomato leaf` |
| `bacterial_spot` | `Bacterial Spot`, `bacterial_spot`, `Tomato leaf bacterial spot` |
| `early_blight` | `Early_Blight`, `early_blight`, `Tomato Early blight leaf` |
| `late_blight` | `Late_blight`, `late_blight`, `Tomato leaf late blight` |
| `leaf_mold` | `Leaf Mold`, `Tomato mold leaf` |
| `septoria_leaf_spot` | `Tomato Septoria leaf spot` |
| `target_spot` | `Target_Spot` |
| `spider_mites` | `Tomato two spotted spider mites leaf` |
| `mosaic_virus` | `Tomato leaf mosaic virus` |
| `yellow_leaf_curl_virus` | `Tomato leaf yellow virus` |
| `powdery_mildew` | `powdery_mildew` (tomato_6k) |
| `black_spot` | `black spot` |

Non-tomato PlantDoc classes (apple, corn, grape, etc.) are explicitly ignored.
Classes with fewer than `--min-samples-per-class` crops are dropped and recorded
in the report. The final kept classes are written to `weights/classes.json`.

## Training (Step 4)

```bash
# macOS/Linux: source venv/bin/activate   |  Windows: .\venv\Scripts\Activate.ps1
python training/disease_classification/train_classifier.py \
  --arch resnet50 --device mps   # resnet50 is the default/production arch; or: 0 / cpu / (omit)
```

Optimized for **validation Macro-F1** (the saved best checkpoint is the epoch
with the highest val Macro-F1, not accuracy — per-class quality matters more
than majority-class accuracy). Implements transfer learning, a frozen-backbone
warm-up → **progressive unfreezing**, **class-weighted** loss + label smoothing
for imbalance, cosine LR, and **early stopping on val Macro-F1** (training runs
until early stopping triggers naturally; `--epochs` is only an upper bound).

**Augmentation policy.** Disease symptoms are *local* patterns (spots, lesions,
discoloration), so the default augmentation is **strong but realistic**: random
rotation, H/V flips (valid for top-down leaves), brightness/contrast + colour
jitter, mild autocontrast, small scale variation, mild Gaussian blur, and mild
Gaussian noise. **MixUp/CutMix are OFF by default** — mixing whole leaves can
fabricate unrealistic symptom layouts. Enable them only to test whether they
actually raise val Macro-F1:

```bash
python training/disease_classification/train_classifier.py --mixup 0.2   # or --cutmix 0.2
```

**Reproducibility.** Every run writes `experiment_metadata.json` capturing the
full config, seeds, exact augmentation settings, optimizer/scheduler settings,
architecture/hyper-parameters, and per-class/per-split dataset statistics — so
any result can be reproduced exactly.

Artifacts:
- `weights/leaf_classifier.pt` — checkpoint (arch + classes + normalization + config + best val Macro-F1)
- `weights/classes.json`
- `training/disease_classification/reports/{training_metrics.json, experiment_metadata.json, training_curves.png, confusion_matrix.png}`

The benchmark orchestrator additionally writes a **Model Card** (`MODEL_CARD.md`)
for the winning model (architecture, dataset, train/val/test image counts,
training config, best val Macro-F1, test accuracy + Macro-F1, known limitations,
recommended use cases) and a **failure-case analysis** (per-class most-confused
diseases, likely reasons, and recommended improvements) in `BENCHMARK_REPORT.md`.

## Evaluation + comparison with YOLO (Step 5)

Single-model evaluation:

```bash
python training/disease_classification/evaluate_classifier.py \
  --split test --compare-yolo weights/best.pt --device mps
```

Writes `reports/evaluation_metrics.json` (accuracy, macro precision/recall/F1,
per-class metrics) + `confusion_matrix_test.png`. With `--compare-yolo` it also
runs the legacy YOLO disease model on the same test crops (mapping its labels to
canonical) and reports both accuracies side by side.

**Honest comparison.** The classifier is expected to beat the repurposed YOLO
detector on cropped leaves mainly by eliminating the "no-box → healthy" failure
and by producing real per-class probabilities. The key remaining weakness is the
**domain gap**: most training crops are studio/field single-leaf images, while
the webapp feeds small drone-video crops; accuracy on real drone leaves is
capped until some drone crops are added/fine-tuned. Rare classes
(`target_spot`, `spider_mites`, `powdery_mildew`) will have weaker per-class
recall due to fewer samples.

## Architecture benchmark study (fair, reproducible)

To select the backbone on **evidence** rather than assumption, an orchestrator
trains and evaluates several architectures under an identical protocol (same
dataset, split, seed, image size, augmentation, optimizer, schedule,
early-stopping, class weighting — only `--arch` changes) and aggregates the
results.

```bash
# trains each model in an isolated process; incremental — re-running skips
# finished models. Add more archs as desired (mandatory three shown).
python training/disease_classification/run_benchmark.py \
  --archs efficientnet_v2_s resnet50 mobilenet_v3_large convnext_tiny efficientnet_b0 \
  --epochs 40 --device mps --compare-yolo weights/best.pt
```

Outputs (`training/disease_classification/reports/benchmark/`):
- `weights/benchmark/<arch>.pt` — one checkpoint per model (kept for future use)
- per-model `reports/benchmark/<arch>/` — training curves, confusion matrix, per-class F1, eval JSON
- `comparison.json` / `comparison.csv` — the full table (accuracy, precision,
  recall, F1, avg inference ms, params, model size, composite selection score)
- `cmp_accuracy_f1.png`, `cmp_inference.png`, `cmp_size_params.png`
- **`BENCHMARK_REPORT.md`** — an academic-style report (objective, dataset,
  methodology, training config, results, tables, graphs, discussion, selection
  rationale, YOLO comparison, limitations, future work) with the real numbers
  filled in.

**Selection is not accuracy-only.** The composite score is
`0.60·macro-F1 + 0.25·speed + 0.15·footprint` (speed/footprint normalized so
smaller is better), reflecting that the webapp classifies many crops per video
on possibly-CPU hardware. The report recommends the top-composite model but
tells you to prefer a near-equal model with better rare-class F1 if the
per-class charts justify it.

Promote the chosen model to the deployed path (only after reviewing the report):

```bash
python training/disease_classification/run_benchmark.py --promote auto   # copies best -> weights/leaf_classifier.pt
# or copy a specific arch manually:
cp weights/benchmark/resnet50.pt weights/leaf_classifier.pt
```



## WebApp integration & backend switch (Steps 6–7)

The FastAPI worker (`webapp/backend/app/worker.py`) selects the disease backend
via the `DISEASE_BACKEND` environment variable:

| `DISEASE_BACKEND` | Behaviour |
|---|---|
| `auto` (default) | use the classifier if `weights/leaf_classifier.pt` exists, else YOLO |
| `classifier` | force the dedicated image classifier |
| `yolo` | force the legacy YOLO-on-crop detector |

```bash
# Force the classifier explicitly:
DISEASE_BACKEND=classifier bash scripts/start_backend.sh 8000          # macOS/Linux
$env:DISEASE_BACKEND="classifier"; .\scripts\start_backend.ps1         # Windows
```

The leaf detector (`weights/leaf_best.pt`) is always used for
localization/tracking. Disease prediction goes through a single dispatcher
(`_classify_crop`) that reuses the shared `LeafDiseaseClassifier`
(`smart_leaf_detection/leaf_disease_classifier.py`) — no duplicated logic. If
classifier weights are absent the worker transparently falls back to YOLO, so
nothing breaks before training is complete.

## How to retrain / extend

1. (Optional) adjust the taxonomy/mapping in `class_mapping.py`.
2. Rebuild: `python training/disease_classification/build_classification_dataset.py`.
3. Retrain: `python training/disease_classification/train_classifier.py --arch <arch>`.
4. Evaluate: `python training/disease_classification/evaluate_classifier.py --compare-yolo weights/best.pt`.
5. The webapp picks up `weights/leaf_classifier.pt` automatically (`DISEASE_BACKEND=auto`).

To add a new architecture, extend `model_factory.build_model` only — the
trainer, evaluator, and webapp inference all build from it.

## Final Production Architecture

The deployed SmartLeafDetection pipeline (for the project report):

```
                 Drone video (.mp4/.mov/.avi/.mkv)
                            │
                 Frame extraction (OpenCV, time-based sampling
                  + near-duplicate skipping)
                            │
                 Best-frame selection (sharpest, most leaf-dense
                  contiguous window)
                            │
                 Leaf detection  ──  YOLOv11  (weights/leaf_best.pt)
                  (localization only)
                            │
                 Leaf tracking   ──  ByteTrack (stable per-leaf IDs)
                            │
                 Leaf crop extraction (per tracked leaf)
                            │
                 Normalization (224×224, BGR→RGB, ImageNet stats)
                            │
                 Disease classification ── ResNet50
                  (weights/leaf_classifier.pt, 11 canonical classes)
                            │
                 Temporal aggregation (confidence-weighted majority
                  vote over each leaf's track)
                            │
                 Final per-leaf disease prediction  →  DB  →  Web UI
```

**Production model:** ResNet50 (`weights/leaf_classifier.pt`), selected over the
marginally-stronger EfficientNetV2-S to keep the implementation, documentation,
thesis, and evaluation consistent (see "Selected production model").

**Canonical classes (11):** healthy, bacterial_spot, early_blight, late_blight,
leaf_mold, septoria_leaf_spot, target_spot, mosaic_virus, yellow_leaf_curl_virus,
powdery_mildew, black_spot.

**Roles:** YOLOv11 = localization + tracking only; ResNet50 = the sole disease
predictor. Backend is configurable (`DISEASE_BACKEND=classifier|yolo|auto`,
default resolves to the classifier when `weights/leaf_classifier.pt` exists).
EfficientNetV2-S and MobileNetV3-Large checkpoints remain in `weights/benchmark/`
for future experimentation but are not used in production.



This section documents the work to improve the **leaf detection / localization
model** (the object detector that finds and locates individual leaves in a
frame). It is independent of, and does **not** modify, the disease/health
classification model, which already performs well.

## What this model is

In the pipeline, leaf detection is the YOLOv11 stage that takes a frame (or a
plant ROI) and returns bounding boxes around individual leaves
(`smart_leaf_detection/leaf_detector.py`, `class 0 = leaf`). Its output feeds
leaf tracking → normalization → the (unchanged) ResNet/disease classifier.

## New training dataset

- **Images:** `cvat_frames_000000_000905/` — 906 drone video frames (`.jpg`).
- **Labels:** `leaf_labels/labels/train/` — 854 per-image YOLO label files,
  single class `leaf`.

The raw images and label files are treated as **read-only** and are never
modified. A cleaned, split copy is generated under `datasets/leaves_yolo/`.

## Bugs found and fixed

| # | Date | Issue | Root cause | Fix | Files |
|---|------|-------|-----------|-----|-------|
| 1 | 2026-06-22 | Labels won't train as YOLO detection | CVAT exported a **6th column (track id)** on most label lines (`class x y w h track_id`), inconsistently mixed with 5-column lines. YOLO detection requires exactly 5 columns. | `prepare_leaf_dataset.py` keeps only the first 5 tokens per line, dropping the track id. | `leaf_labels/` (read), `datasets/leaves_yolo/` (clean output) |
| 2 | 2026-06-22 | `train.txt` unusable | It lists `data/images/train/frame_XXXXXX.png` — a non-existent directory **and** the wrong extension (`.png`; real frames are `.jpg`). | We don't use `train.txt`; the prep script pairs images↔labels by stem and writes a fresh `data.yaml`. | `leaf_labels/train.txt` (ignored) |
| 3 | 2026-06-22 | No way to evaluate | `leaf_labels/data.yaml` defines only `train:` (no `val`/`test`). | Prep script creates train/val/test splits and a complete `data.yaml`. | `datasets/leaves_yolo/data.yaml` |
| 4 | 2026-06-22 | Optimistic metrics risk | Frames are consecutive video frames; adjacent frames are near-duplicates. A random split leaks near-identical images across train/test. | **Grouped temporal block split**: consecutive frames stay in the same split. | `prepare_leaf_dataset.py` |
| 5 | 2026-06-22 | 52 images had no labels | 906 images vs 854 label files. | Unlabeled frames are excluded by default (avoids teaching "no leaves"); `--include-unlabeled` keeps them as background. Reported in `dataset_report.json`. | `prepare_leaf_dataset.py` |
| 6 | 2026-06-22 | App pointed at a non-existent model | `leaf_weights_path` default was `yolo11_leaves.pt`, which does not exist; no trained leaf detector shipped. | Default now `weights/leaf_best.pt` (produced by the improved training run). | `smart_leaf_detection/config.py`, `run_on_frames.py` |
| 7 | 2026-06-22 | Web app never used a leaf detector | `worker.py` ran the disease model directly on whole frames; the dedicated leaf detector was unused. | Added a two-stage flow: leaf detector locates leaves → disease model classifies each crop. Gated on `weights/leaf_best.pt` existing (safe fallback otherwise); `LEAF_CONF` env var controls the threshold. | `webapp/backend/app/worker.py` |
| 8 | 2026-06-25 | Backend crashed on startup (Python 3.9) | `schemas.py` used PEP 604 `X | None` type hints, which pydantic v2 evaluates at runtime via `get_type_hints` — the `|` operator on types is unsupported on Python 3.9 (`TypeError: unsupported operand type(s) for \|`). | Replaced the union hints in the pydantic models with `typing.Optional[...]`, which evaluates correctly on 3.9 and removes the runtime dependency. `eval_type_backport` is also pinned in `requirements.txt` as a backstop. | `webapp/backend/app/schemas.py`, `webapp/backend/requirements.txt` |
| 9 | 2026-06-25 | Plant-centric view; no leaf tracking | The web app gave every detection a brand-new "plant" id and selected frames with a green-pixel heuristic, so leaves weren't tracked and the view wasn't leaf-centric. | Rewrote the worker to be **leaf-centric with tracking**: it focuses on the sharpest, most leaf-dense *contiguous* segment (frames in focus detect many more leaves), runs the trained leaf detector in **ByteTrack tracking mode** (stable per-leaf IDs across frames), classifies each leaf crop with the disease model, and aggregates per leaf. One result = one tracked leaf. Tunables: `FRAME_STRIDE_SEC`, `MAX_LEAF_FRAMES`, `MIN_LEAVES`, `MIN_TRACK_LEN`, `LEAF_CONF`. | `webapp/backend/app/worker.py` |
| 10 | 2026-06-22 | Frames with no detected leaves were still saved/shown | The worker pre-created a `FrameRecord` for every selected frame and saved an annotated image even when the detector found zero leaves, cluttering the results with useless frames. | Frame records and annotated images are now created **lazily, only for frames that actually contain at least one detected leaf**; empty frames are skipped entirely. `total_frames` reports the count of leaf-containing frames. | `webapp/backend/app/worker.py` |
| 11 | 2026-06-26 | Resuming the improved run crashed: `SyntaxError: 'cls_pw' is not a valid YOLO argument` | The interrupted run's `last.pt`/`best.pt` were written by an **older Ultralytics** that stored `cls_pw` in the checkpoint's `train_args`. After upgrading to Ultralytics 8.4.24, `--resume` rebuilds the validator config via `get_cfg`, which **strictly rejects unknown keys**. The training loop tolerates the stale key (it prints `cls_pw=0.0`), but validator construction does not, so resume crashed right after the optimizer was built. | Added `training/leaf_detection/fix_resume_checkpoint.py`, which backs up each checkpoint (`*.bak`) and strips any `train_args` key not present in the installed Ultralytics' `DEFAULT_CFG_DICT` (here `cls_pw`). Re-run resume afterwards. Model weights are untouched. | `training/leaf_detection/fix_resume_checkpoint.py` (new); repaired `runs/leaves/leaf_improved/weights/{last,best}.pt` |
| 12 | 2026-06-26 | Web app stored unusable file paths on Windows | `worker.py` made paths relative with `str(path).replace(str(OUTPUT_BASE) + "/", "")`, which assumes `/` separators. On Windows paths use `\`, so the prefix never matched and absolute paths leaked into the DB / served URLs. | Added `_rel_to_output()` using `Path.relative_to(OUTPUT_BASE).as_posix()` so stored paths are always forward-slash relative (correct URLs on every OS). | `webapp/backend/app/worker.py` |
| 13 | 2026-06-26 | Default device `"cuda"` was Mac/Windows-hostile | `PipelineConfig.device` defaulted to `"cuda"` and `DiseaseClassifier` only fell back cuda→cpu, ignoring Apple-Silicon MPS and crashing if someone forced an unavailable backend. | New `smart_leaf_detection/device_utils.py` resolves `"auto"` → CUDA → MPS → CPU; config/classifier default to `"auto"`; training & eval scripts route `--device` through it so an unavailable choice degrades to CPU instead of crashing. The `test_default_values` assertion was updated `"cuda"`→`"auto"` to match. | `smart_leaf_detection/device_utils.py` (new), `config.py`, `disease_classifier.py`, `training/leaf_detection/{train,evaluate}_leaf_detector.py`, `tests/test_pipeline_config.py` |
| 14 | 2026-06-26 | WebApp "classified" diseases with the YOLO **detector** on leaf crops, and the UI claimed a ResNet50 + plant-detection pipeline that didn't exist | The detector run on a tight single-leaf crop often found no box → leaf silently defaulted to "healthy" (healthy-biased results); UI text described modules that were never executed. | Added a dedicated disease **image classifier** pipeline (dataset builder from YOLO labels with leakage-safe splits + canonical mapping, multi-arch trainer defaulting to EfficientNetV2-S, evaluator with YOLO comparison, reusable `LeafDiseaseClassifier`). Worker now selects backend via `DISEASE_BACKEND` (auto/classifier/yolo), reusing one dispatcher; YOLO stays only for leaf localization/tracking. UI text rewritten to the real pipeline. | `training/disease_classification/*` (new), `smart_leaf_detection/leaf_disease_classifier.py` (new), `webapp/backend/app/worker.py`, `webapp/frontend/src/components/{InvestigationTab,ResultsTab}.tsx`, `requirements.txt` |
| 15 | 2026-06-26 | `build_classification_dataset.py` crashed: `KeyError: Unmapped source label 'Target_Spot' (normalized 'target spot')` | `class_mapping._normalize` converts `_`→space, but several mapping-dict **keys** still contained underscores (`target_spot`, `early_blight`, ...), so they could never match the normalized lookup. The build aborted on the first such label, so `datasets/leaf_clf` was never created and the benchmark then skipped every model. | Keys are now written in readable form and **normalized at construction** (`_RAW_TO_CANONICAL = {_normalize(k): v ...}`), so space/underscore form no longer matters. `build_id_to_canonical` now returns an `UNMAPPED` sentinel (recorded + skipped by the builder) instead of raising, so one stray label can't kill a long run. Worker also loads the YOLO disease model as a fallback when `DISEASE_BACKEND=classifier` is forced before classifier weights exist. | `training/disease_classification/class_mapping.py`, `webapp/backend/app/worker.py` |
| 16 | 2026-06-26 | Dataset build crashed on one PlantDoc image: `OSError: [Errno 63] File name too long` in `_find_image` | A PlantDoc label stem was long enough that `Path.exists()`/`glob` on the candidate image path exceeded the OS filename limit (`ENAMETOOLONG`), and the unguarded filesystem call aborted the entire build. | `_find_image` now wraps all filesystem calls in `try/except OSError`, returning a reason (`image_path_too_long` / `invalid_image_path` / `image_not_found`) instead of raising. Bad files are counted in `discarded` and recorded (dataset, label file, stem, reason) under `problem_files` in `dataset_report.json`; the build continues. Crop **writes** are also guarded and output filenames are sanitized + length-bounded + hash-suffixed so a long source stem cannot produce an over-long output path. | `training/disease_classification/build_classification_dataset.py` |
| 17 | 2026-06-26 | Web UI was frame-centric and "Plant"-labelled, contradicting the leaf-centric pipeline; the final verdict's confidence/agreement were not exposed honestly | The old Investigation/Results tabs browsed every frame and grouped by "Plant", and the stored confidence was the mean over all views (dragged down by dissenting frames). The temporal-aggregation evidence (per-view predictions, agreement, winning-class confidence) was computed but not surfaced. | Rebuilt the UI as a **leaf-centric inspection tool**: Overview (coverage funnel + verdict + distribution + needs-review) → Leaves gallery (diseased-first) → Leaf Detail (representative frame with highlighted bbox + per-observation evidence list with ✓/⚠ dissent + honest consensus that respects the confidence-weighted vote). Backend now stores the **winning-class** confidence, `views_total`/`views_agreeing`/`weighted_decision`, per-observation `frame_index`, and coverage stats (`total_video_frames`/`relevant_frames`/`total_detections`); an idempotent SQLite `ensure_columns()` migration adds the new columns without data loss. Frames are now evidence, not navigation. | `webapp/backend/app/{models,database,main,worker,schemas}.py`, `routes/flight_routes.py`, `webapp/frontend/src/{leaf.ts,api.ts,pages/Dashboard.tsx,components/{OverviewTab,LeavesTab,LeafDetail}.tsx}` (deleted InvestigationTab/ResultsTab) |
| 18 | 2026-06-26 | Tracking lost leaf identity on drone footage, so almost no leaves reached the Results page | On a real flight, 79 leaf detections across 17 frames collapsed to only **2** tracked leaves (both `views_total=2`). ByteTrack has **no global motion compensation**; with a constantly moving drone camera each leaf shifts too far between sampled frames for IoU/motion association, so the tracker assigns a **fresh ID nearly every frame**. The one-view "singletons" that result are then dropped by `MIN_TRACK_LEN=2` before reaching the DB. Lowering `FRAME_STRIDE_SEC` to 0.3 helped only marginally. | Added a **configurable tracker backend** via `LEAF_TRACKER` (`botsort`\|`bytetrack`). **BoT-SORT** enables **Global Motion Compensation** (`gmc_method: sparseOptFlow`) which cancels the camera's frame-to-frame motion before association, plus tuned `track_buffer=60` so a briefly-occluded leaf keeps its ID. Version-matched tracker configs live under `webapp/backend/app/trackers/`; `track()` now receives `tracker=<config>`. A read-only `compare_trackers.py` A/B tool reports stable leaves, avg/median views, singletons and final cards, and recommends promoting BoT-SORT to default **only if** it beats ByteTrack on both stable leaves and average views. ByteTrack kept as fallback; default stays `bytetrack` until the comparison confirms the win. ResNet50 classifier and disease-aggregation logic unchanged. See the "Bug #18" section below for the comparison protocol. | `webapp/backend/app/trackers/{botsort_gmc,bytetrack}.yaml` (new), `webapp/backend/app/worker.py`, `webapp/backend/compare_trackers.py` (new) |

**Testing recommendation for the fixes:** run `python -m pytest tests/ -v`
(the config default assertion was updated to match fix #6), and inspect
`training/leaf_detection/reports/dataset_report.json` after preparing the
dataset to confirm box counts, track-id lines fixed, and split sizes.

### Bug #18 — Leaf tracker selection & BoT-SORT comparison

The leaf tracker is now configurable via the `LEAF_TRACKER` environment variable:

| Value | Tracker | Notes |
|-------|---------|-------|
| `botsort` (default) | BoT-SORT **+ GMC** | Global Motion Compensation for the moving drone camera; tuned `track_buffer=60`. Selected as the production tracker — visibly tracks the same leaf across far more frames. |
| `bytetrack` | ByteTrack baseline | No motion compensation; the prior behaviour, kept as a fallback. |

Configs live under `webapp/backend/app/trackers/` (`bytetrack.yaml`, `botsort_gmc.yaml`)
and mirror the installed Ultralytics tracker schema.

**A/B comparison protocol** — run both analyses on the **same** video with
`MIN_TRACK_LEN=1` so the one-view singletons (which production normally drops)
are visible, then compare:

```bash
# from webapp/backend, with the project conda/venv active
# 1) baseline — upload the video in the UI, let it finish
FRAME_STRIDE_SEC=0.3 LEAF_TRACKER=bytetrack MIN_TRACK_LEN=1 uvicorn app.main:app --port 8000

# 2) candidate — restart, upload the SAME video again
FRAME_STRIDE_SEC=0.3 LEAF_TRACKER=botsort  MIN_TRACK_LEN=1 uvicorn app.main:app --port 8000

# 3) compare the two most recent flights (newest=botsort, prior=bytetrack)
python compare_trackers.py
```

`compare_trackers.py` is **read-only** (opens the SQLite DB in `mode=ro`), prints
a side-by-side table (total detections, stable tracked leaves, avg/median
`views_total`, 1-view singletons, 2+-view leaves, final result cards) and a
verdict, and writes a timestamped CSV under `webapp/backend/reports/`.

**Promotion rule:** if BoT-SORT increases **both** the number of stable tracked
leaves (≥2 views) **and** the average views per leaf, make it the default by
setting `_DEFAULT_TRACKER = "botsort"` in `webapp/backend/app/worker.py`.
Otherwise keep ByteTrack. **Status:** BoT-SORT was selected as the production
default (2026-06-26); ByteTrack remains available via `LEAF_TRACKER=bytetrack`.

## Improvements applied

- **Data:** corrected labels, deterministic leakage-safe split, dataset QA report.
- **Architecture:** baseline `yolo11n` → improved `yolo11s`.
- **Input size:** 640 → **960 px** (leaves are small/dense in drone frames).
- **Optimization:** AdamW + cosine LR + warmup, longer schedule (150 epochs)
  with early stopping (`patience=40`).
- **Augmentation:** mosaic with `close_mosaic=15` (disabled for the final
  epochs to sharpen localization), lighter mixup, vertical+horizontal flips
  (valid for top-down drone views), moderate rotation/shear.
- **Thresholds:** evaluation sweeps the confidence threshold and recommends the
  best-F1 operating point for `leaf_confidence_threshold`.

## How to run (in your terminal / project venv)

> **Device flag (`--device`) is cross-platform.** Omit it to auto-select
> (CUDA → MPS → CPU). Or pass explicitly: `--device mps` (Apple Silicon),
> `--device 0` (first NVIDIA GPU), `--device cpu`. An unavailable choice
> degrades to CPU instead of crashing, so the same command is safe on Mac and
> Windows.

```bash
# activate first: macOS/Linux -> source venv/bin/activate
#                 Windows     -> .\venv\Scripts\Activate.ps1
pip install -r requirements.txt     # ultralytics, torch, etc.

# 1) Build the clean dataset (writes datasets/leaves_yolo/ + a QA report)
python training/leaf_detection/prepare_leaf_dataset.py

# 2) Full before/after run (trains baseline + improved, evaluates both on the
#    held-out TEST split, writes a comparison). Drop --device to auto-select.
python training/leaf_detection/run_leaf_pipeline.py --device mps   # or: 0 / cpu / (omit)

# --- or run stages individually ---
python training/leaf_detection/train_leaf_detector.py --preset baseline --device mps
python training/leaf_detection/train_leaf_detector.py --preset improved --device mps
python training/leaf_detection/evaluate_leaf_detector.py --weights weights/leaf_baseline.pt --tag baseline
python training/leaf_detection/evaluate_leaf_detector.py --weights weights/leaf_improved.pt --tag improved --sweep-conf

# Quick smoke test (tiny epochs, just to verify the wiring end-to-end):
python training/leaf_detection/run_leaf_pipeline.py --baseline-epochs 3 --improved-epochs 3 --device mps
```

### Verify the labels visually

Before trusting a training run, draw the ground-truth boxes on a frame:

```bash
python training/leaf_detection/draw_labels.py \
  --image cvat_frames_000000_000905/frame_000000.jpg
# -> training/leaf_detection/reports/label_preview_frame_000000.jpg
```

### Better / full training (robust to the Mac throttling)

The first improved run stalled around epoch 32 of 150 because the laptop
thermally throttled (per-epoch time ballooned). To get a fully-trained model:

```bash
# macOS: keep the Mac awake + cache images to disk so epochs don't stall on I/O
caffeinate -dimsu python training/leaf_detection/train_leaf_detector.py \
  --preset improved --device mps --cache disk

# if a run gets interrupted, resume it from its last checkpoint:
caffeinate -dimsu python training/leaf_detection/train_leaf_detector.py \
  --preset improved --device mps --resume
```

```powershell
# Windows (PowerShell): NVIDIA GPU build recommended; --cache disk helps I/O.
python training\leaf_detection\train_leaf_detector.py --preset improved --device 0 --cache disk
# resume an interrupted run:
python training\leaf_detection\train_leaf_detector.py --preset improved --device 0 --resume
```

`caffeinate -dimsu` (macOS only) stops the display/disk/system from sleeping
during the run. **On Windows**, prevent sleep during long training via *Settings
→ System → Power & battery → Screen and sleep* (set to *Never*), or run
`powercfg /change standby-timeout-ac 0` in an admin shell; there is no
`caffeinate` equivalent to prefix the command with.
`--cache disk` caches preprocessed images to cut per-epoch data-loading time;
`--resume` continues from `runs/leaves/leaf_improved/weights/last.pt` to the
original epoch target. When training finishes it overwrites
`weights/leaf_best.pt`, which the web app loads automatically.

> **If `--resume` fails with `'cls_pw' is not a valid YOLO argument`** (or a
> similar "not a valid YOLO argument" error): the checkpoint was written by an
> older Ultralytics whose `train_args` contains keys the installed version no
> longer accepts. Repair the checkpoints once, then resume:
>
> ```bash
> ./venv/bin/python training/leaf_detection/fix_resume_checkpoint.py \
>   runs/leaves/leaf_improved/weights/last.pt \
>   runs/leaves/leaf_improved/weights/best.pt
> ```
>
> This backs up each file to `*.bak` and strips only the stale config keys; the
> model weights are untouched. See bug #11 above.

Outputs:
- Cleaned dataset: `datasets/leaves_yolo/`
- Reports: `training/leaf_detection/reports/` — `dataset_report.json`,
  `clean_manifest.csv`, `metrics_baseline.json`, `metrics_improved.json`,
  `metrics_comparison.csv`, `comparison_summary.json`, `pipeline_progress.json`
- Deployed model: `weights/leaf_best.pt` (used by `run_on_frames.py` and the
  `Pipeline`/`LeafDetector` via `PipelineConfig.leaf_weights_path`).

## Before / After metrics

Evaluated on the held-out **test** split (`datasets/leaves_yolo`, grouped
temporal split, 120 images), IoU 0.6, mAP computed at conf 0.001.

The **improved** model (`weights/leaf_best.pt`, yolo11s @ 960 px) numbers below
are real, from `training/leaf_detection/reports/metrics_improved.json`
(evaluated 2026-06-25):

| Metric | Baseline (yolo11n @ 640) | Improved (yolo11s @ 960) |
|--------|--------------------------|--------------------------|
| Precision | _run baseline eval_ | **0.678** |
| Recall | _run baseline eval_ | **0.818** |
| mAP@0.5 | _run baseline eval_ | **0.787** |
| mAP@0.5:0.95 | _run baseline eval_ | **0.476** |

> The baseline weights (`weights/leaf_baseline.pt`) already exist but were not
> evaluated in the same pass, so the "before" column is pending one quick eval
> (no retraining needed — evaluation is a single pass over the 120 test images):
>
> ```bash
> python training/leaf_detection/evaluate_leaf_detector.py \
>   --weights weights/leaf_baseline.pt --tag baseline --device mps
> ```
>
> This writes `metrics_baseline.json`; the Δ column can then be filled from
> `metrics_comparison.csv`.

**Confidence-threshold sweep (improved model).** Recall peaks at conf ≤ 0.35
(R = 0.818) and precision rises as conf increases. The web app default
`LEAF_CONF=0.3` keeps recall high (detects the most leaves) while suppressing
the lowest-confidence noise; raise it toward 0.5–0.6 if you prefer fewer, more
certain boxes. Full sweep in `metrics_improved.json` → `conf_sweep`.

> Note: on MPS, NMS time-limit warnings during the conf 0.001 mAP pass can
> slightly deflate the reported mAP; the sweep values are more representative of
> real operating points.

## Wiring into the web app

- `Pipeline` / `LeafDetector` and `run_on_frames.py` default to
  `weights/leaf_best.pt`, so once training produces that file the leaf detector
  is used automatically.
- The FastAPI worker (`webapp/backend/app/worker.py`) now runs a **two-stage
  flow** when `weights/leaf_best.pt` exists:
  1. the **leaf detector** locates each leaf in the frame, then
  2. the **unchanged disease model** classifies each leaf crop.
  If the leaf weights are absent, the worker falls back to its previous
  behaviour (disease model on the whole frame, then green-segmentation), so
  nothing breaks before training finishes.
- The leaf-detector confidence used by the worker is configurable via the
  `LEAF_CONF` environment variable (default `0.3`). Set it to the value
  recommended by the evaluation conf sweep
  (`training/leaf_detection/reports/metrics_improved.json` →
  `recommended_conf_by_f1`), e.g.:

  ```bash
  LEAF_CONF=0.3 PYTHONPATH=".:../.." python -m uvicorn app.main:app --port 8000 --reload
  ```

  The disease/health model and its weights are not modified.

## Improving leaf-detection accuracy — research-backed roadmap

The current improved detector (yolo11s @ 960 px) reaches mAP@0.5 ≈ 0.79,
recall ≈ 0.82 on the held-out test split. The techniques below are drawn from
recent small-object / video-detection literature and are ordered by
effort-to-payoff. *Content from the sources was rephrased for licensing
compliance; see the inline links for the originals.*

### Quick wins (no retraining)

1. **Sliced inference (SAHI) for high-res drone frames.** Leaves are small and
   dense; running detection on overlapping tiles of a large frame and merging
   the results recovers small objects a single down-scaled pass misses. The
   technique reports several points of AP improvement on aerial datasets, and
   Ultralytics ships native support.
   ([SAHI paper, arXiv 2202.06934](https://arxiv.org/abs/2202.06934v2);
   [Ultralytics SAHI guide](https://docs.ultralytics.com/guides/sahi-tiled-inference);
   adaptive follow-up [ASAHI](https://arxiv.org/abs/2604.19233)).
   *Tradeoff:* slower inference. Good for the offline video pipeline; gate it
   behind a flag for the live worker.

2. **Test-time augmentation (TTA).** Pass `augment=True` to `model.predict`/
   `val` for a small recall/mAP bump at the cost of speed — useful when
   generating the "best" annotated frames offline.

3. **Tune the operating confidence.** Keep using the eval conf-sweep's best-F1
   point (`metrics_improved.json → recommended_conf_by_f1`) for `LEAF_CONF`.

### Medium effort (retraining)

4. **Push input resolution / add a small-object head.** Try `--imgsz 1280`, or
   a YOLO variant with a P2 (high-resolution) detection head — small-object
   aggregation designs improve aerial small-object AP on VisDrone.
   ([MFA-YOLO, Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-32247-9)).

5. **Exploit the temporal dimension (it's video!).** Instead of treating each
   frame independently, aggregate evidence across consecutive frames. Two
   complementary routes: (a) tracking-based voting — require a leaf to be
   tracked across ≥N frames and average its track confidence (the worker
   already uses ByteTrack + `MIN_TRACK_LEN`; strengthen the voting); (b)
   multi-frame input, where stacking neighbouring frames stabilises detections
   on blurry/occluded frames.
   ([Lightweight multi-frame YOLO, arXiv 2506.20550](https://arxiv.org/html/2506.20550v1);
   [flow-guided feature aggregation, arXiv 1703.10025](https://ar5iv.labs.arxiv.org/html/1703.10025);
   [greenhouse counting with inter-frame prediction, MDPI Agronomy 2025](https://www.mdpi.com/2073-4395/15/5/1135/htm)).

### Larger effort (data + architecture)

6. **More data diversity.** All frames come from one flight, which caps
   generalization. Add flights at different altitudes/lighting/fields; consider
   pseudo-labelling new flights with the current model then human-correcting.

7. **Tomato-tuned architecture tweaks.** Recent tomato-leaf work adds dynamic
   convolution / attention modules to a YOLO backbone for better feature
   representation on leaf textures.
   ([Improved YOLOv8n for tomato leaf disease, Sci. Rep. 2025](https://www.nature.com/articles/s41598-025-00405-8);
   [SBCS-YOLOv5s for occluded/overlapping objects](https://www.frontiersin.org/articles/10.3389/fpls.2023.1292766/full)).

8. **Step up model size** (`yolo11m`) if the training/inference budget allows —
   trades speed for accuracy.

> Keep the existing **annotation policy** (only clear, camera-facing leaves are
> labelled) when adding data, so new labels stay consistent with the current
> ground truth and don't teach the model to fire on blurry leaves.



- Class `0` means `leaf`; any other class index in a raw label is dropped.
- A 6th token on a label line is a CVAT track id and is removed.
- **Annotation policy (confirmed with the annotator):** only **clear,
  camera-facing** leaves are labelled. Blurry, edge-on, or out-of-focus leaves
  are intentionally left **unlabelled** (treated as background). The detector's
  job is therefore to find *salient, front-facing* leaves — not every leaf in
  the frame. Consequently: (a) "missed" blurry leaves are usually correct
  behaviour, not errors; (b) augmentation deliberately avoids blur / random
  erasing so the model is never taught to fire on unclear leaves; and (c) this
  aligns with the web app's focus-weighted frame selection (sharper frames →
  more labelled-style leaves).
- Coordinates are normalized; out-of-range values are clamped to `[0, 1]`.
- Boxes with non-positive width/height are dropped.
- Adjacent frames are near-duplicates, so splitting is done by contiguous
  frame blocks (default block size 20) to avoid train/test leakage.
- Unlabeled frames are excluded unless `--include-unlabeled` is passed.

## Limitations

- Single source video: all frames come from one drone flight, so the detector
  may not generalize to very different fields, altitudes, or lighting without
  additional data.
- Annotation quality is taken as-is (boxes derived from CVAT); missed leaves in
  the ground truth will cap achievable recall and can penalize precision.
- The grouped temporal split reduces but does not fully eliminate similarity
  between splits within the same flight; absolute numbers should be read as
  in-flight performance.
- Training/evaluation must be run in an environment with `ultralytics` + a
  working PyTorch (MPS/CUDA/CPU); metrics above are populated after that run.
