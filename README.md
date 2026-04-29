# 🌿 SmartLeafDetection

**Hybrid Deep Learning System for Drone-Based Tomato Leaf Disease Detection**

A complete end-to-end system that processes drone video of tomato fields to detect and classify leaf diseases using YOLOv11 and ResNet50.

## Features

- **Video Processing**: Upload drone video → frame extraction → disease analysis
- **Disease Detection**: YOLOv11s trained on 6,500+ images (Roboflow + PlantDoc + CVAT + Tomato-6K)
- **7 Disease Classes**: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Target Spot, Healthy, Black Spot
- **Full Pipeline**: Plant detection → Leaf detection → ByteTrack tracking → Disease classification → Temporal aggregation → Plant-level inference
- **Web Application**: React + FastAPI dashboard with upload, investigation, and results tabs
- **GPS Support**: Associates results with drone SRT telemetry data

## Quick Start

### 1. Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Test on sample images
```bash
# Download YOLO weights first (see Model Weights section below)
python run_on_frames.py \
  --frames test1.jpg test2.jpg test3.jpg \
  --disease-weights weights/best.pt \
  --confidence 0.3
```

### 3. Run the web application

**Terminal 1 — Backend:**
```bash
source venv/bin/activate
cd webapp/backend
PYTHONPATH=".:../.." python -m uvicorn app.main:app --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd webapp/frontend
npm install
npm run dev
```

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
│   ├── disease_classifier.py   # ResNet50 classification
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
│   ├── train_resnet50.py       # ResNet50 fine-tuning
│   ├── train_yolo_leaves.py    # Leaf detection training
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
→ Normalization (224×224) → Disease Classification (ResNet50)
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
source venv/bin/activate
python -m pytest tests/ -v
```

## Technology Stack

- **Detection**: YOLOv11 (Ultralytics)
- **Classification**: ResNet50 (PyTorch/torchvision)
- **Tracking**: ByteTrack-style IOU tracker
- **Backend**: FastAPI + SQLAlchemy + SQLite
- **Frontend**: React + TypeScript + Vite
- **CV**: OpenCV

## Authors

Noy Ariel — Technion Final Project
