# Model Card — Leaf-Disease Classifier (deployed)

## Selected architecture
**resnet50** (ImageNet-pretrained, fine-tuned). Selected by composite score
(0.6·macro-F1 + 0.25·speed + 0.15·footprint);
primary quality criterion is **validation Macro-F1**.

## Training dataset
Leaf crops generated from the project's YOLO detection datasets
(tomato_diseases, merged_diseases, plantdoc, tomato_6k), unified to a
canonical taxonomy with source-image-level, leakage-free splits.
- Classes (11): healthy, bacterial_spot, early_blight, late_blight, leaf_mold, septoria_leaf_spot, target_spot, mosaic_virus, yellow_leaf_curl_virus, powdery_mildew, black_spot

## Data volumes
- Training images: 14958
- Validation images: 1923
- Test images: 1868

## Training configuration
arch=resnet50, img_size=224, epochs(max)=18, freeze_epochs=3, batch=32, lr=0.001, weight_decay=0.0001, label_smoothing=0.05, patience=6, seed=0, mixup=0.0, cutmix=0.0
- Optimizer/scheduler: {'type': 'AdamW', 'lr_head_warmup': 0.001, 'lr_after_unfreeze': 0.0001, 'weight_decay': 0.0001} / {'type': 'CosineAnnealingLR', 'T_max_phase1': 18, 'T_max_phase2': 15}
- Loss: class-weighted CrossEntropy + label smoothing
- Augmentation: realistic (rotation, flips, colour jitter, autocontrast, mild blur, mild noise);
  MixUp/CutMix: mixup=0.0, cutmix=0.0
- Reproducibility: full config + seeds + augmentation + dataset stats in `experiment_metadata.json`

## Performance
- **Best validation Macro-F1:** 0.8202271933388148
- **Test accuracy:** 0.9424
- **Test Macro-F1:** 0.8487
- Avg inference: 13.967 ms/img on mps
- Parameters: 23.53 M; size: 90.06 MB

## Known limitations
- Domain gap: trained mostly on studio/field single-leaf crops; accuracy on real
  drone-video crops is capped until fine-tuned on drone data.
- Rare classes (few crops) have weaker recall despite class weighting.
- Predicts one disease per leaf crop; co-occurring diseases on one leaf are not modelled.
- Single fixed test split: metrics are point estimates.

## Recommended use cases
- Per-leaf disease screening inside the SmartLeafDetection drone pipeline
  (leaf detector localizes/tracks; this model classifies each crop).
- Offline analysis of tomato-leaf imagery for the canonical disease set above.
- **Not** a substitute for expert agronomic diagnosis; treat as decision support.
