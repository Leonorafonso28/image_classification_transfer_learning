# Transfer Learning for Pokémon Type Classification

## Overview

This notebook applies Transfer Learning to classify Pokémon images into 9 types: Bug, Fighting, Fire, Grass, Ground, Normal, Poison, Rock, and Water. Pre-trained ImageNet backbones (ResNet-50, EfficientNet-B0/B2, MobileNetV3-Large) are evaluated with progressive fine-tuning, data augmentation, and regularisation.

## Dataset

1,194 training images across 9 classes. All images are 400×300 px PNG files. Class imbalance ratio of 4.20 (Water: 252 images vs Fighting: 60 images). Stratified 80/20 train/validation split. No missing files or duplicate IDs.

---

## Notebook Structure

### EDA

Full exploratory analysis: class distribution pie chart (type-coloured), normalised entropy (0.9469) and Gini index (0.8622) confirming moderate imbalance, mean RGB values per type, image sharpness via Laplacian variance (mean: 1515 — high but variable), resolution check, LBP-based texture complexity per class, and t-SNE on raw pixels to assess class separability.

---

### 3.1 — Model Selection & Frozen Baseline

Four models evaluated with a fully frozen backbone (head-only training, 10 epochs):

| Model | Val Acc (frozen baseline) |
|---|---|
| ResNet-50 | **21.34%** |
| EfficientNet-B0 | 18.41% |
| EfficientNet-B2 | 14.64% |
| MobileNetV3-Large | — |

**ResNet-50 selected as the primary model:** strongest frozen baseline, 2,048-dimensional feature space, and residual connections that are robust to vanishing gradients. Loss uses `CrossEntropyLoss` with class weights (`w_c = sqrt(N/n_c)`) to handle class imbalance.

---

### 3.2 — Fine-Tuning & Adaptation

ResNet-50 with partial backbone unfreezing (`layer4` only, `fine_tune_blocks=1`), discriminative learning rates (backbone: 1e-5, head: 1e-3), `ReduceLROnPlateau` (factor=0.3, patience=3), label smoothing (ε=0.1), gradient clipping (max_norm=1.0), early stopping (patience=7, monitoring `val_loss`).

**Outcome:** Early stopping fired prematurely at epoch 9. Root cause: label smoothing artificially inflates the val loss, causing the patience counter to fill before the model had time to adapt.

---

### 3.3 — Data Augmentation & Regularisation

Addresses the premature stopping issue from 3.2. Full set of techniques applied:

| Technique | Detail |
|---|---|
| Training augmentation | `RandomResizedCrop` (scale 0.7–1.0), H/V flips, ±15° rotation, `RandomErasing(p=0.2)` |
| Validation | Clean only — resize 256 + `CenterCrop(224)` + normalize |
| Dropout | Raised from 0.2 → **0.5** in the classification head |
| MixUp | α=0.4 — blends image pairs and their labels during training |
| LR warm-up | 5 linear epochs (0.1×lr → lr) to stabilise early head updates |
| Progressive unfreezing | `layer3` unfrozen at epoch 15; optimizer rebuilt with `lr_head=5e-4` |
| Early stopping | patience=12, monitored on **val_acc** (fixes the 3.2 problem) |

**Outcome:** Full 80 epochs completed. Best val_acc = **61.09%** (epoch 75), Macro F1 ≈ 0.58. Train loss ≈ 0.92 vs val loss ≈ 1.51 — moderate overfitting expected given the small dataset. Learning plateaued after epoch 63.

---

### 3.4 — Evaluation & Interpretation

Per-class classification report, confusion matrix heatmap, and training curves (loss, accuracy, backbone LR, head LR plotted separately). Summary metrics: overall accuracy and balanced accuracy.

## Dependencies

```
torch, torchvision, sklearn, matplotlib, seaborn
pandas, numpy, PIL, cv2, skimage, scipy
google.colab, CustomImageDataset  # local dataset module
```