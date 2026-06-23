# Pokémon Type Classification

A progressive deep learning study on image classification applied to a Pokémon type dataset, covering three modelling approaches of increasing complexity — from a basic MLP to fine-tuned transfer learning.

The project was structured as a Kaggle competition, with each task producing a submission CSV for evaluation. All notebooks run on Google Colab with a T4 GPU.

---

## Dataset

The dataset contains ~3,600 labelled Pokémon images (64×64 PNG, RGB) across **9 type classes**:

`Bug · Fighting · Fire · Grass · Ground · Normal · Poison · Rock · Water`

| Split | Size |
|-------|------|
| Train | ~2,880 images |
| Test  | held-out (no labels) |

**Class distribution** is moderately imbalanced (imbalance ratio ≈ 2.76 — Water is the most frequent class, Ground the least). Normalised entropy ≈ 0.97, indicating reasonable diversity across classes. All images share a uniform 64×64 resolution with no missing files or duplicate IDs.

---

---

## Tasks

### Task 1 — Multilayer Perceptron (MLP)

`task1.ipynb`

A fully connected network trained on flattened pixel vectors (64×64×3 = 12,288 input features).

**Architecture** — 4-layer MLP:
Input (12288)  
→ Linear(1024)  
→ BatchNorm1d(1024)  
→ LeakyReLU  
→ Dropout(0.5)  

→ Linear(256)  
→ BatchNorm1d(256)  
→ LeakyReLU  
→ Dropout(0.4)  

→ Linear(64)  
→ BatchNorm1d(64)  
→ LeakyReLU  
→ Dropout(0.3)  

→ Linear(9)


**Key design choices:**
- Kaiming Normal weight initialisation (adapted for LeakyReLU)
- Square-root class weights in `CrossEntropyLoss` to address imbalance
- Adam optimiser (`lr=1e-4`, `weight_decay=1e-4`) with `ReduceLROnPlateau`
- Early stopping (patience = 15), model checkpointed on best validation loss
- Training augmentation: horizontal flip, random resized crop, random rotation

---

### Task 2 — Convolutional Neural Network (CNN)

`task2.ipynb`

A custom CNN that preserves spatial structure, contrasting directly with the MLP baseline.

**Architecture** — 3 convolutional blocks + Global Average Pooling head:
Input (3×64×64)

→ Block 1
  - Conv2d(32)
  - BatchNorm2d(32)
  - ReLU
  - Conv2d(32)
  - BatchNorm2d(32)
  - ReLU
  - MaxPool2d
  - Dropout2d(0.1)
  - Output: 32×32×32

→ Block 2
  - Conv2d(64)
  - BatchNorm2d(64)
  - ReLU
  - Conv2d(64)
  - BatchNorm2d(64)
  - ReLU
  - MaxPool2d
  - Dropout2d(0.2)
  - Output: 64×16×16

→ Block 3
  - Conv2d(128)
  - BatchNorm2d(128)
  - ReLU
  - Conv2d(128)
  - BatchNorm2d(128)
  - ReLU
  - MaxPool2d
  - Dropout2d(0.3)
  - Output: 128×8×8

→ GlobalAveragePooling
  - Output: (128,)

→ Linear(256)
→ BatchNorm1d(256)
→ ReLU
→ Dropout(0.5)

→ Linear(9)

→ Output: 9 classes

Global Average Pooling replaces a flat dense layer, drastically reducing parameter count and improving regularisation.

**Key design choices:**
- Same square-root class weighting as Task 1
- Adam (`lr=1e-3`, `weight_decay=1e-4`) with `ReduceLROnPlateau` (factor 0.5, patience 7)
- Early stopping (patience = 7); training tracks loss, accuracy, and macro F1 per epoch
- Stronger augmentation: vertical flip, ImageNet-style normalisation

---

### Task 3 — Transfer Learning & Fine-Tuning

`task3.ipynb`

Transfer learning from ImageNet-pretrained backbones, with progressive fine-tuning and advanced regularisation. Images are resized to 224×224 to match pretrained input requirements.

**Models evaluated in frozen baseline (10 epochs, head only):**

| Model | Val Accuracy |
|---|---|
| ResNet-50 | 21.34% |
| EfficientNet-B0 | 18.41% |
| EfficientNet-B2 | 14.64% |
| MobileNetV3-Large | — |

ResNet-50 was selected for full fine-tuning based on its superior baseline performance.

**Fine-tuning strategy (section 3.2 → 3.3):**
- Frozen backbone with head-only training initially
- Progressive unfreezing: `layer4` unlocked from epoch 1; `layer3` unlocked at epoch 15
- Discriminative learning rates: backbone `lr=1e-5`, head `lr=1e-3`
- Linear LR warm-up for the first 5 epochs
- **MixUp** (α=0.4) applied during training to smooth decision boundaries
- Label smoothing (ε=0.1) in `CrossEntropyLoss`
- Gradient clipping (`max_norm=1.0`) to stabilise early backbone updates
- Stronger dropout in the head (0.5 vs 0.2)
- Early stopping monitored on **val accuracy** (patience=12) — not val loss, which is inflated by label smoothing

**Best result:** ~61% validation accuracy at epoch 75 (Macro F1 ≈ 0.58).

---

## Setup

All notebooks are designed to run on **Google Colab** with a **T4 GPU**.

1. Upload the dataset to Google Drive at:
2. Open any notebook in Colab.
3. Switch the runtime to **GPU (T4)**: `Runtime → Change runtime type → T4 GPU`.
4. Mount Drive and run all cells.

**Main dependencies** (pre-installed on Colab):
- PyTorch (`torch`)
- TorchVision (`torchvision`)
- Scikit-learn (`scikit-learn`)
- Pandas (`pandas`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Seaborn (`seaborn`)
- Pillow (`PIL`)
- SciPy (`scipy`)
- OpenCV (`opencv-python`)
- Scikit-image (`scikit-image`)

---

## Results Summary

| Model | Val Accuracy | Notes |
|---|---|---|
| MLP | — | Baseline; no spatial awareness |
| Custom CNN | — | Spatial features, GAP head |
| ResNet-50 (fine-tuned) | ~61% | Best overall; transfer learning |

> Accuracy figures for MLP and CNN reflect Kaggle submission scores and may differ from local validation metrics depending on the run.

---

## Common Observations Across Tasks

- **Rock** and **Fighting** are consistently the hardest types to classify (low sample count, visually ambiguous).
- **Water**, **Poison**, and **Fire** are the best-classified types across all models.
- Class imbalance is moderate and manageable with square-root class weighting.
- Background clutter in images weakens pure colour-based signals (RGB statistics alone are not discriminative enough).