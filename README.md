# melanoma-classifier

CNN-based melanoma classification on dermoscopic images.  
Graduate-level research project with **Prof. M. Emre Celebi**, University of Central Arkansas.

---

## What this is

A deep learning pipeline for binary classification of skin lesions as malignant (melanoma) or benign, trained on the ISIC dermoscopic imaging dataset. This was graduate-level coursework — I did this research as a high school student in collaboration with Prof. Celebi's lab.

The project placed at **MOSTRATEC 2024** (world's 2nd largest international science fair, 23 countries) — **3rd place overall**.

---

## Results

| Metric | Value |
|--------|-------|
| Dataset | ISIC 2020 (33,126 images) |
| Architecture | EfficientNet-B3 (pretrained, fine-tuned) |
| AUC-ROC | 0.891 |
| Sensitivity (melanoma recall) | 0.83 |
| Specificity | 0.87 |
| Training time | ~4h on Google Colab Pro (T4) |

---

## Approach

### Problem
Melanoma is the deadliest form of skin cancer. Early detection dramatically improves survival rates. Dermoscopic imaging provides detailed views of skin lesions, but manual diagnosis requires specialist expertise. Automated classification can support dermatologists in high-volume screening.

### Dataset
[ISIC 2020 Challenge dataset](https://www.isic-archive.com/): 33,126 training images, heavily class-imbalanced (~98% benign, ~2% melanoma).

### Key challenges
1. **Class imbalance** — 98:2 split. Naive training predicts "benign" always.
2. **Image variability** — different cameras, zoom levels, hair artifacts, lighting.
3. **Generalization** — model must not overfit to dataset-specific artifacts.

### Solutions
1. Weighted loss function + oversampling of melanoma class
2. Aggressive augmentation (flip, rotation, color jitter, cutout, random crop)
3. Test-time augmentation (TTA) — average predictions over 8 augmented views
4. Ensemble of EfficientNet-B3 + ResNet-50

---

## Architecture

```python
import timm
import torch.nn as nn

class MelanomaClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, 
            pretrained=pretrained, 
            num_classes=0  # remove head
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(self.backbone.num_features, 1)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        return self.head(features)
```

---

## Setup

```bash
git clone https://github.com/ErturkCan/melanoma-classifier
cd melanoma-classifier
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, torch, torchvision, timm, albumentations, pandas, scikit-learn, matplotlib

---

## Usage

**Download data:**
```bash
python data/download_isic.py
```

**Train:**
```bash
python train.py --model efficientnet_b3 --epochs 30 --batch-size 32
```

**Evaluate:**
```bash
python evaluate.py --checkpoint models/best.pth
```

**Predict on single image:**
```bash
python predict.py --image path/to/lesion.jpg --checkpoint models/best.pth
```

---

## Project Structure

```
melanoma-classifier/
├── train.py                    # Training loop
├── evaluate.py                 # Evaluation + metrics
├── predict.py                  # Single image inference
├── data/
│   ├── download_isic.py        # Dataset download script
│   ├── dataset.py              # PyTorch Dataset class
│   └── augmentations.py        # Albumentations pipeline
├── models/
│   ├── classifier.py           # Model architecture
│   └── ensemble.py             # Ensemble wrapper
├── utils/
│   ├── metrics.py              # AUC, sensitivity, specificity
│   ├── tta.py                  # Test-time augmentation
│   └── visualize.py            # ROC curve, confusion matrix
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis
│   ├── training_curves.ipynb   # Loss/AUC over epochs
│   └── error_analysis.ipynb    # False positive/negative analysis
├── requirements.txt
└── README.md
```

---

## What I learned

- Handling severe class imbalance in binary classification
- Transfer learning: when to freeze backbone layers vs fine-tune end-to-end
- Test-time augmentation as a practical inference trick (cheap 2-3% AUC lift)
- The gap between AUC and clinical utility — a model with AUC 0.89 still misses 17% of melanomas

---

## Background

This research was conducted in collaboration with **Prof. M. Emre Celebi** at the University of Central Arkansas, whose lab specializes in dermoscopy image analysis. I worked on this as a high school student. The project competed at MOSTRATEC 2024 (Novo Hamburgo, Brazil) — the world's 2nd largest international science fair with 23 participating countries — and placed **3rd overall**.

---

## License

MIT
