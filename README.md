# CrackSeg-Topology

**Topology-Aware Crack Segmentation via Multi-Source Transfer Learning for Physical Dimension Quantification**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)

## Overview

CrackSeg-Topology is a two-stage deep learning pipeline for automated concrete crack segmentation and physical dimension estimation (length in cm, width in cm). The key insight is that **topological connectivity at branching junctions**—not aggregate pixel overlap—is the dominant driver of physical measurement reliability.

The framework:
1. **Stage 1 (Pre-training):** Jointly pre-trains on three heterogeneous public crack datasets (CFD, Crack500, DeepCrack) to regularise mask topology and boundary continuity.
2. **Stage 2 (Fine-tuning):** Fine-tunes on SUT-Crack with calibrated physical annotations using 5-fold cross-validation.
3. **Measurement:** Extracts crack length via morphological skeletonisation and crack width via Euclidean Distance Transform sampling.

### Architecture

- **Backbone:** U-Net++ with configurable encoders (MobileNet-V2, EfficientNet-B4, ResNet-18, VGG-16)
- **SE Attention:** Squeeze-and-Excitation channel recalibration on encoder stages
- **ASPP:** Atrous Spatial Pyramid Pooling at the encoder bottleneck for multi-scale context

## Installation

```bash
git clone https://github.com/<Conson123>/TopoCrackSeg.git
cd TopoCrackSeg
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- PyTorch >= 1.12 with CUDA support
- NVIDIA GPU with >= 8 GB VRAM (training); CPU supported for inference
- See `requirements.txt` for full dependency list

## Dataset Preparation

Download the following datasets and organise them as shown:

| Dataset | Description | Link |
|---------|-------------|------|
| CFD | Low-contrast asphalt pavement cracks | [GitHub](https://github.com/cuilimeng/CrackForest-dataset) |
| Crack500 | Branching cracks on rough aggregate | [GitHub](https://github.com/fyangneil/pavement-crack-detection) |
| DeepCrack | High-contrast cracks on smooth concrete | [GitHub](https://github.com/yhlleo/DeepCrack) |
| SUT-Crack | Moderate-width cracks with physical annotations | [DOI](https://doi.org/10.1016/j.dib.2023.109642) |

### Expected directory structure

```
data/
  CFD/
    train/images/    train/masks/
    test/images/     test/masks/
  Crack500/
    train/images/    train/masks/
    test/images/     test/masks/
  DeepCrack/
    train/images/    train/masks/
    test/images/     test/masks/
  SUT-Crack/
    images/          labels/
    Crack Length.txt
```

The `Crack Length.txt` file is tab-separated with columns `filename` and `Crack Length (cm)`.

## Usage

### Stage 1: Multi-Source Pre-training

Pre-train on each source dataset individually or on a combined set:

```bash
# Pre-train on CFD
python main.py --mode pretrain --data_dir data/CFD --dataset_name CFD \
    --model UnetPlusPlus --encoder mobilenet_v2 --epochs 200 --batch_size 16

# Pre-train on Crack500
python main.py --mode pretrain --data_dir data/Crack500 --dataset_name Crack500 \
    --model UnetPlusPlus --encoder mobilenet_v2

# Pre-train on DeepCrack
python main.py --mode pretrain --data_dir data/DeepCrack --dataset_name DeepCrack \
    --model UnetPlusPlus --encoder mobilenet_v2
```

### Stage 2: Fine-tuning on SUT-Crack

```bash
# Fine-tune with pre-trained weights (5-fold CV)
python main.py --mode finetune --data_dir data/SUT-Crack \
    --load_weights run_results/pretrain_CFD_UnetPlusPlus_mobilenet_v2/pretrained_UnetPlusPlus_mobilenet_v2.pth \
    --model UnetPlusPlus --encoder mobilenet_v2 --use_aspp --attention se

# Baseline without pre-training
python main.py --mode finetune --data_dir data/SUT-Crack \
    --load_weights none --model UnetPlusPlus --encoder mobilenet_v2
```

### Architecture and Encoder Comparisons

```bash
# Compare architectures
for arch in Unet Linknet UnetPlusPlus MAnet; do
    python main.py --mode finetune --data_dir data/SUT-Crack \
        --model $arch --encoder mobilenet_v2
done

# Compare encoders
for enc in mobilenet_v2 resnet18 vgg16 efficientnet-b4; do
    python main.py --mode finetune --data_dir data/SUT-Crack \
        --model UnetPlusPlus --encoder $enc
done
```

### Analyse Results

```bash
python main.py --mode analyze
```

This generates comparison tables, bar charts, radar plots, and LaTeX tables from all experiments in `run_results/`.

## Outputs

Each experiment produces:
- `pretrained_*.pth` or `model_fold_*.pth` — saved model weights
- `*_log.csv` — per-epoch training/validation metrics
- `*_curves.png` — loss and metric convergence plots
- `predictions_log.csv` — per-image segmentation, length, and width metrics
- `predicted_masks_*/` — binary prediction masks
- `*_pred.png` — side-by-side input / ground-truth / prediction visualisations

## Project Structure

```
TopoCrackSeg/
├── LICENSE
├── README.md
├── requirements.txt
├── main.py                    # Entry point (CLI)
├── models/
│   ├── __init__.py
│   ├── network.py             # Model construction and weight loading
│   ├── attention.py           # CBAM, SE, ECA attention blocks
│   ├── aspp.py                # ASPP multi-scale context module
│   ├── losses.py              # Loss functions (Focal, Tversky, Lovasz, ...)
│   └── width_estimator.py     # EDT-based crack width estimation
├── data/
│   ├── __init__.py
│   ├── dataset.py             # PyTorch Dataset class
│   ├── transforms.py          # Albumentations augmentation pipelines
│   └── loader.py              # Data discovery and loading
├── scripts/
│   ├── postprocess.py         # Morphological and CRF post-processing
│   ├── metrics.py             # Skeleton length and width computation
│   ├── train.py               # Training and validation loops
│   ├── predict.py             # Inference and visualisation
│   └── analyze.py             # Cross-experiment comparison
└── results/                   # (Generated) Experiment outputs
```

## Citation

If you use this code, please cite:

```bibtex
@article{kou2026topology,
  title={Topology-Aware Crack Segmentation via Multi-Source Transfer Learning
         for Physical Dimension Quantification},
  author={Kou, Pinglang and Kang, Sen and Yuan, Zhengwu and Li, Huajin},
  journal={Computers \& Geosciences},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Contact

Pinglang Kou (Corresponding Author)
- Email: pinglangkou@163.com
- Affiliation: Chongqing University of Posts and Telecommunications, Chongqing 400065, China
