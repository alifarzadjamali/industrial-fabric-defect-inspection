# Industrial Fabric Defect Inspection

Reproducible computer-vision research on the public AITEX Fabric Image Database.
The project prioritises pixel-level defect localisation, leakage-safe evaluation,
and lightweight methods suitable for a single consumer GPU.

> This independent project was inspired by a real freelance requirement posted
> on Upwork in September 2026 for a camera-based fabric defect detection system
> intended for industrial textile inspection.

This repository is independent work and was not commissioned by or developed for
the original client.

## Status

Phases 0 and 1 are implemented: authoritative dataset preparation and audit,
leakage-safe splitting, and a classical computer-vision baseline. Dataset files
and generated outputs are deliberately excluded from Git.

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.14.0+cu130 torchvision==0.29.0+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[dev]"
```

See [data/README.md](data/README.md) for data provenance and preparation.

## Dataset preparation

```powershell
python scripts/prepare_dataset.py --download
```

The command downloads the three official AITEX archives, validates every image
and mask, assigns original images to train/validation/test sets, and then creates
patch coordinates. It never splits patches independently. See the committed
[dataset audit](docs/dataset_audit.md) for findings and snapshot checksums.

## Classical baseline

```powershell
python scripts/run_baseline.py --config configs/baseline.yaml
```

The validation-tuned local texture baseline reaches a held-out test Dice of
**0.0129**, IoU of **0.0065**, and image-level F1 of **0.4000**. Its poor
localisation is an honest lower bound and demonstrates why a learned segmentation
model is necessary. See the [baseline report](docs/baseline_results.md) for the
full protocol, metrics, and qualitative output.

## U-Net training

```powershell
python scripts/train.py --config configs/unet.yaml
```

Phase 2 uses a U-Net with an ImageNet-pretrained ResNet-18 encoder, balanced
positive-patch sampling, realistic paired augmentation, combined weighted BCE
and Dice loss, CUDA mixed precision, checkpointing, learning-rate reduction, and
early stopping. The fixed-seed run selected epoch 4 at validation Dice
**0.5738**. The held-out test set remains untouched until Phase 3.
