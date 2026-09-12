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

Phase 0 is implemented: authoritative dataset preparation and audit plus
leakage-safe splitting. Dataset files and generated outputs are deliberately
excluded from Git.

## Quick start

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
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
