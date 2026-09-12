# AITEX dataset audit

Audit date: 12 September 2026  
Preparation seed: 42  
Patch size: 256 × 256

## Source snapshot

The authoritative AITEX archives contained 247 readable original images: 106
defective and 141 normal. This differs slightly from the 245-image total reported
in the 2019 paper and on the dataset page; the repository reports the downloaded
snapshot rather than silently removing files.

| Archive | SHA-256 |
| --- | --- |
| `Defect_images.7z` | `2153ef691020ce0c6a0acaee2d1de76c75f0784a04b931706de7d5e1df7a1b63` |
| `NODefect_images.7z` | `4409ca615f069e0b92eaf1bd2058c8a37d9cb983f3aad205e7d69f30816f5664` |
| `Mask_images.7z` | `2d64486dbb5a2e1aa462b43c0f19e614af5aea6607c11d1a928e4ad9feb558d5` |

There are 107 mask files. Some images have multiple masks, which the pipeline
unions without altering the source annotations. Defective image `0100_025_08`
has no source mask and is retained for image-level classification but must be
excluded from supervised segmentation metrics and training. No corrupt files,
mask dimension mismatches, exact image duplicates, or perceptual-hash pairs at
Hamming distance ≤ 2 were found.

Of the images, 246 are 4096 × 256 pixels and one is 3796 × 256 pixels. Defective
pixels represent 0.1891% of all source pixels. Among defective images, annotated
area ranges from 0% (the missing-mask record) to 22.2682%, with a median of
0.00896%. This confirms extreme foreground imbalance and supports the planned
combined BCE and Dice loss.

## Defect distribution

| Defect | Images | Defect | Images |
| --- | ---: | --- | ---: |
| Broken end | 9 | Broken yarn | 8 |
| Broken pick | 10 | Weft curling | 3 |
| Fuzzyball | 39 | Cut selvage | 9 |
| Crease | 5 | Warp ball | 6 |
| Knots | 1 | Contamination | 1 |
| Nep | 14 | Weft crack | 1 |

The severe category imbalance is why the initial task remains binary segmentation
rather than multiclass classification.

## Leakage-safe split

Original images are assigned before patch coordinates are generated. Every patch
inherits its source image's assignment; the pipeline asserts that each source ID
occurs in exactly one split.

| Split | Original images | Normal | Defective | 256 × 256 patches |
| --- | ---: | ---: | ---: | ---: |
| Train | 172 | 98 | 74 | 2,751 |
| Validation | 37 | 21 | 16 | 592 |
| Test | 38 | 22 | 16 | 608 |

The one non-standard-width image produces a 212 × 256 edge patch, retained so no
source pixels are discarded; model input code will pad it when required. The
split is stratified by fabric structure and binary defect status where group
counts permit. Validation data is reserved for threshold and model selection;
the test split is held out from all tuning.

Detailed local outputs are generated in `results/audit/` and `data/processed/`.
They include the image inventory, integrity report, patch coordinates, split
manifest, and small/large-defect overlay figure.
