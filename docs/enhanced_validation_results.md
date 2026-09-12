# Enhanced Pipeline: Frozen Validation Results

The replacement final holdout remained unopened throughout implementation, training, checkpoint
selection, and threshold selection. Both models use the locked protocol in
`enhanced_evaluation_protocol.md` and the same unchanged 37-image validation split.

## Changes tested

The enhanced pipeline keeps the ResNet18 U-Net architecture so the comparison isolates the data
and inference strategy. It adds 128-pixel source crops resized to 256 pixels, 50% overlap with
weighted blending, stronger measured photometric and geometric augmentation, inverse-square-root
emphasis for small positive masks, extra sampling of hard-negative fabrics 01/04/06, and focal
BCE plus Dice loss.

## Frozen validation comparison

| Metric | Fresh reference | Enhanced |
|---|---:|---:|
| Global Dice | 0.466 | **0.793** |
| Global IoU | 0.303 | **0.657** |
| Pixel precision | 0.355 | **0.703** |
| Pixel recall | 0.676 | **0.909** |
| Macro defective-image Dice | 0.108 | **0.380** |
| Image accuracy | 0.730 | **0.838** |
| Image F1 | 0.667 | **0.813** |
| Image ROC-AUC | 0.661 | **0.793** |

The reference segmentation and image thresholds are 0.75 and 0.0041790. The enhanced thresholds
are 0.03 and 0.0000448227. The low enhanced probability threshold is expected from focal-loss
calibration and was selected only on validation. Checkpoint hashes and full threshold sweeps are
stored in the corresponding `results/evaluation/` directories.

These results justify opening the final holdout once. They are not final performance estimates.
