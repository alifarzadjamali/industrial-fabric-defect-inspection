# Enhanced Pipeline: Final Holdout Results

Both checkpoints, their SHA-256 hashes, and all decision thresholds were committed before this
evaluation. The locked 38-image replacement holdout was then opened once for both pipelines. No
model, loss, sampler, inference setting, or threshold was changed afterward.

## Paired final comparison

| Metric | Fresh reference | Enhanced | Absolute change |
|---|---:|---:|---:|
| Global Dice | 0.2589 | **0.5321** | **+0.2732** |
| Global IoU | 0.1487 | **0.3625** | **+0.2138** |
| Pixel precision | 0.1576 | **0.3820** | **+0.2244** |
| Pixel recall | 0.7246 | **0.8765** | **+0.1520** |
| Macro defective-image Dice | 0.1340 | **0.2497** | **+0.1156** |
| Image accuracy | 0.7105 | **0.7368** | **+0.0263** |
| Image precision | **0.7778** | 0.6875 | -0.0903 |
| Image recall | 0.4375 | **0.6875** | **+0.2500** |
| Image F1 | 0.5600 | **0.6875** | **+0.1275** |
| Image ROC-AUC | 0.7074 | **0.8778** | **+0.1705** |

The enhanced segmentation Dice is 2.06 times the matched reference, while false-negative images
fall from 9 to 5. The trade-off is three additional false-positive images (5 versus 2), reflected
in lower image precision. For industrial inspection this is the safer direction when missed
defects are more costly, but operating thresholds should ultimately be selected from explicit
line-level false-reject and escape-cost requirements.

The earlier Phase 3 Dice of 0.2225 and image F1 of 0.6286 came from a different, now-consumed test
split and are not used as the primary comparison. The fair result is the paired reference versus
enhanced comparison above.

## Interpretation

This is a substantial and clean improvement on AITEX, not evidence of industrial readiness. The
sample is small, acquisition conditions are narrow, tiny-defect localisation remains weak, and no
latency or production-line acceptance test has been completed. The next meaningful gains require
new camera-specific data, repeated-fabric group splits, explicit annotation of very small defects,
probability calibration, and validation against agreed false-negative limits before deployment.
