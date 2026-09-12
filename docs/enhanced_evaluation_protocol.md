# Enhanced Evaluation Protocol

The original test split was used for Phase 4 diagnosis and is therefore no longer an
unseen test set. It is development data from this point forward.

Before implementing the improvements, `configs/splits/enhanced_protocol.csv` locked a
replacement 38-image final holdout selected from the original training split. Selection was
deterministic (seed 2409), stratified by fabric and defect status where strata had sufficient
support, and did not use mask size, model predictions, or performance.

The enhanced experiment follows these rules:

1. The original validation split remains the only threshold-selection and early-stopping set.
2. The original test split is folded into training.
3. A fresh reference pipeline and the enhanced pipeline are both trained from scratch without
   access to the replacement holdout.
4. Both checkpoints and both pairs of thresholds are frozen using validation only.
5. The replacement holdout is evaluated once, for a paired comparison.
6. No result-driven tuning follows that evaluation; further work requires another protocol.

The committed lock contains only image IDs and split provenance. Labels and paths remain in the
ignored local materialisation under `data/processed/`.
