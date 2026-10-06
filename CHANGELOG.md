# Changelog

## Unreleased

- Avoided redundant copies when training or inference tiles already match the requested size.
- Reused inference buffers while weighting and normalizing reconstructed probability maps.
- Computed epoch metrics directly from logits to avoid an extra sigmoid tensor per batch.
- Streamed tiled inference batches and cached blend windows to reduce repeated allocations.
- Vectorized baseline component filtering and reduced training metric mask allocations.
- Precomputed patch metadata and made padding safe for one-pixel image dimensions.
- Consolidated checkpoint integrity checks on the shared streamed SHA-256 helper.
- Added contributor setup and validation guidance.
- Added reproducibility notes for dataset preparation and locked splits.
- Added common environment and installation troubleshooting tips.
