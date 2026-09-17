# Reproducibility notes

The experiment outputs in `results/` are versioned summaries and figures. The
dataset itself, downloaded archives, processed manifests, and checkpoints are
kept out of Git because they are large and can be recreated locally.

For a clean rerun:

1. Create the Python environment using the [README setup](../README.md#set-up-the-environment).
2. Download and prepare AITEX with `python scripts/prepare_dataset.py --download`.
3. Run the configuration-specific commands in [Reproduce the experiments](../README.md#reproduce-the-experiments).
4. Run the checks listed in [Contributing](../CONTRIBUTING.md#development-setup).

Splits are assigned at the source-image level. Keep the committed split lock
and configuration files unchanged when comparing results; changing either can
make a run incomparable with the published outputs.
