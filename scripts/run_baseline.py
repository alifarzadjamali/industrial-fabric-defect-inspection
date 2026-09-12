"""Tune the classical baseline on validation data and evaluate frozen settings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from fabric_inspection.baseline.evaluate import run_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    print(json.dumps(run_baseline(config), indent=2))


if __name__ == "__main__":
    main()
