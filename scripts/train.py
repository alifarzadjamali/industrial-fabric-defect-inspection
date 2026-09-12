"""Train the configured segmentation model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fabric_inspection.training.trainer import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/unet.yaml"))
    args = parser.parse_args()
    print(json.dumps(train(load_config(args.config), args.config), indent=2))


if __name__ == "__main__":
    main()
