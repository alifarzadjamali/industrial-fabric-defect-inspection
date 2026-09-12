"""Generate Phase 4 error and controlled robustness analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fabric_inspection.evaluation.analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/analysis.yaml"))
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.config), indent=2))


if __name__ == "__main__":
    main()
