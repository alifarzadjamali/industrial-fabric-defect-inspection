"""Freeze validation thresholds or evaluate the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fabric_inspection.evaluation.model_evaluator import (
    evaluate_frozen_test,
    load_evaluation_config,
    select_and_freeze_thresholds,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation.yaml"))
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--select-thresholds", action="store_true")
    action.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    config = load_evaluation_config(args.config)
    result = (
        select_and_freeze_thresholds(config)
        if args.select_thresholds
        else evaluate_frozen_test(config)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
