"""Materialise the locked post-analysis development/final-holdout protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from fabric_inspection.data.splitting import create_patch_manifest


def create_lock(original: pd.DataFrame, seed: int, final_size: int) -> pd.DataFrame:
    """Select a fresh holdout only from the original training partition."""

    candidates = original[original["split"] == "train"].copy()
    key = candidates["fabric_code"].astype(str) + "_" + candidates["is_defective"].astype(str)
    counts = key.value_counts()
    eligible = candidates[key.map(counts) >= 3]
    eligible_key = (
        eligible["fabric_code"].astype(str) + "_" + eligible["is_defective"].astype(str)
    )
    _, final = train_test_split(
        eligible,
        test_size=final_size,
        random_state=seed,
        stratify=eligible_key,
    )
    final_ids = set(final["image_id"])
    lock = original[["image_id", "split"]].rename(columns={"split": "previous_split"})
    lock["split"] = "train"
    lock.loc[lock["previous_split"] == "validation", "split"] = "validation"
    lock.loc[lock["image_id"].isin(final_ids), "split"] = "final_test"
    return lock.sort_values("image_id", ignore_index=True)


def materialise(original: pd.DataFrame, lock: pd.DataFrame, output_dir: Path) -> None:
    if set(lock["image_id"]) != set(original["image_id"]):
        raise ValueError("Protocol lock IDs do not exactly match the source manifest")
    if lock["image_id"].duplicated().any():
        raise ValueError("Protocol lock contains duplicate image IDs")
    merged = original.drop(columns="split").merge(
        lock[["image_id", "split"]], on="image_id", validate="one_to_one"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "enhanced_splits.csv", index=False)
    for size in (128, 256):
        create_patch_manifest(merged, patch_size=size).to_csv(
            output_dir / f"enhanced_patches_{size}.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/processed/splits.csv"))
    parser.add_argument(
        "--lock", type=Path, default=Path("configs/splits/enhanced_protocol.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--create-lock", action="store_true")
    parser.add_argument("--seed", type=int, default=2409)
    parser.add_argument("--final-size", type=int, default=38)
    args = parser.parse_args()
    original = pd.read_csv(args.source)
    if args.create_lock:
        if args.lock.exists():
            raise FileExistsError(f"Refusing to replace existing protocol lock: {args.lock}")
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        create_lock(original, args.seed, args.final_size).to_csv(args.lock, index=False)
    lock = pd.read_csv(args.lock)
    materialise(original, lock, args.output_dir)
    print(lock.groupby(["split", "previous_split"]).size().to_string())


if __name__ == "__main__":
    main()
