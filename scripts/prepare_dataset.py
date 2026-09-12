"""Download, audit, and split AITEX before defining any patches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fabric_inspection.data.aitex import discover_records, download_and_extract
from fabric_inspection.data.audit import audit_dataset
from fabric_inspection.data.splitting import create_split_manifest, save_manifests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--audit-dir", type=Path, default=Path("results/audit"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checksums = {}
    if args.download or args.force_download:
        checksums = download_and_extract(args.raw_dir, force=args.force_download)
    records = discover_records(args.raw_dir)
    summary = audit_dataset(records, args.audit_dir)
    splits = create_split_manifest(records, seed=args.seed)
    split_path, patch_path = save_manifests(splits, args.processed_dir, args.patch_size)
    run_metadata = {
        "seed": args.seed,
        "patch_size": args.patch_size,
        "archive_sha256": checksums,
        "split_counts": splits["split"].value_counts().to_dict(),
        "source_images_are_unique_across_splits": bool(splits["image_id"].is_unique),
    }
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    (args.processed_dir / "preparation_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved {split_path} and {patch_path}")


if __name__ == "__main__":
    main()
