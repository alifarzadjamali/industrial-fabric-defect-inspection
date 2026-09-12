"""AITEX AFID download and file-discovery helpers."""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import py7zr
from PIL import Image

AITEX_ARCHIVES = {
    "Defect_images.7z": "https://www.aitex.es/wp-content/uploads/2019/07/Defect_images.7z",
    "NODefect_images.7z": "https://www.aitex.es/wp-content/uploads/2019/07/NODefect_images.7z",
    "Mask_images.7z": "https://www.aitex.es/wp-content/uploads/2019/07/Mask_images.7z",
}

DEFECT_NAMES = {
    "002": "broken_end",
    "006": "broken_yarn",
    "010": "broken_pick",
    "016": "weft_curling",
    "019": "fuzzyball",
    "022": "cut_selvage",
    "023": "crease",
    "025": "warp_ball",
    "027": "knots",
    "029": "contamination",
    "030": "nep",
    "036": "weft_crack",
}

_IMAGE_RE = re.compile(r"^(?P<number>\d+)_(?P<defect>\d+)_(?P<fabric>\d+)$", re.IGNORECASE)
_MASK_RE = re.compile(r"^(?P<source>.+)_mask(?P<index>\d*)$", re.IGNORECASE)


@dataclass(frozen=True)
class AitexRecord:
    """One authoritative original image and its zero or more masks."""

    image_id: str
    image_path: Path
    is_defective: bool
    defect_code: str
    defect_name: str
    fabric_code: str
    mask_paths: tuple[Path, ...]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_and_extract(raw_dir: Path, force: bool = False) -> dict[str, str]:
    """Download the three official archives and extract them into ``raw_dir``."""

    archive_dir = raw_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    checksums: dict[str, str] = {}
    for filename, url in AITEX_ARCHIVES.items():
        archive = archive_dir / filename
        if force or not archive.exists():
            print(f"Downloading {url}")
            request = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "fabric-inspection-research/0.1 (+https://github.com/farzadismyid/industrial-fabric-defect-inspection)"
                },
            )
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                archive.open("wb") as destination,
            ):
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
        checksums[filename] = sha256_file(archive)
        expected_dir = raw_dir / archive.stem
        if force or not expected_dir.exists():
            print(f"Extracting {archive.name}")
            with py7zr.SevenZipFile(archive, mode="r") as bundle:
                bundle.extractall(path=raw_dir)
    return checksums


def discover_records(raw_dir: Path) -> list[AitexRecord]:
    """Discover source images and associate all masks by canonical image stem."""

    pngs = sorted(raw_dir.rglob("*.png"))
    mask_map: dict[str, list[Path]] = {}
    images: list[Path] = []
    for path in pngs:
        mask_match = _MASK_RE.match(path.stem)
        if mask_match:
            mask_map.setdefault(mask_match.group("source").lower(), []).append(path)
        elif "archive" not in {part.lower() for part in path.parts}:
            images.append(path)

    records: list[AitexRecord] = []
    for path in images:
        match = _IMAGE_RE.match(path.stem)
        if not match:
            continue
        defect_code = match.group("defect").zfill(3)
        is_defective = defect_code != "000" or "nodefect" not in str(path).lower()
        if defect_code == "000":
            is_defective = False
        records.append(
            AitexRecord(
                image_id=path.stem,
                image_path=path.resolve(),
                is_defective=is_defective,
                defect_code=defect_code,
                defect_name=DEFECT_NAMES.get(
                    defect_code, "normal" if not is_defective else "unknown"
                ),
                fabric_code=match.group("fabric").zfill(2),
                mask_paths=tuple(sorted(mask_map.get(path.stem.lower(), []))),
            )
        )
    if not records:
        raise FileNotFoundError(f"No AITEX images found below {raw_dir.resolve()}")
    return records


def load_grayscale(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def load_union_mask(record: AitexRecord, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Load the union of every mask associated with a record."""

    if shape is None:
        with Image.open(record.image_path) as image:
            shape = (image.height, image.width)
    union = np.zeros(shape, dtype=bool)
    for path in record.mask_paths:
        mask = load_grayscale(path)
        if mask.shape != shape:
            raise ValueError(
                f"Mask dimension mismatch for {record.image_id}: {mask.shape} != {shape}"
            )
        union |= mask > 0
    return union
