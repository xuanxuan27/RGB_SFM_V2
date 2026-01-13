#!/usr/bin/env python3
"""
Utility script for exporting CIFAR-10 images to disk.

Example:
    python dataloader/save_cifar10_images.py \
        --root ./data \
        --split train \
        --output ./cifar10_images/train \
        --limit 5000
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple

from torchvision.datasets import CIFAR10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read CIFAR-10 and save each sample as an image file."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data"),
        help="Dataset root directory (will also be used for downloads).",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Which split to export.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./cifar10_images"),
        help="Directory where images will be saved.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of images to export.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download CIFAR-10 to the root directory if missing.",
    )
    return parser.parse_args()


def build_dataset(root: Path, split: str, download: bool) -> CIFAR10:
    train = split == "train"
    dataset = CIFAR10(
        root=str(root),
        train=train,
        download=download,
        transform=None,
        target_transform=None,
    )
    return dataset


def export_images(
    dataset: CIFAR10,
    output_dir: Path,
    limit: int | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(dataset) if limit is None else min(limit, len(dataset))
    class_names = dataset.classes

    for idx in range(total):
        img, label = dataset[idx]
        class_name = class_names[label]
        filename = f"{idx:05d}_{label}_{class_name}.png"
        save_path = output_dir / filename
        img.save(save_path)
        if (idx + 1) % 1000 == 0 or idx == total - 1:
            print(f"[{idx + 1}/{total}] Saved {save_path}")


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.root, args.split, args.download)
    export_target = args.output
    if args.split == "train":
        export_target = export_target / "train"
    else:
        export_target = export_target / "test"

    export_images(dataset, export_target, args.limit)


if __name__ == "__main__":
    main()

