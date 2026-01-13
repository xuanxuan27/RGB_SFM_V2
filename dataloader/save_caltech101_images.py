#!/usr/bin/env python3
"""
Export Caltech-101 images as resized (224x224) PNG files.

Example:
    python dataloader/save_caltech101_images.py \
        --root ./data \
        --output ./caltech101_images \
        --split train \
        --limit 2000 \
        --download
"""

from __future__ import annotations

import argparse
from pathlib import Path

from torchvision.datasets import Caltech101
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Caltech-101 and save resized 224x224 images."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("./data"),
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./caltech101_images"),
        help="Directory to store exported images.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="train",
        help="Use the same 80/20 split logic as Caltech101Dataset.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of images to export.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download Caltech-101 if it is missing.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Split ratio to emulate train/test partition.",
    )
    return parser.parse_args()


def build_dataset(root: Path, download: bool) -> Caltech101:
    dataset = Caltech101(
        root=str(root),
        download=download,
    )
    return dataset


def compute_indices(total: int, train_ratio: float, split: str) -> range:
    split_index = max(1, int(total * train_ratio))
    split_index = min(split_index, total - 1)
    if split == "train":
        return range(0, split_index)
    return range(split_index, total)


def export_images(dataset: Caltech101, indices: range, limit: int | None, output_dir: Path) -> None:
    resize = transforms.Resize((224, 224))
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(indices) if limit is None else min(limit, len(indices))
    for idx, dataset_idx in enumerate(indices[:total]):
        img, label = dataset[dataset_idx]
        img = resize(img)
        class_name = dataset.categories[label]
        filename = f"{idx:05d}_{label}_{class_name}.png"
        save_path = output_dir / filename
        img.save(save_path)
        if (idx + 1) % 500 == 0 or idx == total - 1:
            print(f"[{idx + 1}/{total}] Saved {save_path}")


def main() -> None:
    args = parse_args()
    dataset = build_dataset(args.root, args.download)
    indices = compute_indices(len(dataset), args.train_ratio, args.split)
    split_dir = args.output / args.split
    export_images(dataset, indices, args.limit, split_dir)


if __name__ == "__main__":
    main()

