from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.data_core.dataset_safety import sample_fingerprint_set, sha256_file


@dataclass(frozen=True)
class SplitDatasetConfig:
    input_file: Path
    out_dir: Path
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42
    train_name: str = "train.json"
    val_name: str = "val.json"
    test_name: str = "test.json"
    metadata_name: str = "split_metadata.json"
    preserve_existing_splits: bool = False


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if train_ratio < 0 or val_ratio < 0 or test_ratio < 0:
        raise ValueError("train/val/test ratios must be non-negative")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"train/val/test ratios must sum to 1.0, got {total}")


def _load_rows(path: Path) -> list[Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"dataset must be a JSON list: {path}")
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_optional_rows(path: Path) -> list[Any]:
    if not path.exists():
        return []
    return _load_rows(path)


def _split_indices(
    n: int,
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)

    train_n = int(n * train_ratio)
    val_n = int(n * val_ratio)
    test_n = n - train_n - val_n

    train_idx = idx[:train_n]
    val_idx = idx[train_n:train_n + val_n]
    test_idx = idx[train_n + val_n:train_n + val_n + test_n]
    return train_idx, val_idx, test_idx


def _sample_fingerprint(sample: Any) -> str:
    canonical = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _split_indices_grouped_by_fingerprint(
    rows: list[Any],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Split indices while keeping duplicate samples (same fingerprint) in the same split."""
    groups: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        key = _sample_fingerprint(row)
        groups.setdefault(key, []).append(idx)

    grouped_indices = list(groups.values())
    rng = random.Random(seed)
    rng.shuffle(grouped_indices)

    n = len(rows)
    train_target = int(n * train_ratio)
    val_target = int(n * val_ratio)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []

    for grp in grouped_indices:
        if len(train_idx) < train_target:
            train_idx.extend(grp)
        elif len(val_idx) < val_target:
            val_idx.extend(grp)
        else:
            test_idx.extend(grp)
    return train_idx, val_idx, test_idx


def run_split_dataset(cfg: SplitDatasetConfig) -> dict[str, Any]:
    input_file = cfg.input_file.expanduser().resolve()
    out_dir = cfg.out_dir.expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"input_file not found: {input_file}")

    _validate_ratios(cfg.train_ratio, cfg.val_ratio, cfg.test_ratio)
    rows = _load_rows(input_file)

    train_idx, val_idx, test_idx = _split_indices_grouped_by_fingerprint(
        rows,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        seed=cfg.seed,
    )
    train_rows = [rows[i] for i in train_idx]
    val_rows = [rows[i] for i in val_idx]
    test_rows = [rows[i] for i in test_idx]

    train_file = (out_dir / cfg.train_name).resolve()
    val_file = (out_dir / cfg.val_name).resolve()
    test_file = (out_dir / cfg.test_name).resolve()
    metadata_file = (out_dir / cfg.metadata_name).resolve()

    mode = "replace"
    existing_train_rows: list[Any] = []
    existing_val_rows: list[Any] = []
    existing_test_rows: list[Any] = []
    existing_seen: set[str] = set()
    new_rows = rows
    new_source_rows = len(rows)

    if cfg.preserve_existing_splits:
        mode = "append"
        existing_train_rows = _load_optional_rows(train_file)
        existing_val_rows = _load_optional_rows(val_file)
        existing_test_rows = _load_optional_rows(test_file)

        for existing_row in existing_train_rows + existing_val_rows + existing_test_rows:
            existing_seen.add(_sample_fingerprint(existing_row))

        new_rows = [
            row for row in rows
            if _sample_fingerprint(row) not in existing_seen
        ]
        new_source_rows = len(new_rows)

        train_idx, val_idx, test_idx = _split_indices_grouped_by_fingerprint(
            new_rows,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            seed=cfg.seed,
        )
        train_rows = existing_train_rows + [new_rows[i] for i in train_idx]
        val_rows = existing_val_rows + [new_rows[i] for i in val_idx]
        test_rows = existing_test_rows + [new_rows[i] for i in test_idx]

    _write_json(train_file, train_rows)
    _write_json(val_file, val_rows)
    _write_json(test_file, test_rows)

    train_fp = sample_fingerprint_set(train_file)
    val_fp = sample_fingerprint_set(val_file)
    test_fp = sample_fingerprint_set(test_file)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": cfg.seed,
        "ratios": {
            "train": cfg.train_ratio,
            "val": cfg.val_ratio,
            "test": cfg.test_ratio,
        },
        "source": {
            "path": str(input_file),
            "num_samples": len(rows),
            "sha256": sha256_file(input_file),
        },
        "mode": mode,
        "new_rows_considered": new_source_rows,
        "splits": {
            "train": {
                "path": str(train_file),
                "num_samples": len(train_rows),
                "sha256": sha256_file(train_file),
                "sample_set_sha256": "",
            },
            "val": {
                "path": str(val_file),
                "num_samples": len(val_rows),
                "sha256": sha256_file(val_file),
                "sample_set_sha256": "",
            },
            "test": {
                "path": str(test_file),
                "num_samples": len(test_rows),
                "sha256": sha256_file(test_file),
                "sample_set_sha256": "",
            },
        },
        "overlap": {
            "train_val": len(train_fp & val_fp),
            "train_test": len(train_fp & test_fp),
            "val_test": len(val_fp & test_fp),
        },
    }

    if cfg.preserve_existing_splits:
        metadata["preserved_existing_splits"] = True
        metadata["existing_before_append"] = {
            "train": len(existing_train_rows),
            "val": len(existing_val_rows),
            "test": len(existing_test_rows),
        }
        metadata["appended_counts"] = {
            "train": len(train_rows) - len(existing_train_rows),
            "val": len(val_rows) - len(existing_val_rows),
            "test": len(test_rows) - len(existing_test_rows),
        }

    # Use stable set hash to identify split membership independent of row ordering.
    for split_name, fp_set in (("train", train_fp), ("val", val_fp), ("test", test_fp)):
        joined = "\n".join(sorted(fp_set))
        metadata["splits"][split_name]["sample_set_sha256"] = (
            hashlib.sha256(joined.encode("utf-8")).hexdigest()
        )

    _write_json(metadata_file, metadata)
    metadata["metadata_file"] = str(metadata_file)
    return metadata


def run_split_from_merged_config(
    config: dict[str, Any],
    *,
    input_file: Path | None = None,
    out_dir: Path | None = None,
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    seed: int | None = None,
    train_name: str | None = None,
    val_name: str | None = None,
    test_name: str | None = None,
    metadata_name: str | None = None,
    preserve_existing_splits: bool | None = None,
) -> dict[str, Any]:
    """Build :class:`SplitDatasetConfig` from merged config and optional CLI overrides."""
    section = (
        config.get("dataset_prepare", {}).get("split", {})
        if isinstance(config.get("dataset_prepare"), dict)
        else {}
    )
    input_file_raw = input_file or section.get("input_file")
    out_dir_raw = out_dir or section.get("out_dir")
    if not input_file_raw:
        raise ValueError("dataset_prepare.split.input_file is required (or pass --input-file).")
    if not out_dir_raw:
        raise ValueError("dataset_prepare.split.out_dir is required (or pass --out-dir).")

    cfg = SplitDatasetConfig(
        input_file=Path(input_file_raw),
        out_dir=Path(out_dir_raw),
        train_ratio=float(train_ratio if train_ratio is not None else section.get("train_ratio", SplitDatasetConfig.train_ratio)),
        val_ratio=float(val_ratio if val_ratio is not None else section.get("val_ratio", SplitDatasetConfig.val_ratio)),
        test_ratio=float(test_ratio if test_ratio is not None else section.get("test_ratio", SplitDatasetConfig.test_ratio)),
        seed=int(seed if seed is not None else section.get("seed", SplitDatasetConfig.seed)),
        train_name=str(train_name or section.get("train_name") or section.get("train_file", SplitDatasetConfig.train_name)),
        val_name=str(val_name or section.get("val_name") or section.get("val_file", SplitDatasetConfig.val_name)),
        test_name=str(test_name or section.get("test_name") or section.get("test_file", SplitDatasetConfig.test_name)),
        metadata_name=str(
            metadata_name
            or section.get("metadata_name")
            or section.get("metadata_file", SplitDatasetConfig.metadata_name)
        ),
        preserve_existing_splits=bool(
            preserve_existing_splits
            if preserve_existing_splits is not None
            else section.get(
                "preserve_existing_splits",
                SplitDatasetConfig.preserve_existing_splits,
            )
        ),
    )
    return run_split_dataset(cfg)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Split a JSON-list dataset into train/val/test.")
    parser.add_argument("--input-file", type=Path, required=True, help="Source JSON list dataset file.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for split files.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--train-name", type=str, default="train.json", help="Train file name.")
    parser.add_argument("--val-name", type=str, default="val.json", help="Validation file name.")
    parser.add_argument("--test-name", type=str, default="test.json", help="Test file name.")
    parser.add_argument("--metadata-name", type=str, default="split_metadata.json", help="Metadata file name.")
    parser.add_argument(
        "--preserve-existing-splits",
        action="store_true",
        help="仅对新增样本做切分并追加到现有 train/val/test，保留旧数据不覆盖。",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    cfg = SplitDatasetConfig(
        input_file=args.input_file,
        out_dir=args.out_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_name=args.train_name,
        val_name=args.val_name,
        test_name=args.test_name,
        metadata_name=args.metadata_name,
        preserve_existing_splits=args.preserve_existing_splits,
    )
    metadata = run_split_dataset(cfg)
    print(f"[split] train: {metadata['splits']['train']['num_samples']} -> {metadata['splits']['train']['path']}")
    print(f"[split] val  : {metadata['splits']['val']['num_samples']} -> {metadata['splits']['val']['path']}")
    print(f"[split] test : {metadata['splits']['test']['num_samples']} -> {metadata['splits']['test']['path']}")
    print(f"[split] meta : {cfg.out_dir / cfg.metadata_name}")


if __name__ == "__main__":
    main()
