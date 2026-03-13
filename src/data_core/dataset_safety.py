from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _sample_fingerprint(sample: Any) -> str:
    canonical = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sample_fingerprint_set(dataset_file: Path) -> set[str]:
    rows = json.loads(dataset_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"dataset must be a JSON list: {dataset_file}")
    return {_sample_fingerprint(row) for row in rows}


@dataclass(frozen=True)
class LeakageCheckResult:
    path_overlap_pairs: list[tuple[str, str]]
    same_content_pairs: list[tuple[str, str]]
    sample_overlap_pairs: list[tuple[str, str, int]]

    @property
    def has_issues(self) -> bool:
        return bool(self.path_overlap_pairs or self.same_content_pairs or self.sample_overlap_pairs)


def check_train_eval_leakage(
    *,
    train_file: Path | None,
    val_file: Path | None,
    test_file: Path | None,
    check_content_overlap: bool = True,
) -> LeakageCheckResult:
    named_paths = {
        "train": train_file,
        "val": val_file,
        "test": test_file,
    }
    existing: dict[str, Path] = {}
    for name, p in named_paths.items():
        if p is None:
            continue
        rp = p.expanduser().resolve()
        if not rp.exists():
            raise FileNotFoundError(f"{name}_file not found: {rp}")
        existing[name] = rp

    path_overlap_pairs: list[tuple[str, str]] = []
    same_content_pairs: list[tuple[str, str]] = []
    sample_overlap_pairs: list[tuple[str, str, int]] = []

    keys = list(existing.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            left = keys[i]
            right = keys[j]
            lp = existing[left]
            rp = existing[right]
            if lp == rp:
                path_overlap_pairs.append((left, right))

    if check_content_overlap:
        file_hashes = {name: sha256_file(path) for name, path in existing.items()}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                left = keys[i]
                right = keys[j]
                if file_hashes[left] == file_hashes[right]:
                    same_content_pairs.append((left, right))

        fingerprints = {name: sample_fingerprint_set(path) for name, path in existing.items()}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                left = keys[i]
                right = keys[j]
                overlap_cnt = len(fingerprints[left] & fingerprints[right])
                if overlap_cnt > 0:
                    sample_overlap_pairs.append((left, right, overlap_cnt))

    return LeakageCheckResult(
        path_overlap_pairs=path_overlap_pairs,
        same_content_pairs=same_content_pairs,
        sample_overlap_pairs=sample_overlap_pairs,
    )


def enforce_train_eval_no_leakage(
    *,
    train_file: Path | None,
    val_file: Path | None,
    test_file: Path | None,
    strict: bool = True,
    check_content_overlap: bool = True,
) -> LeakageCheckResult:
    result = check_train_eval_leakage(
        train_file=train_file,
        val_file=val_file,
        test_file=test_file,
        check_content_overlap=check_content_overlap,
    )
    if not result.has_issues:
        return result

    reasons: list[str] = []
    for left, right in result.path_overlap_pairs:
        reasons.append(f"path overlap: {left} and {right} point to the same file")
    for left, right in result.same_content_pairs:
        reasons.append(f"identical file hash: {left} and {right}")
    for left, right, cnt in result.sample_overlap_pairs:
        reasons.append(f"sample overlap: {left} and {right} share {cnt} sample(s)")
    message = "Potential dataset leakage detected. " + "; ".join(reasons)

    if strict:
        raise ValueError(message)
    print(f"[warn] {message}")
    return result
