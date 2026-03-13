from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_FRANKA_MJCF = "xml/franka_emika_panda/panda.xml"
SUPPORTED_ROBOT_TYPES = {"mjcf", "urdf"}


class SimBootstrapError(RuntimeError):
    """Raised when simulation dependency or asset preflight fails."""


@dataclass(frozen=True)
class SimPreflightResult:
    genesis_repo_dir: Path | None
    asset_root_dir: Path | None
    robot_file: str
    resolved_robot_file: Path


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_existing_dir(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    else:
        path = path.resolve()
    if path.is_dir():
        return path
    return None


def _candidate_genesis_repo_dirs(explicit_genesis_repo: str | Path | None = None) -> list[Path]:
    repo_root = get_repo_root()
    candidates: list[Path] = []
    for raw in (
        explicit_genesis_repo,
        os.getenv("GENESIS_REPO_DIR"),
        repo_root / "Genesis",
        repo_root / "third_party" / "Genesis",
    ):
        path = _normalize_existing_dir(raw)
        if path is not None and path not in candidates:
            candidates.append(path)
    return candidates


def ensure_genesis_importable(*, genesis_repo: str | Path | None = None) -> Any:
    checked_repo_dirs = _candidate_genesis_repo_dirs(explicit_genesis_repo=genesis_repo)
    for repo_dir in checked_repo_dirs:
        repo_text = str(repo_dir)
        if repo_text not in sys.path:
            sys.path.insert(0, repo_text)

    try:
        return importlib.import_module("genesis")
    except Exception as exc:
        checked_text = ", ".join(str(p) for p in checked_repo_dirs) if checked_repo_dirs else "(none)"
        raise SimBootstrapError(
            "Genesis python package is not importable.\n"
            f"Checked local Genesis repo dirs: {checked_text}\n"
            "Try bootstrap first: `bash scripts/bootstrap_sim_assets.sh`\n"
            "Then set `GENESIS_REPO_DIR` if Genesis was installed in a custom path."
        ) from exc


def _candidate_asset_roots(
    *,
    explicit_asset_root: str | Path | None = None,
    genesis_repo: str | Path | None = None,
) -> list[Path]:
    repo_root = get_repo_root()
    roots: list[Path] = []

    for raw in (explicit_asset_root, os.getenv("GENESIS_ASSETS_ROOT")):
        path = _normalize_existing_dir(raw)
        if path is not None and path not in roots:
            roots.append(path)

    cwd_root = Path.cwd().resolve()
    if cwd_root not in roots:
        roots.append(cwd_root)
    if repo_root not in roots:
        roots.append(repo_root)

    for repo_dir in _candidate_genesis_repo_dirs(explicit_genesis_repo=genesis_repo):
        for root in (
            repo_dir,
            repo_dir / "assets",
            repo_dir / "genesis" / "assets",
        ):
            if root.is_dir() and root not in roots:
                roots.append(root)
    return roots


def resolve_robot_asset_path(
    file: str,
    *,
    robot_type: str = "mjcf",
    asset_root: str | Path | None = None,
    genesis_repo: str | Path | None = None,
) -> Path:
    robot_file = str(file).strip()
    if not robot_file:
        raise SimBootstrapError("robot asset file path is empty.")

    rt = str(robot_type).strip().lower()
    if rt not in SUPPORTED_ROBOT_TYPES:
        raise SimBootstrapError(
            f"unsupported robot_type: {robot_type}. Supported: {sorted(SUPPORTED_ROBOT_TYPES)}"
        )

    direct = Path(robot_file).expanduser()
    if direct.is_absolute() and direct.is_file():
        return direct.resolve()

    checked_paths: list[Path] = []
    for root in _candidate_asset_roots(explicit_asset_root=asset_root, genesis_repo=genesis_repo):
        candidate = (root / robot_file).resolve()
        checked_paths.append(candidate)
        if candidate.is_file():
            return candidate

    checked_text = "\n".join(f"- {p}" for p in checked_paths)
    raise SimBootstrapError(
        "Simulation robot asset file was not found.\n"
        f"Requested file: {robot_file}\n"
        f"robot_type: {rt}\n"
        "Checked candidate paths:\n"
        f"{checked_text}\n"
        "Try bootstrap first: `bash scripts/bootstrap_sim_assets.sh`\n"
        "Then set one of:\n"
        "- `GENESIS_REPO_DIR=/abs/path/to/Genesis`\n"
        "- `GENESIS_ASSETS_ROOT=/abs/path/to/assets/root`"
    )


def preflight_sim_environment(
    *,
    robot_file: str = DEFAULT_FRANKA_MJCF,
    robot_type: str = "mjcf",
    genesis_repo: str | Path | None = None,
    asset_root: str | Path | None = None,
) -> SimPreflightResult:
    ensure_genesis_importable(genesis_repo=genesis_repo)
    resolved_robot = resolve_robot_asset_path(
        robot_file,
        robot_type=robot_type,
        asset_root=asset_root,
        genesis_repo=genesis_repo,
    )
    repo_dir = _normalize_existing_dir(genesis_repo) or _normalize_existing_dir(os.getenv("GENESIS_REPO_DIR"))
    assets_dir = _normalize_existing_dir(asset_root) or _normalize_existing_dir(os.getenv("GENESIS_ASSETS_ROOT"))
    return SimPreflightResult(
        genesis_repo_dir=repo_dir,
        asset_root_dir=assets_dir,
        robot_file=robot_file,
        resolved_robot_file=resolved_robot,
    )
