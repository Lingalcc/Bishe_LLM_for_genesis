"""API-based dataset generator for Franka robot tool-call fine-tuning.

Sends action definitions to a high-level LLM API and collects diverse
natural-language instruction → action-JSON pairs at varying complexity
levels. Outputs Alpaca and ShareGPT formats compatible with LLaMA Factory.
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
import hashlib
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.data_core.api_client import call_chat_api, extract_json_array, resolve_api_key
from src.data_core.format_utils import (
    SYSTEM_PROMPT_FOR_DATASET,
    to_alpaca_format,
    to_sharegpt_format,
    validate_sample,
)
from src.utils.secrets import redact_text

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[dict[str, Any]], None]

# ---------------------------------------------------------------------------
# Action registry — every supported robot action with schema & value ranges
# ---------------------------------------------------------------------------

ACTION_SCHEMAS: dict[str, dict[str, Any]] = {
    "wait": {
        "description": "暂停模拟，等待一段时间",
        "params": {},
        "example": {"action": "wait"},
    },
    "get_state": {
        "description": "查询当前机器人状态（关节位置、速度、末端位姿）",
        "params": {},
        "example": {"action": "get_state"},
    },
    "open_gripper": {
        "description": "打开机械臂夹爪",
        "params": {
            "position": {
                "type": "float", "range": [0.0, 0.04], "default": 0.04,
                "desc": "夹爪开合宽度(m)，0.04为完全打开",
            }
        },
        "example": {"action": "open_gripper", "position": 0.04},
    },
    "close_gripper": {
        "description": "关闭机械臂夹爪以抓取物体",
        "params": {
            "position": {
                "type": "float", "range": [0.0, 0.04], "default": 0.0,
                "desc": "夹爪开合宽度(m)，0.0为完全关闭",
            }
        },
        "example": {"action": "close_gripper", "position": 0.0},
    },
    "set_qpos": {
        "description": "直接设置机械臂全部9个关节的角度（7个臂关节+2个夹爪关节）",
        "params": {
            "qpos": {
                "type": "list[float]", "length": 9,
                "arm_range": [-2.8, 2.8], "gripper_range": [0.0, 0.04],
                "desc": "9维关节角度列表，前7个为臂关节(rad)，后2个为夹爪(m)",
            }
        },
        "example": {
            "action": "set_qpos",
            "qpos": [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.7, 0.04, 0.04],
        },
    },
    "set_dofs_position": {
        "description": "设置指定自由度的目标位置",
        "params": {
            "values": {"type": "list[float]", "desc": "目标位置值列表(rad或m)"},
            "dofs_idx_local": {
                "type": "list[int]", "optional": True,
                "desc": "自由度索引列表，0-6为臂关节，7-8为夹爪",
            },
        },
        "example": {
            "action": "set_dofs_position",
            "values": [0.0, -0.5, 0.0, -1.5],
            "dofs_idx_local": [0, 1, 2, 3],
        },
    },
    "control_dofs_position": {
        "description": "位置控制模式：控制指定自由度到目标位置",
        "params": {
            "values": {"type": "list[float]", "desc": "目标位置值列表(rad或m)"},
            "dofs_idx_local": {
                "type": "list[int]", "optional": True,
                "desc": "自由度索引列表，0-6为臂关节，7-8为夹爪",
            },
        },
        "example": {
            "action": "control_dofs_position",
            "values": [-0.5, 0.3],
            "dofs_idx_local": [1, 3],
        },
    },
    "control_dofs_velocity": {
        "description": "速度控制模式：控制指定自由度的运动速度",
        "params": {
            "values": {
                "type": "list[float]", "range": [-1.0, 1.0],
                "desc": "目标速度值列表(rad/s)",
            },
            "dofs_idx_local": {
                "type": "list[int]", "optional": True,
                "desc": "自由度索引列表",
            },
        },
        "example": {
            "action": "control_dofs_velocity",
            "values": [0.1, -0.2],
            "dofs_idx_local": [0, 1],
        },
    },
    "control_dofs_force": {
        "description": "力控制模式：对指定自由度施加力/力矩",
        "params": {
            "values": {
                "type": "list[float]", "range": [-10.0, 10.0],
                "desc": "目标力值列表(N或Nm)",
            },
            "dofs_idx_local": {
                "type": "list[int]", "optional": True,
                "desc": "自由度索引列表",
            },
        },
        "example": {
            "action": "control_dofs_force",
            "values": [0.5, -0.3],
            "dofs_idx_local": [2, 4],
        },
    },
    "move_ee": {
        "description": "将末端执行器移动到指定的笛卡尔空间位置和姿态（通过逆运动学）",
        "params": {
            "pos": {
                "type": "list[float]", "length": 3,
                "range": {"x": [0.2, 0.8], "y": [-0.5, 0.5], "z": [0.02, 0.6]},
                "desc": "目标位置 [x, y, z] (m)",
            },
            "quat": {
                "type": "list[float]", "length": 4,
                "desc": "目标姿态四元数 [w, x, y, z]，常用: [0,1,0,0]竖直向下, [1,0,0,0]默认",
            },
        },
        "example": {"action": "move_ee", "pos": [0.65, 0.0, 0.25], "quat": [0, 1, 0, 0]},
    },
    "reset_scene": {
        "description": "重置仿真场景到初始状态",
        "params": {},
        "example": {"action": "reset_scene"},
    },
}

# Composite task templates (multi-action sequences)
COMPOSITE_TEMPLATES: dict[str, dict[str, Any]] = {
    "grasp_and_lift": {
        "description": "抓取并举起物体的完整流程",
        "actions": ["open_gripper", "move_ee", "close_gripper", "wait", "move_ee"],
    },
    "place_object": {
        "description": "将物体放置到目标位置",
        "actions": ["move_ee", "open_gripper", "wait", "move_ee"],
    },
    "pick_and_place": {
        "description": "完整的抓取-搬运-放置流程",
        "actions": [
            "open_gripper", "move_ee", "close_gripper", "wait",
            "move_ee", "open_gripper", "wait",
        ],
    },
    "inspect_state": {
        "description": "查询状态后移动到目标位置",
        "actions": ["get_state", "move_ee"],
    },
    "reset_and_home": {
        "description": "重置场景并回到初始位姿",
        "actions": ["reset_scene", "set_qpos"],
    },
    "multi_joint_control": {
        "description": "分步控制不同关节组",
        "actions": ["control_dofs_position", "wait", "control_dofs_position"],
    },
}

# Difficulty levels
DIFFICULTY_LEVELS: dict[str, dict[str, Any]] = {
    "simple":  {"min_actions": 1, "max_actions": 1, "description": "单步简单动作"},
    "medium":  {"min_actions": 2, "max_actions": 3, "description": "2-3步组合动作"},
    "complex": {"min_actions": 4, "max_actions": 7, "description": "4-7步复杂操作序列"},
}


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_generation_system_prompt() -> str:
    """Construct the system prompt that teaches the LLM about available actions."""
    action_docs: list[str] = []
    for name, schema in ACTION_SCHEMAS.items():
        params_desc = ""
        if schema["params"]:
            lines = []
            for pname, pinfo in schema["params"].items():
                opt = " (可选)" if pinfo.get("optional") else ""
                lines.append(f"    - {pname}: {pinfo['desc']}{opt}")
            params_desc = "\n" + "\n".join(lines)
        action_docs.append(
            f"  {name}: {schema['description']}{params_desc}\n"
            f"    示例: {json.dumps(schema['example'], ensure_ascii=False)}"
        )

    return (
        "你是一个机器人操作数据集生成专家。你的任务是为 Franka Emika Panda 机械臂生成"
        "自然语言指令与对应 JSON 动作命令的配对数据。\n\n"
        "## 机械臂配置\n"
        "- 7个臂关节 (DOF 0-6) + 2个夹爪关节 (DOF 7-8)\n"
        "- 夹爪开合范围: 0.0 (关闭) ~ 0.04m (打开)\n"
        "- 工作空间: x∈[0.2,0.8], y∈[-0.5,0.5], z∈[0.02,0.6] (米)\n"
        "- 常用姿态四元数: [0,1,0,0] 竖直向下抓取, [1,0,0,0] 默认\n\n"
        "## 可用动作\n"
        + "\n".join(action_docs) + "\n\n"
        "## 输出格式要求\n"
        "每条数据必须严格按以下JSON格式输出（放在JSON数组中）：\n"
        '{"instruction": "自然语言指令", '
        '"output": "{\\"commands\\": [{\\"action\\": \\"...\\" ...}]}"}\n\n'
        "## 规则\n"
        "1. instruction 必须是自然、口语化的中文指令，可以包含具体数值也可以用描述性语言\n"
        "2. output 必须是合法的 JSON 字符串，包含 commands 数组\n"
        "3. 每个 command 必须有 action 字段，不允许 steps 字段\n"
        "4. move_ee 的 pos 必须为3元素列表，quat 必须为4元素列表\n"
        "5. set_qpos 的 qpos 必须为9元素列表\n"
        "6. 数值参数必须在合理范围内\n"
        "7. 生成的指令要多样化，覆盖不同场景和说法\n"
        "8. 只输出 JSON 数组，不要输出任何解释文字"
    )


def _build_generation_user_prompt(
    difficulty: str,
    batch_size: int,
    with_state_context: bool,
    seed_hint: int,
) -> str:
    level = DIFFICULTY_LEVELS[difficulty]
    state_ctx_note = ""
    if with_state_context:
        state_ctx_note = (
            "\n注意：其中部分指令需要包含场景状态上下文，格式为在instruction开头附加："
            '[STATE_CONTEXT]{"entities":[{"name":"cube","category":"object",'
            '"state":{"pos":[0.65,0.0,0.05]}}]}[/STATE_CONTEXT]\n用户指令: ...'
            "\n此时指令应针对场景中特定物体的位置做出相应动作。"
        )

    return (
        f"请生成 {batch_size} 条 Franka 机械臂的指令-动作配对数据。\n"
        f"难度级别: {difficulty} ({level['description']})\n"
        f"每条数据的动作序列长度: {level['min_actions']} ~ {level['max_actions']} 步\n"
        f"{state_ctx_note}\n"
        f"随机种子提示: {seed_hint}（用于保证多样性，每次生成不同内容）\n\n"
        f"请直接输出包含 {batch_size} 条数据的 JSON 数组，格式为:\n"
        '[{"instruction": "...", "output": "{\\"commands\\": [...]}"}, ...]\n'
        "确保每条 output 是合法的嵌套 JSON 字符串。\n"
        "同一批次内严禁重复样本（instruction 与 output 组合不能重复）。"
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateDatasetConfig:
    """All parameters for API-based dataset generation."""

    # Output paths
    out_dir: Path = Path("data_prepare")
    alpaca_file: str = "genesis_franka_toolcall_alpaca.json"
    sharegpt_file: str = "genesis_franka_toolcall_sharegpt.json"
    stats_file: str = "genesis_franka_toolcall_stats.json"
    progress_file: str = "genesis_franka_toolcall_progress.json"
    append_to_existing: bool = False

    # Generation parameters
    num_samples: int = 4000
    seed: int = 42
    batch_size: int = 20
    state_context_ratio: float = 0.7

    # Difficulty distribution (weights, should sum to 1.0)
    simple_ratio: float = 0.3
    medium_ratio: float = 0.4
    complex_ratio: float = 0.3

    # Parallelism
    max_workers: int = 4

    # API configuration
    api_base: str = "https://api.openai.com/v1"
    model: str = "gpt-4o"
    api_key: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.9
    max_tokens: int = 4096
    timeout: int = 120
    max_retries: int = 5
    sleep_seconds: float = 0.5
    dedup_max_rounds: int = 8


def _sample_fingerprint(sample: dict[str, Any]) -> str:
    canonical = json.dumps(sample, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _build_runtime_paths(cfg: GenerateDatasetConfig) -> dict[str, Path]:
    out_dir = cfg.out_dir
    return {
        "out_dir": out_dir,
        "alpaca": out_dir / cfg.alpaca_file,
        "sharegpt": out_dir / cfg.sharegpt_file,
        "stats": out_dir / cfg.stats_file,
        "progress": out_dir / cfg.progress_file,
    }


def _load_existing_alpaca_samples(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"alpaca dataset must be a JSON list: {path}")

    restored: list[dict[str, Any]] = []
    invalid_count = 0
    for row in payload:
        if not isinstance(row, dict):
            invalid_count += 1
            continue
        sample = {
            "instruction": row.get("instruction"),
            "output": row.get("output"),
        }
        if validate_sample(sample):
            restored.append(sample)
        else:
            invalid_count += 1

    if invalid_count > 0:
        logger.warning(
            "Existing alpaca file %s contained %d invalid row(s); ignored them.",
            path,
            invalid_count,
        )
    return restored


def _merge_unique_samples(*sample_groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in sample_groups:
        for sample in group:
            fp = _sample_fingerprint(sample)
            if fp in seen:
                continue
            seen.add(fp)
            merged.append(sample)
    return merged


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class DatasetGenerator:
    """Generates instruction→action dataset via LLM API calls."""

    def __init__(self, cfg: GenerateDatasetConfig) -> None:
        self.cfg = cfg
        self._rng = random.Random(cfg.seed)
        self._api_key = resolve_api_key(cfg.api_key, cfg.api_key_env)
        self._system_prompt = _build_generation_system_prompt()

    def _pick_difficulty(self) -> str:
        r = self._rng.random()
        if r < self.cfg.simple_ratio:
            return "simple"
        if r < self.cfg.simple_ratio + self.cfg.medium_ratio:
            return "medium"
        return "complex"

    def generate_batch(
        self, difficulty: str, batch_size: int, with_state: bool,
    ) -> list[dict[str, Any]]:
        """Call API once to generate *batch_size* samples."""
        seed_hint = self._rng.randint(0, 999999)
        user_prompt = _build_generation_user_prompt(
            difficulty=difficulty,
            batch_size=batch_size,
            with_state_context=with_state,
            seed_hint=seed_hint,
        )

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                raw = call_chat_api(
                    api_base=self.cfg.api_base,
                    api_key=self._api_key,
                    model=self.cfg.model,
                    system_prompt=self._system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout,
                )
                items = extract_json_array(raw)
                valid = [s for s in items if isinstance(s, dict) and validate_sample(s)]
                if valid:
                    return valid
                last_err = ValueError(
                    f"batch returned 0 valid samples out of {len(items)}"
                )
            except Exception as exc:
                last_err = exc
                logger.warning("API attempt %d failed: %s", attempt + 1, redact_text(str(exc)))

            time.sleep(min(30.0, self.cfg.sleep_seconds * (2 ** attempt)))

        logger.error(
            "Batch failed after %d retries: %s",
            self.cfg.max_retries,
            redact_text(str(last_err)) if last_err is not None else "unknown error",
        )
        return []

    def _prepare_batch_args(self, count: int) -> list[tuple[str, int, bool, int]]:
        """Pre-generate batch arguments (difficulty, size, with_state, seed_hint)."""
        args: list[tuple[str, int, bool, int]] = []
        remaining = count
        while remaining > 0:
            difficulty = self._pick_difficulty()
            batch = min(self.cfg.batch_size, remaining)
            with_state = self._rng.random() < self.cfg.state_context_ratio
            seed_hint = self._rng.randint(0, 999999)
            args.append((difficulty, batch, with_state, seed_hint))
            remaining -= batch
        return args

    def _generate_batch_from_args(
        self,
        difficulty: str,
        batch_size: int,
        with_state: bool,
        seed_hint: int,
        progress_callback: ProgressCallback | None = None,
        batch_label: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        """Thread-safe single batch generation (no shared mutable state)."""
        user_prompt = _build_generation_user_prompt(
            difficulty=difficulty,
            batch_size=batch_size,
            with_state_context=with_state,
            seed_hint=seed_hint,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "batch_started",
                    "difficulty": difficulty,
                    "requested_batch_size": batch_size,
                    "batch_label": batch_label,
                    "max_retries": self.cfg.max_retries,
                    "timeout": self.cfg.timeout,
                }
            )
        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            try:
                raw = call_chat_api(
                    api_base=self.cfg.api_base,
                    api_key=self._api_key,
                    model=self.cfg.model,
                    system_prompt=self._system_prompt,
                    user_prompt=user_prompt,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    timeout=self.cfg.timeout,
                )
                items = extract_json_array(raw)
                valid = [s for s in items if isinstance(s, dict) and validate_sample(s)]
                if valid:
                    return difficulty, valid
                last_err = ValueError(
                    f"batch returned 0 valid samples out of {len(items)}"
                )
            except Exception as exc:
                last_err = exc
                logger.warning("API attempt %d failed: %s", attempt + 1, redact_text(str(exc)))
                if progress_callback is not None:
                    progress_callback(
                        {
                            "event": "batch_retry",
                            "difficulty": difficulty,
                            "requested_batch_size": batch_size,
                            "batch_label": batch_label,
                            "attempt": attempt + 1,
                            "max_retries": self.cfg.max_retries,
                            "timeout": self.cfg.timeout,
                            "error": redact_text(str(exc)),
                        }
                    )
            time.sleep(min(30.0, self.cfg.sleep_seconds * (2 ** attempt)))

        logger.error(
            "Batch failed after %d retries: %s",
            self.cfg.max_retries,
            redact_text(str(last_err)) if last_err is not None else "unknown error",
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "event": "batch_failed",
                    "difficulty": difficulty,
                    "requested_batch_size": batch_size,
                    "batch_label": batch_label,
                    "max_retries": self.cfg.max_retries,
                    "timeout": self.cfg.timeout,
                    "error": redact_text(str(last_err)) if last_err is not None else "unknown error",
                }
            )
        return difficulty, []

    def _load_progress(self, progress_path: Path) -> dict[str, Any] | None:
        if not progress_path.exists():
            return None
        try:
            payload = json.loads(progress_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read progress file %s: %s", progress_path, exc)
            return None

        if not isinstance(payload, dict):
            logger.warning("Ignoring invalid progress payload in %s", progress_path)
            return None

        samples = payload.get("samples", [])
        stats = payload.get("stats", {})
        if not isinstance(samples, list) or not isinstance(stats, dict):
            logger.warning("Ignoring malformed progress payload in %s", progress_path)
            return None

        valid_samples = [s for s in samples if isinstance(s, dict) and validate_sample(s)]
        if len(valid_samples) != len(samples):
            logger.warning(
                "Progress file %s contained %d invalid samples; ignored them.",
                progress_path,
                len(samples) - len(valid_samples),
            )

        loaded = {
            "samples": valid_samples,
            "seen": {_sample_fingerprint(s) for s in valid_samples},
            "api_calls": int(stats.get("api_calls", 0)),
            "invalid_discarded": int(stats.get("invalid_discarded", 0)),
            "duplicate_discarded": int(stats.get("duplicate_discarded", 0)),
            "target_samples": int(stats.get("target_samples", len(valid_samples))),
            "requested_new_samples": int(stats.get("requested_new_samples", 0)),
            "existing_base_samples": int(stats.get("existing_base_samples", 0)),
            "generation_mode": str(stats.get("generation_mode", "replace")),
            "difficulty_distribution": {
                "simple": int(stats.get("difficulty_distribution", {}).get("simple", 0)),
                "medium": int(stats.get("difficulty_distribution", {}).get("medium", 0)),
                "complex": int(stats.get("difficulty_distribution", {}).get("complex", 0)),
            },
            "dedup_rounds": int(stats.get("dedup_rounds", 0)),
        }
        logger.info(
            "Loaded progress from %s: %d sample(s), %d api calls",
            progress_path,
            len(valid_samples),
            loaded["api_calls"],
        )
        return loaded

    def _persist_outputs(
        self,
        *,
        runtime_paths: dict[str, Path],
        collected: list[dict[str, Any]],
        stats: dict[str, Any],
        persist_count: int,
    ) -> None:
        samples = collected[:persist_count]
        shuffled_samples = deepcopy(samples)
        random.Random(self.cfg.seed).shuffle(shuffled_samples)

        _atomic_write_json(runtime_paths["alpaca"], to_alpaca_format(shuffled_samples))
        _atomic_write_json(runtime_paths["sharegpt"], to_sharegpt_format(shuffled_samples))
        _atomic_write_json(runtime_paths["stats"], stats)
        _atomic_write_json(
            runtime_paths["progress"],
            {
                "samples": samples,
                "stats": stats,
            },
        )

    def _build_stats(
        self,
        *,
        total_samples: int,
        target: int,
        requested_new_samples: int,
        existing_base_samples: int,
        total_api_calls: int,
        total_invalid: int,
        duplicate_discarded: int,
        round_idx: int,
        difficulty_counts: dict[str, int],
        workers: int,
        resumed_from_progress: bool,
        progress_path: Path,
    ) -> dict[str, Any]:
        return {
            "total_samples": total_samples,
            "target_samples": target,
            "requested_new_samples": requested_new_samples,
            "existing_base_samples": existing_base_samples,
            "generation_mode": "append" if self.cfg.append_to_existing else "replace",
            "api_calls": total_api_calls,
            "invalid_discarded": total_invalid,
            "duplicate_discarded": duplicate_discarded,
            "dedup_rounds": round_idx,
            "target_reached": total_samples >= target,
            "difficulty_distribution": difficulty_counts,
            "state_context_ratio": self.cfg.state_context_ratio,
            "max_workers": workers,
            "model": self.cfg.model,
            "seed": self.cfg.seed,
            "resumed_from_progress": resumed_from_progress,
            "progress_file": str(progress_path),
        }

    def generate_all(
        self,
        progress_callback: ProgressCallback | None = None,
        runtime_paths: dict[str, Path] | None = None,
    ) -> dict[str, Any]:
        """Run parallel generation with deduplication, refilling until target or max rounds."""
        all_samples: list[dict[str, Any]] = []
        total_api_calls = 0
        total_invalid = 0
        difficulty_counts = {"simple": 0, "medium": 0, "complex": 0}
        duplicate_discarded = 0
        target = self.cfg.num_samples
        workers = max(1, self.cfg.max_workers)
        max_rounds = max(1, int(self.cfg.dedup_max_rounds))
        runtime_paths = runtime_paths or _build_runtime_paths(self.cfg)

        logger.info(
            "Starting generation: target=%d samples, max_workers=%d",
            target, workers,
        )

        lock = threading.Lock()
        collected: list[dict[str, Any]] = []
        seen: set[str] = set()
        stats_invalid = 0
        stats_diff: dict[str, int] = {"simple": 0, "medium": 0, "complex": 0}
        resumed_from_progress = False
        requested_new_samples = self.cfg.num_samples
        existing_base_samples = 0

        if self.cfg.append_to_existing:
            existing_samples = _load_existing_alpaca_samples(runtime_paths["alpaca"])
            existing_base_samples = len(existing_samples)
            collected = list(existing_samples)
            seen = {_sample_fingerprint(sample) for sample in collected}
            target = existing_base_samples + requested_new_samples
            logger.info(
                "Append mode enabled: existing=%d, requested_new=%d, planned_total=%d",
                existing_base_samples,
                requested_new_samples,
                target,
            )

        loaded_progress = self._load_progress(runtime_paths["progress"])
        if loaded_progress is not None:
            progress_samples = list(loaded_progress["samples"])
            if self.cfg.append_to_existing:
                collected = _merge_unique_samples(collected, progress_samples)
                seen = {_sample_fingerprint(sample) for sample in collected}
                progress_target = int(loaded_progress.get("target_samples", 0))
                progress_mode = str(loaded_progress.get("generation_mode", ""))
                progress_requested_new = int(loaded_progress.get("requested_new_samples", 0))
                progress_existing_base = int(loaded_progress.get("existing_base_samples", 0))
                if progress_mode == "append" and progress_target >= len(collected):
                    target = progress_target
                    if progress_requested_new > 0:
                        requested_new_samples = progress_requested_new
                    if progress_existing_base > 0:
                        existing_base_samples = progress_existing_base
            else:
                collected = progress_samples
                seen = set(loaded_progress["seen"])
            total_api_calls = loaded_progress["api_calls"]
            stats_invalid = loaded_progress["invalid_discarded"]
            duplicate_discarded = loaded_progress["duplicate_discarded"]
            stats_diff = dict(loaded_progress["difficulty_distribution"])
            round_idx = loaded_progress["dedup_rounds"]
            resumed_from_progress = len(collected) > 0
            if len(collected) >= target:
                logger.info(
                    "Progress already satisfies target: %d/%d sample(s).",
                    len(collected),
                    target,
                )
                all_samples = collected[:target]
                self._rng.shuffle(all_samples)
                stats = self._build_stats(
                    total_samples=len(all_samples),
                    target=target,
                    requested_new_samples=requested_new_samples,
                    existing_base_samples=existing_base_samples,
                    total_api_calls=total_api_calls,
                    total_invalid=stats_invalid,
                    duplicate_discarded=duplicate_discarded,
                    round_idx=round_idx,
                    difficulty_counts=stats_diff,
                    workers=workers,
                    resumed_from_progress=resumed_from_progress,
                    progress_path=runtime_paths["progress"],
                )
                self._persist_outputs(
                    runtime_paths=runtime_paths,
                    collected=collected,
                    stats=stats,
                    persist_count=target,
                )
                return {"samples": all_samples, "stats": stats}
        else:
            round_idx = 0

        def _on_result(difficulty: str, samples: list[dict[str, Any]], req_size: int) -> None:
            nonlocal stats_invalid, duplicate_discarded
            with lock:
                if not samples:
                    stats_invalid += req_size
                for s in samples:
                    fp = _sample_fingerprint(s)
                    if fp in seen:
                        duplicate_discarded += 1
                        continue
                    seen.add(fp)
                    collected.append(s)
                    stats_diff[difficulty] += 1

        while len(collected) < target and round_idx < max_rounds:
            remaining = target - len(collected)
            batch_args = self._prepare_batch_args(remaining)
            total_api_calls += len(batch_args)
            round_idx += 1
            logger.info(
                "Generation round %d/%d: requesting %d samples via %d batch(es)",
                round_idx,
                max_rounds,
                remaining,
                len(batch_args),
            )

            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(
                        self._generate_batch_from_args,
                        diff,
                        bsz,
                        ws,
                        sh,
                        progress_callback,
                        f"r{round_idx}-b{idx}",
                    ): (diff, bsz)
                    for idx, (diff, bsz, ws, sh) in enumerate(batch_args, 1)
                }
                done_count = 0
                for future in as_completed(futures):
                    diff_req, bsz_req = futures[future]
                    done_count += 1
                    unique_before = len(collected)
                    duplicates_before = duplicate_discarded
                    invalid_before = stats_invalid
                    try:
                        diff_ret, samples = future.result()
                        _on_result(diff_ret, samples, bsz_req)
                    except Exception as exc:
                        logger.error("Batch raised: %s", exc)
                        diff_ret = diff_req
                        samples = []
                        _on_result(diff_req, [], bsz_req)
                    unique_after = len(collected)
                    duplicates_after = duplicate_discarded
                    invalid_after = stats_invalid
                    accepted_count = unique_after - unique_before
                    duplicate_count = duplicates_after - duplicates_before
                    invalid_count = invalid_after - invalid_before
                    logger.info(
                        "Progress: round %d batch %d/%d unique_samples %d/%d",
                        round_idx,
                        done_count,
                        len(batch_args),
                        len(collected),
                        target,
                    )
                    stats_snapshot = self._build_stats(
                        total_samples=min(len(collected), target),
                        target=target,
                        requested_new_samples=requested_new_samples,
                        existing_base_samples=existing_base_samples,
                        total_api_calls=total_api_calls,
                        total_invalid=stats_invalid,
                        duplicate_discarded=duplicate_discarded,
                        round_idx=round_idx,
                        difficulty_counts=dict(stats_diff),
                        workers=workers,
                        resumed_from_progress=resumed_from_progress,
                        progress_path=runtime_paths["progress"],
                    )
                    self._persist_outputs(
                        runtime_paths=runtime_paths,
                        collected=collected,
                        stats=stats_snapshot,
                        persist_count=target,
                    )
                    if progress_callback is not None:
                        progress_callback(
                            {
                                "event": "batch_completed",
                                "round_idx": round_idx,
                                "max_rounds": max_rounds,
                                "batch_idx": done_count,
                                "batch_total": len(batch_args),
                                "requested_batch_size": bsz_req,
                                "difficulty": diff_ret,
                                "returned_count": len(samples),
                                "accepted_count": accepted_count,
                                "duplicate_count": duplicate_count,
                                "invalid_count": invalid_count,
                                "unique_samples": unique_after,
                                "target_samples": target,
                                "remaining_samples": max(0, target - unique_after),
                                "api_calls": total_api_calls,
                            }
                        )

        if len(collected) < target:
            logger.warning(
                "Generation finished below target after %d rounds: got %d/%d unique samples.",
                round_idx,
                len(collected),
                target,
            )

        # Trim to target & shuffle
        all_samples = collected[:target]
        self._rng.shuffle(all_samples)
        total_invalid = stats_invalid
        difficulty_counts = stats_diff

        stats = self._build_stats(
            total_samples=len(all_samples),
            target=target,
            requested_new_samples=requested_new_samples,
            existing_base_samples=existing_base_samples,
            total_api_calls=total_api_calls,
            total_invalid=total_invalid,
            duplicate_discarded=duplicate_discarded,
            round_idx=round_idx,
            difficulty_counts=difficulty_counts,
            workers=workers,
            resumed_from_progress=resumed_from_progress,
            progress_path=runtime_paths["progress"],
        )
        self._persist_outputs(
            runtime_paths=runtime_paths,
            collected=collected,
            stats=stats,
            persist_count=target,
        )
        return {"samples": all_samples, "stats": stats}


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def run_generate_dataset(
    cfg: GenerateDatasetConfig,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Generate dataset and write Alpaca + ShareGPT + stats files."""
    runtime_paths = _build_runtime_paths(cfg)
    runtime_paths["out_dir"].mkdir(parents=True, exist_ok=True)

    generator = DatasetGenerator(cfg)
    result = generator.generate_all(
        progress_callback=progress_callback,
        runtime_paths=runtime_paths,
    )
    samples, stats = result["samples"], result["stats"]
    alpaca_path = runtime_paths["alpaca"]
    sharegpt_path = runtime_paths["sharegpt"]
    stats_path = runtime_paths["stats"]

    logger.info("Generated %d samples → %s, %s", len(samples), alpaca_path, sharegpt_path)
    return {
        "alpaca_path": str(alpaca_path),
        "sharegpt_path": str(sharegpt_path),
        "stats_path": str(stats_path),
        "progress_path": str(runtime_paths["progress"]),
        "total_samples": len(samples),
    }


def run_generate_from_merged_config(
    config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Build :class:`GenerateDatasetConfig` from a merged YAML dict and run."""
    section = (
        config.get("dataset_prepare", {}).get("generate", {})
        if isinstance(config.get("dataset_prepare"), dict)
        else {}
    )
    cfg = GenerateDatasetConfig(
        out_dir=Path(section.get("out_dir", GenerateDatasetConfig.out_dir)),
        alpaca_file=str(section.get("alpaca_file", GenerateDatasetConfig.alpaca_file)),
        sharegpt_file=str(section.get("sharegpt_file", GenerateDatasetConfig.sharegpt_file)),
        stats_file=str(section.get("stats_file", GenerateDatasetConfig.stats_file)),
        progress_file=str(section.get("progress_file", GenerateDatasetConfig.progress_file)),
        append_to_existing=bool(
            section.get("append_to_existing", GenerateDatasetConfig.append_to_existing)
        ),
        num_samples=int(section.get("num_samples", GenerateDatasetConfig.num_samples)),
        seed=int(section.get("seed", GenerateDatasetConfig.seed)),
        batch_size=int(section.get("batch_size", GenerateDatasetConfig.batch_size)),
        state_context_ratio=float(section.get("state_context_ratio", GenerateDatasetConfig.state_context_ratio)),
        simple_ratio=float(section.get("simple_ratio", GenerateDatasetConfig.simple_ratio)),
        medium_ratio=float(section.get("medium_ratio", GenerateDatasetConfig.medium_ratio)),
        complex_ratio=float(section.get("complex_ratio", GenerateDatasetConfig.complex_ratio)),
        max_workers=int(section.get("max_workers", GenerateDatasetConfig.max_workers)),
        api_base=str(section.get("api_base", GenerateDatasetConfig.api_base)),
        model=str(section.get("model", GenerateDatasetConfig.model)),
        api_key=str(section.get("api_key", GenerateDatasetConfig.api_key)),
        api_key_env=str(section.get("api_key_env", GenerateDatasetConfig.api_key_env)),
        temperature=float(section.get("temperature", GenerateDatasetConfig.temperature)),
        max_tokens=int(section.get("max_tokens", GenerateDatasetConfig.max_tokens)),
        timeout=int(section.get("timeout", GenerateDatasetConfig.timeout)),
        max_retries=int(section.get("max_retries", GenerateDatasetConfig.max_retries)),
        sleep_seconds=float(section.get("sleep_seconds", GenerateDatasetConfig.sleep_seconds)),
        dedup_max_rounds=int(section.get("dedup_max_rounds", GenerateDatasetConfig.dedup_max_rounds)),
    )
    return run_generate_dataset(cfg, progress_callback=progress_callback)
