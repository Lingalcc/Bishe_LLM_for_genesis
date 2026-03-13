"""Training metrics collector for LLaMA Factory fine-tuning.

Provides:
  - GPU VRAM monitoring during training (background thread polling nvidia-smi)
  - Loss curve extraction from LLaMA Factory trainer_state.json
  - Unified report generation for fine-tuning experiments
"""
from __future__ import annotations

import json
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GPU VRAM Monitor (background polling via nvidia-smi)
# ---------------------------------------------------------------------------

@dataclass
class VRAMSnapshot:
    timestamp: float
    gpu_index: int
    used_mb: float
    total_mb: float
    pid: int | None = None
    source: str = "gpu_total"


class GPUMonitor:
    """Background thread that polls nvidia-smi for VRAM usage."""

    def __init__(
        self,
        gpu_indices: list[int] | None = None,
        interval_sec: float = 0.5,
        target_pid: int | None = None,
    ):
        # `None` means monitor all visible GPUs.
        self._gpu_indices = gpu_indices
        self._interval = interval_sec
        self._target_pid = target_pid
        self._snapshots: list[VRAMSnapshot] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def set_target_pid(self, pid: int) -> None:
        self._target_pid = pid

    def _query_gpu_info(self) -> dict[str, tuple[int, float]]:
        """Return gpu_uuid -> (gpu_index, total_vram_mb)."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,uuid,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return {}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {}

        info: dict[str, tuple[int, float]] = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                gpu_index = int(parts[0])
                gpu_uuid = parts[1]
                total_mb = float(parts[2])
                info[gpu_uuid] = (gpu_index, total_mb)
            except (ValueError, IndexError):
                continue
        return info

    def _poll_compute_apps_once(self) -> list[VRAMSnapshot]:
        """Query process-level VRAM via nvidia-smi compute-apps table."""
        gpu_info = self._query_gpu_info()
        if not gpu_info:
            return []

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        now = time.time()
        pid = self._target_pid
        usage_by_gpu: dict[int, float] = {}
        total_by_gpu: dict[int, float] = {}
        matched_pid = False

        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                app_pid = int(parts[0])
                gpu_uuid = parts[1]
                used_mb = float(parts[2])
            except (ValueError, IndexError):
                continue

            if pid is not None and app_pid != pid:
                continue
            if pid is not None:
                matched_pid = True

            gpu_meta = gpu_info.get(gpu_uuid)
            if gpu_meta is None:
                continue
            idx, total_mb = gpu_meta
            if self._gpu_indices is not None and idx not in self._gpu_indices:
                continue
            usage_by_gpu[idx] = usage_by_gpu.get(idx, 0.0) + used_mb
            total_by_gpu[idx] = total_mb

        # If we target a specific pid and it is not active yet, skip this sample.
        if pid is not None and not matched_pid:
            return []

        snaps: list[VRAMSnapshot] = []
        for idx in sorted(usage_by_gpu):
            snaps.append(
                VRAMSnapshot(
                    timestamp=now,
                    gpu_index=idx,
                    used_mb=usage_by_gpu[idx],
                    total_mb=total_by_gpu.get(idx, 0.0),
                    pid=pid,
                    source="process",
                )
            )
        return snaps

    def _poll_once(self) -> list[VRAMSnapshot]:
        """Query nvidia-smi for current VRAM usage."""
        # Prefer process-level accounting (more precise for benchmark comparisons).
        process_snaps = self._poll_compute_apps_once()
        if process_snaps:
            return process_snaps

        # Fallback to total GPU memory usage if process-level metrics are unavailable.
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode != 0:
                return []
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return []

        now = time.time()
        snaps: list[VRAMSnapshot] = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            try:
                idx = int(parts[0])
                if self._gpu_indices is None or idx in self._gpu_indices:
                    snaps.append(VRAMSnapshot(
                        timestamp=now, gpu_index=idx,
                        used_mb=float(parts[1]), total_mb=float(parts[2]),
                        pid=self._target_pid, source="gpu_total",
                    ))
            except (ValueError, IndexError):
                continue
        return snaps

    def _run(self) -> None:
        while not self._stop_event.is_set():
            snaps = self._poll_once()
            if snaps:
                with self._lock:
                    self._snapshots.extend(snaps)
            self._stop_event.wait(self._interval)

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    @property
    def snapshots(self) -> list[VRAMSnapshot]:
        with self._lock:
            return list(self._snapshots)

    def summary(self) -> dict[str, Any]:
        """Return peak/avg VRAM stats per GPU."""
        snaps = self.snapshots
        if not snaps:
            return {"peak_vram_mb": 0.0, "avg_vram_mb": 0.0, "num_snapshots": 0, "per_gpu": {}}

        gpu_data: dict[int, list[float]] = {}
        for s in snaps:
            gpu_data.setdefault(s.gpu_index, []).append(s.used_mb)

        per_gpu = {}
        all_peaks: list[float] = []
        all_avgs: list[float] = []
        all_peak_deltas: list[float] = []
        all_avg_deltas: list[float] = []
        for idx, values in sorted(gpu_data.items()):
            peak = max(values)
            avg = sum(values) / len(values)
            baseline = values[0]
            peak_delta = peak - baseline
            avg_delta = avg - baseline
            total = next((s.total_mb for s in snaps if s.gpu_index == idx), 0.0)
            sample_source = next((s.source for s in snaps if s.gpu_index == idx), "unknown")
            per_gpu[str(idx)] = {
                "peak_vram_mb": peak,
                "avg_vram_mb": round(avg, 1),
                "baseline_vram_mb": baseline,
                "peak_delta_vram_mb": round(peak_delta, 1),
                "avg_delta_vram_mb": round(avg_delta, 1),
                "total_vram_mb": total,
                "peak_utilization": round(peak / total * 100, 1) if total > 0 else 0.0,
                "num_samples": len(values),
                "sample_source": sample_source,
            }
            all_peaks.append(peak)
            all_avgs.append(avg)
            all_peak_deltas.append(peak_delta)
            all_avg_deltas.append(avg_delta)

        return {
            "peak_vram_mb": max(all_peaks),
            "avg_vram_mb": round(sum(all_avgs) / len(all_avgs), 1),
            "peak_delta_vram_mb": round(max(all_peak_deltas), 1),
            "avg_delta_vram_mb": round(sum(all_avg_deltas) / len(all_avg_deltas), 1),
            "target_pid": self._target_pid,
            "num_snapshots": len(snaps),
            "per_gpu": per_gpu,
        }


# ---------------------------------------------------------------------------
# Loss Curve Extraction
# ---------------------------------------------------------------------------

def parse_trainer_state(trainer_state_path: Path) -> dict[str, Any]:
    """Extract loss curve data from LLaMA Factory's trainer_state.json."""
    if not trainer_state_path.exists():
        logger.warning("trainer_state.json not found: %s", trainer_state_path)
        return {"steps": [], "losses": [], "epochs": [], "learning_rates": []}

    data = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    log_history = data.get("log_history", [])

    steps: list[int] = []
    losses: list[float] = []
    epochs: list[float] = []
    learning_rates: list[float] = []
    eval_steps: list[int] = []
    eval_losses: list[float] = []

    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue

        # Training loss entries
        if "loss" in entry:
            steps.append(int(step))
            losses.append(float(entry["loss"]))
            epochs.append(float(entry.get("epoch", 0.0)))
            learning_rates.append(float(entry.get("learning_rate", 0.0)))

        # Eval loss entries
        if "eval_loss" in entry:
            eval_steps.append(int(step))
            eval_losses.append(float(entry["eval_loss"]))

    total_steps = int(data.get("max_steps", data.get("global_step", 0)))
    total_time = 0.0
    # Try to compute total training time
    if log_history:
        first_ts = log_history[0].get("timestamp")
        last_ts = log_history[-1].get("timestamp")
        if first_ts and last_ts:
            total_time = float(last_ts) - float(first_ts)

    result: dict[str, Any] = {
        "total_steps": total_steps,
        "total_epochs": float(data.get("epoch", 0.0)),
        "total_time_sec": total_time,
        "train_loss": {
            "steps": steps,
            "losses": losses,
            "epochs": epochs,
            "learning_rates": learning_rates,
        },
    }

    if losses:
        result["final_loss"] = losses[-1]
        result["min_loss"] = min(losses)
        result["min_loss_step"] = steps[losses.index(min(losses))]

    if eval_steps:
        result["eval_loss"] = {"steps": eval_steps, "losses": eval_losses}
        result["final_eval_loss"] = eval_losses[-1]

    return result


def parse_training_log(log_text: str) -> dict[str, Any]:
    """Extract loss values from training stdout/stderr text (fallback)."""
    steps: list[int] = []
    losses: list[float] = []
    # Match patterns like: {'loss': 1.234, ... 'step': 100}
    for m in re.finditer(r"'loss':\s*([\d.]+).*?'step':\s*(\d+)", log_text):
        losses.append(float(m.group(1)))
        steps.append(int(m.group(2)))
    # Also match: Step 100/500 loss=1.234
    for m in re.finditer(r"[Ss]tep\s+(\d+).*?loss[=:]\s*([\d.]+)", log_text):
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))

    return {"steps": steps, "losses": losses}


def find_trainer_state(output_dir: str | Path) -> Path | None:
    """Search for trainer_state.json in common LLaMA Factory output locations."""
    output_dir = Path(output_dir)
    candidates = [
        output_dir / "trainer_state.json",
        output_dir / "checkpoint-*" / "trainer_state.json",
    ]
    # Direct path
    if candidates[0].exists():
        return candidates[0]
    # Glob for checkpoints
    for p in sorted(output_dir.glob("checkpoint-*/trainer_state.json")):
        return p  # Return the first (latest) one
    # Search deeper
    for p in sorted(output_dir.rglob("trainer_state.json")):
        return p
    return None


# ---------------------------------------------------------------------------
# Unified Training Report
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Collected metrics from a single fine-tuning run."""
    method: str = ""
    model: str = ""
    dataset: str = ""
    # Timing
    total_time_sec: float = 0.0
    total_steps: int = 0
    total_epochs: float = 0.0
    # Loss
    final_loss: float = 0.0
    min_loss: float = 0.0
    min_loss_step: int = 0
    loss_curve: dict[str, Any] = field(default_factory=dict)
    # VRAM
    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0
    peak_delta_vram_mb: float = 0.0
    avg_delta_vram_mb: float = 0.0
    vram_detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "model": self.model,
            "dataset": self.dataset,
            "total_time_sec": self.total_time_sec,
            "total_steps": self.total_steps,
            "total_epochs": self.total_epochs,
            "final_loss": self.final_loss,
            "min_loss": self.min_loss,
            "min_loss_step": self.min_loss_step,
            "peak_vram_mb": self.peak_vram_mb,
            "avg_vram_mb": self.avg_vram_mb,
            "peak_delta_vram_mb": self.peak_delta_vram_mb,
            "avg_delta_vram_mb": self.avg_delta_vram_mb,
            "loss_curve": self.loss_curve,
            "vram_detail": self.vram_detail,
        }
