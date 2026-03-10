#!/usr/bin/env python3
from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

try:
    import pynvml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    pynvml = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    torch = None


MB = 1024 * 1024


def estimate_tokens_from_text(text: str | None) -> int:
    """A lightweight token estimator based on string length."""
    if not text:
        return 0
    return len(text)


class _BaseGpuMonitor:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> int:
        raise NotImplementedError


class _NullGpuMonitor(_BaseGpuMonitor):
    def start(self) -> None:
        return None

    def stop(self) -> int:
        return 0


class _NvmlGpuMonitor(_BaseGpuMonitor):
    def __init__(self, device_index: int, sample_interval_sec: float) -> None:
        if pynvml is None:
            raise RuntimeError("pynvml is not available")
        self.device_index = device_index
        self.sample_interval_sec = max(0.01, sample_interval_sec)
        self._handle: Any | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._baseline_used = 0
        self._peak_used = 0
        self._initialized = False

    def _read_used_bytes(self) -> int:
        assert self._handle is not None
        info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        return int(info.used)

    def _sampling_loop(self) -> None:
        while not self._stop_event.wait(self.sample_interval_sec):
            used = self._read_used_bytes()
            if used > self._peak_used:
                self._peak_used = used

    def start(self) -> None:
        pynvml.nvmlInit()
        self._initialized = True
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
        self._baseline_used = self._read_used_bytes()
        self._peak_used = self._baseline_used
        self._thread = threading.Thread(target=self._sampling_loop, daemon=True)
        self._thread.start()

    def stop(self) -> int:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=max(0.05, self.sample_interval_sec * 5))

        if self._handle is not None:
            used = self._read_used_bytes()
            if used > self._peak_used:
                self._peak_used = used

        peak_increase = max(0, self._peak_used - self._baseline_used)
        if self._initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return peak_increase


class _TorchGpuMonitor(_BaseGpuMonitor):
    def __init__(self, device_index: int) -> None:
        if torch is None or not torch.cuda.is_available():
            raise RuntimeError("torch.cuda is not available")
        self.device_index = device_index
        self._baseline = 0

    def start(self) -> None:
        torch.cuda.synchronize(self.device_index)
        torch.cuda.reset_peak_memory_stats(self.device_index)
        self._baseline = int(torch.cuda.memory_allocated(self.device_index))

    def stop(self) -> int:
        torch.cuda.synchronize(self.device_index)
        peak_alloc = int(torch.cuda.max_memory_allocated(self.device_index))
        return max(0, peak_alloc - self._baseline)


def _select_gpu_monitor(device_index: int, sample_interval_sec: float) -> _BaseGpuMonitor:
    if pynvml is not None:
        try:
            return _NvmlGpuMonitor(device_index=device_index, sample_interval_sec=sample_interval_sec)
        except Exception:
            pass
    if torch is not None:
        try:
            return _TorchGpuMonitor(device_index=device_index)
        except Exception:
            pass
    return _NullGpuMonitor()


@dataclass
class PerformanceStats:
    peak_vram_mb: float
    latency_sec: float
    throughput_tps: float

    def to_dict(self) -> dict[str, float]:
        return {
            "peak_vram_mb": self.peak_vram_mb,
            "latency_sec": self.latency_sec,
            "throughput_tps": self.throughput_tps,
        }


class InferencePerformanceMonitor:
    def __init__(
        self,
        *,
        device_index: int = 0,
        sample_interval_sec: float = 0.02,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        input_text: str | None = None,
        output_text: str | None = None,
        token_estimator: Callable[[str | None], int] = estimate_tokens_from_text,
        return_metrics: bool = True,
    ) -> None:
        self._device_index = device_index
        self._sample_interval_sec = sample_interval_sec
        self._return_metrics = return_metrics
        self._gpu_monitor: _BaseGpuMonitor | None = None
        self._start_time: float | None = None
        self._end_time: float | None = None
        self._peak_vram_bytes = 0

        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._input_text = input_text
        self._output_text = output_text
        self._token_estimator = token_estimator

        self._stats: PerformanceStats | None = None

    def __enter__(self) -> "InferencePerformanceMonitor":
        self._gpu_monitor = _select_gpu_monitor(
            device_index=self._device_index, sample_interval_sec=self._sample_interval_sec
        )
        self._start_time = time.perf_counter()
        self._gpu_monitor.start()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        if self._gpu_monitor is None:
            raise RuntimeError("GPU monitor was not initialized")
        self._peak_vram_bytes = self._gpu_monitor.stop()
        self._end_time = time.perf_counter()
        self._stats = self._compute_stats()
        return False

    @property
    def metrics(self) -> dict[str, float]:
        if self._stats is None:
            raise RuntimeError("monitor has not finished yet")
        return self._stats.to_dict()

    def set_input_tokens(self, num_tokens: int) -> None:
        self._input_tokens = max(0, int(num_tokens))

    def set_output_tokens(self, num_tokens: int) -> None:
        self._output_tokens = max(0, int(num_tokens))

    def set_input_text(self, text: str | None) -> None:
        self._input_text = text

    def set_output_text(self, text: str | None) -> None:
        self._output_text = text

    def _resolved_input_tokens(self) -> int:
        if self._input_tokens is not None:
            return self._input_tokens
        return max(0, int(self._token_estimator(self._input_text)))

    def _resolved_output_tokens(self) -> int:
        if self._output_tokens is not None:
            return self._output_tokens
        return max(0, int(self._token_estimator(self._output_text)))

    def _compute_stats(self) -> PerformanceStats:
        if self._start_time is None or self._end_time is None:
            raise RuntimeError("cannot compute stats without start/end time")
        latency_sec = max(0.0, self._end_time - self._start_time)
        total_tokens = self._resolved_input_tokens() + self._resolved_output_tokens()
        throughput_tps = (total_tokens / latency_sec) if latency_sec > 0 else 0.0
        return PerformanceStats(
            peak_vram_mb=self._peak_vram_bytes / MB,
            latency_sec=latency_sec,
            throughput_tps=throughput_tps,
        )

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            monitor = InferencePerformanceMonitor(
                device_index=self._device_index,
                sample_interval_sec=self._sample_interval_sec,
                input_tokens=self._input_tokens,
                output_tokens=self._output_tokens,
                input_text=self._input_text,
                output_text=self._output_text,
                token_estimator=self._token_estimator,
                return_metrics=self._return_metrics,
            )

            dynamic_input = _infer_input_text(args, kwargs)
            if dynamic_input is not None and monitor._input_tokens is None:
                monitor.set_input_text(dynamic_input)

            with monitor:
                output = func(*args, **kwargs)
                if monitor._output_tokens is None and isinstance(output, str):
                    monitor.set_output_text(output)

            metrics = monitor.metrics
            _wrapped.last_performance = metrics
            if self._return_metrics:
                return output, metrics
            return output

        _wrapped.last_performance = None  # type: ignore[attr-defined]
        return _wrapped


def _infer_input_text(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    for key in ("input_text", "prompt", "instruction", "query"):
        value = kwargs.get(key)
        if isinstance(value, str):
            return value
    if args and isinstance(args[0], str):
        return args[0]
    return None


def monitor_inference_performance(
    func: Callable[..., Any] | None = None,
    *,
    device_index: int = 0,
    sample_interval_sec: float = 0.02,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    token_estimator: Callable[[str | None], int] = estimate_tokens_from_text,
    return_metrics: bool = True,
) -> Any:
    """
    Context manager + decorator for inference performance monitoring.

    Context manager mode:
        with monitor_inference_performance(input_text=prompt) as mon:
            out = generate(prompt)
            mon.set_output_text(out)
        print(mon.metrics)

    Decorator mode:
        @monitor_inference_performance(return_metrics=True)
        def generate(prompt: str) -> str:
            ...
        output, metrics = generate("hello")
    """

    monitor = InferencePerformanceMonitor(
        device_index=device_index,
        sample_interval_sec=sample_interval_sec,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_text=input_text,
        output_text=output_text,
        token_estimator=token_estimator,
        return_metrics=return_metrics,
    )

    if func is not None:
        return monitor(func)
    return monitor


def _demo_generation(prompt: str, *, max_new_tokens: int = 64) -> str:
    # Simulate model generation latency.
    time.sleep(0.35)
    return f"[generated x{max_new_tokens}] {prompt[::-1]}"


def demo() -> None:
    prompt = "请生成一个抓取杯子的动作序列。"

    print("=== Context Manager Demo ===")
    with monitor_inference_performance(input_text=prompt) as mon:
        output = _demo_generation(prompt, max_new_tokens=80)
        mon.set_output_text(output)
    print("output:", output)
    print("metrics:", mon.metrics)

    print("\n=== Decorator Demo ===")

    @monitor_inference_performance(return_metrics=True)
    def generate_with_monitor(local_prompt: str) -> str:
        return _demo_generation(local_prompt, max_new_tokens=40)

    output2, metrics2 = generate_with_monitor(prompt)
    print("output:", output2)
    print("metrics:", metrics2)


if __name__ == "__main__":
    demo()
