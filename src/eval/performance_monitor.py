#!/usr/bin/env python3
from __future__ import annotations

import functools
import inspect
import json
import os
import re
import time
from contextlib import ContextDecorator
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - 可选依赖，缺失时退化为 0
    psutil = None

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - 可选依赖，缺失时退化为 CPU 模式
    torch = None


MB = 1024 * 1024
GB = 1024 * 1024 * 1024


def _to_mb(num_bytes: int) -> float:
    return float(num_bytes) / MB


def _to_gb(num_bytes: int) -> float:
    return float(num_bytes) / GB


def estimate_tokens_from_text(text: str | None) -> int:
    """
    默认 token 估算器（不依赖 tokenizer，便于在任何环境下运行）。

    说明：
    1. 对英文/数字连续串按“词块”计数；
    2. 对中文字符按单字计数；
    3. 对其它非空白符号（标点等）按单符号计数。

    该方法是“估算”，不是 tokenizer 的精确 token 数。
    如需精确统计，可在 tracker 中传入自定义 token_counter。
    """
    if not text:
        return 0
    parts = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\s]", text)
    return len(parts)


def _infer_prompt_text(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """尽量从常见参数名中推断 prompt/instruction。"""
    for key in ("prompt", "instruction", "query", "input_text"):
        value = kwargs.get(key)
        if isinstance(value, str):
            return value
    if args and isinstance(args[0], str):
        return args[0]
    return None


def _safe_to_text(obj: Any) -> str:
    """
    将模型输出尽量转为文本，用于 token 估算。
    对 dict/list 等结构化输出，转成紧凑 JSON 字符串。
    """
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    try:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    except Exception:
        return str(obj)


@dataclass
class ResourceMonitor:
    """
    资源监控器：负责采集显存与进程内存指标。

    满足需求点：
    1. 当前显存占用：
       - vram_allocated_mb / vram_allocated_gb
       - vram_reserved_mb / vram_reserved_gb
    2. 显存峰值：
       - vram_peak_allocated_gb
       - vram_peak_reserved_gb
       - vram_peak_gb（两者中的较大值，便于统一展示）

    额外信息：
    - process_rss_mb / process_rss_gb（来自 psutil）
    - device / cuda_available
    """

    device_index: int = 0
    synchronize_cuda: bool = True
    _pid: int = field(default_factory=os.getpid, init=False, repr=False)
    _start_snapshot: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _end_snapshot: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._cuda_available = bool(torch is not None and torch.cuda.is_available())
        self._device_count = int(torch.cuda.device_count()) if self._cuda_available else 0
        if self._cuda_available and (self.device_index < 0 or self.device_index >= self._device_count):
            raise ValueError(
                f"无效的 CUDA 设备索引: {self.device_index}，可用设备数: {self._device_count}"
            )
        self._process = psutil.Process(self._pid) if psutil is not None else None

    @property
    def cuda_available(self) -> bool:
        return self._cuda_available

    @property
    def device_name(self) -> str:
        if self._cuda_available:
            return f"cuda:{self.device_index}"
        return "cpu"

    def _maybe_sync_cuda(self) -> None:
        if self._cuda_available and self.synchronize_cuda and torch is not None:
            torch.cuda.synchronize(self.device_index)

    def _read_process_rss_bytes(self) -> int:
        if self._process is None:
            return 0
        try:
            return int(self._process.memory_info().rss)
        except Exception:
            return 0

    def snapshot(self) -> dict[str, Any]:
        """
        获取当前时刻资源快照。
        该方法可在任意时刻调用；通常用于 start/stop 前后采样。
        """
        self._maybe_sync_cuda()

        allocated = 0
        reserved = 0
        peak_allocated = 0
        peak_reserved = 0
        if self._cuda_available and torch is not None:
            allocated = int(torch.cuda.memory_allocated(self.device_index))
            reserved = int(torch.cuda.memory_reserved(self.device_index))
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device_index))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device_index))

        rss_bytes = self._read_process_rss_bytes()
        peak_bytes = max(peak_allocated, peak_reserved)

        return {
            "timestamp": time.time(),
            "device": self.device_name,
            "cuda_available": self._cuda_available,
            "psutil_available": bool(self._process is not None),
            "vram_allocated_mb": _to_mb(allocated),
            "vram_allocated_gb": _to_gb(allocated),
            "vram_reserved_mb": _to_mb(reserved),
            "vram_reserved_gb": _to_gb(reserved),
            "vram_peak_allocated_gb": _to_gb(peak_allocated),
            "vram_peak_reserved_gb": _to_gb(peak_reserved),
            "vram_peak_mb": _to_mb(peak_bytes),
            "vram_peak_gb": _to_gb(peak_bytes),
            "process_rss_mb": _to_mb(rss_bytes),
            "process_rss_gb": _to_gb(rss_bytes),
        }

    def start(self, reset_peak: bool = True) -> dict[str, Any]:
        """
        开始监控，记录起始快照。
        reset_peak=True 时会重置 CUDA 峰值统计，保证当前推理区间的峰值可比。
        """
        if self._cuda_available and torch is not None and reset_peak:
            self._maybe_sync_cuda()
            torch.cuda.reset_peak_memory_stats(self.device_index)
        self._start_snapshot = self.snapshot()
        return self._start_snapshot

    def stop(self) -> dict[str, Any]:
        """结束监控，记录结束快照。"""
        self._end_snapshot = self.snapshot()
        return self._end_snapshot

    def summary(self) -> dict[str, Any]:
        """
        汇总 start/stop 区间信息。
        若尚未调用 stop，会使用当前 snapshot 作为结束态。
        """
        start = self._start_snapshot or self.snapshot()
        end = self._end_snapshot or self.snapshot()
        return {
            "start": start,
            "end": end,
            "delta_vram_allocated_mb": float(end["vram_allocated_mb"] - start["vram_allocated_mb"]),
            "delta_vram_reserved_mb": float(end["vram_reserved_mb"] - start["vram_reserved_mb"]),
            "delta_process_rss_mb": float(end["process_rss_mb"] - start["process_rss_mb"]),
            # 峰值直接取结束快照中的统计值（由 CUDA 运行时维护）
            "vram_peak_mb": float(end["vram_peak_mb"]),
            "vram_peak_gb": float(end["vram_peak_gb"]),
        }


class TimeAndMemoryTracker(ContextDecorator):
    """
    同时支持“上下文管理器 + 装饰器”的推理性能追踪器。

    可自动收集：
    - TTFT (Time To First Token)
    - 推理总耗时 (Latency)
    - 吞吐量 (Tokens per second)
    - 显存占用 / 显存峰值
    - 当前模型配置（量化、是否 vLLM 等）

    用法 1：上下文管理器
        with time_and_memory_tracker(input_text=prompt, use_vllm=True, quantization="int4") as tracker:
            out = generate(prompt)
            tracker.set_output_text(out)
        metrics = tracker.metrics

    用法 2：装饰器
        @time_and_memory_tracker(use_vllm=True, quantization="int4")
        def generate(prompt: str) -> str:
            ...
        output, metrics = generate("hello")

    TTFT 说明：
    - 对“非流式”函数（一次性返回字符串），无法观测真实首 token 到达时刻，
      默认退化为 ttft == latency。
    - 若是流式推理，可在拿到首 token 时手动调用 mark_first_token()，即可得到真实 TTFT。
    """

    def __init__(
        self,
        *,
        device_index: int = 0,
        input_text: str | None = None,
        output_text: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        token_counter: Callable[[str | None], int] = estimate_tokens_from_text,
        use_vllm: bool | None = None,
        quantization: str | bool | None = None,
        model_config: dict[str, Any] | None = None,
        return_metrics: bool = True,
    ) -> None:
        self.resource_monitor = ResourceMonitor(device_index=device_index)
        self.token_counter = token_counter
        self.return_metrics = return_metrics

        self.input_text = input_text
        self.output_text = output_text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

        base_model_config: dict[str, Any] = {}
        if model_config:
            base_model_config.update(model_config)
        if use_vllm is not None:
            base_model_config["use_vllm"] = bool(use_vllm)
        if quantization is not None:
            base_model_config["quantization"] = quantization
        self.model_config = base_model_config

        self._t_start: float | None = None
        self._t_end: float | None = None
        self._t_first_token: float | None = None
        self._metrics: dict[str, Any] | None = None

    def set_input_text(self, text: str | None) -> None:
        self.input_text = text

    def set_output_text(self, text: str | None) -> None:
        self.output_text = text

    def set_input_tokens(self, num_tokens: int) -> None:
        self.input_tokens = max(0, int(num_tokens))

    def set_output_tokens(self, num_tokens: int) -> None:
        self.output_tokens = max(0, int(num_tokens))

    def mark_first_token(self) -> None:
        """
        在“真实首 token 出现”瞬间调用此函数。
        典型场景：vLLM 流式输出回调中检测到第一个 chunk 时调用。
        """
        if self._t_start is None:
            return
        if self._t_first_token is None:
            self._t_first_token = time.perf_counter()

    def _resolve_input_tokens(self) -> int:
        if self.input_tokens is not None:
            return int(max(0, self.input_tokens))
        return int(max(0, self.token_counter(self.input_text)))

    def _resolve_output_tokens(self) -> int:
        if self.output_tokens is not None:
            return int(max(0, self.output_tokens))
        return int(max(0, self.token_counter(self.output_text)))

    def _finalize_metrics(self) -> dict[str, Any]:
        if self._t_start is None:
            raise RuntimeError("Tracker 尚未启动，无法计算指标。")
        if self._t_end is None:
            self._t_end = time.perf_counter()

        memory_summary = self.resource_monitor.summary()
        end_snapshot = memory_summary["end"]

        latency_sec = max(0.0, self._t_end - self._t_start)
        ttft_source = "manual_first_token"
        if self._t_first_token is None:
            # 非流式模式通常无法观测首 token，退化处理为 latency。
            self._t_first_token = self._t_end
            ttft_source = "fallback_equals_latency"
        ttft_sec = max(0.0, self._t_first_token - self._t_start)

        in_tokens = self._resolve_input_tokens()
        out_tokens = self._resolve_output_tokens()
        total_tokens = in_tokens + out_tokens

        # 常见推理吞吐定义：输出 token / 总耗时
        throughput_tps = (out_tokens / latency_sec) if latency_sec > 0 else 0.0
        decode_time_sec = max(0.0, latency_sec - ttft_sec)
        decode_tps = (out_tokens / decode_time_sec) if decode_time_sec > 0 else 0.0

        # 顶层扁平字段：兼容旧调用方，也便于日志系统直接采集
        metrics: dict[str, Any] = {
            "ttft_sec": ttft_sec,
            "latency_sec": latency_sec,
            "throughput_tps": throughput_tps,
            "decode_tps": decode_tps,
            "decode_time_sec": decode_time_sec,
            "ttft_source": ttft_source,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": total_tokens,
            "vram_allocated_mb": float(end_snapshot["vram_allocated_mb"]),
            "vram_allocated_gb": float(end_snapshot["vram_allocated_gb"]),
            "vram_reserved_mb": float(end_snapshot["vram_reserved_mb"]),
            "vram_reserved_gb": float(end_snapshot["vram_reserved_gb"]),
            "peak_vram_mb": float(memory_summary["vram_peak_mb"]),
            "peak_vram_gb": float(memory_summary["vram_peak_gb"]),
            "vram_peak_gb": float(memory_summary["vram_peak_gb"]),
            "process_rss_mb": float(end_snapshot["process_rss_mb"]),
            "process_rss_gb": float(end_snapshot["process_rss_gb"]),
            "device": end_snapshot["device"],
            "cuda_available": bool(end_snapshot["cuda_available"]),
            "psutil_available": bool(end_snapshot["psutil_available"]),
            "model_config": dict(self.model_config),
            # 保留详细内存结构，排查问题时可直接使用
            "memory_detail": memory_summary,
        }
        return metrics

    def __enter__(self) -> "TimeAndMemoryTracker":
        self._metrics = None
        self._t_start = time.perf_counter()
        self._t_end = None
        self._t_first_token = None
        self.resource_monitor.start(reset_peak=True)
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        self.resource_monitor.stop()
        self._t_end = time.perf_counter()
        self._metrics = self._finalize_metrics()
        # 返回 False，异常继续向上抛出
        return False

    @property
    def metrics(self) -> dict[str, Any]:
        if self._metrics is None:
            raise RuntimeError("监控尚未结束，请在 with 块结束后读取 metrics。")
        return self._metrics

    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        """
        使 tracker 既可作为上下文，也可直接作为装饰器。
        装饰后默认返回 (result, metrics)。
        """

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # 每次调用函数时都创建新的 tracker，避免跨调用状态污染
            monitor = TimeAndMemoryTracker(
                device_index=self.resource_monitor.device_index,
                input_text=self.input_text,
                output_text=self.output_text,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                token_counter=self.token_counter,
                model_config=dict(self.model_config),
                return_metrics=self.return_metrics,
            )

            # 如果调用方没有显式传 input_text，尝试自动推断
            if monitor.input_text is None:
                monitor.input_text = _infer_prompt_text(args, kwargs)

            with monitor:
                result = func(*args, **kwargs)
                # 一次性返回结果（常见于 action JSON）时，自动提取输出文本用于 token 估算
                if monitor.output_tokens is None and not inspect.isgenerator(result):
                    monitor.output_text = _safe_to_text(result)

            metrics = monitor.metrics
            wrapper.last_metrics = metrics  # type: ignore[attr-defined]
            if monitor.return_metrics:
                return result, metrics
            return result

        wrapper.last_metrics = None  # type: ignore[attr-defined]
        return wrapper


def time_and_memory_tracker(
    func: Callable[..., Any] | None = None,
    *,
    device_index: int = 0,
    input_text: str | None = None,
    output_text: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    token_counter: Callable[[str | None], int] = estimate_tokens_from_text,
    use_vllm: bool | None = None,
    quantization: str | bool | None = None,
    model_config: dict[str, Any] | None = None,
    return_metrics: bool = True,
) -> Any:
    """
    统一入口：可作为装饰器或上下文管理器。

    1) 装饰器模式：
        @time_and_memory_tracker(use_vllm=True, quantization="int4")
        def infer(prompt: str) -> str:
            ...
        out, metrics = infer("...")

    2) 上下文模式：
        with time_and_memory_tracker(input_text=prompt, model_config={"use_vllm": True}) as mon:
            out = infer(prompt)
            mon.set_output_text(out)
        print(mon.metrics)
    """
    monitor = TimeAndMemoryTracker(
        device_index=device_index,
        input_text=input_text,
        output_text=output_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        token_counter=token_counter,
        use_vllm=use_vllm,
        quantization=quantization,
        model_config=model_config,
        return_metrics=return_metrics,
    )
    if func is not None:
        return monitor(func)
    return monitor


# -----------------------------------------------------------------------------
# 兼容旧接口（项目中已有调用 monitor_inference_performance）
# -----------------------------------------------------------------------------
InferencePerformanceMonitor = TimeAndMemoryTracker
monitor_inference_performance = time_and_memory_tracker


def _demo_generation(prompt: str, *, delay_sec: float = 0.25) -> str:
    """
    一个最小可运行 demo：模拟模型生成 action JSON。
    """
    time.sleep(delay_sec)
    return json.dumps({"action": "pick", "target": "cup", "instruction": prompt}, ensure_ascii=False)


def demo() -> None:
    prompt = "请生成一条抓取杯子的 action JSON。"

    print("=== 上下文管理器示例 ===")
    with time_and_memory_tracker(
        input_text=prompt,
        use_vllm=True,
        quantization="int4",
        model_config={"model_name": "demo-model"},
    ) as tracker:
        output = _demo_generation(prompt)
        tracker.set_output_text(output)
    print("输出:", output)
    print("指标:", json.dumps(tracker.metrics, ensure_ascii=False, indent=2))

    print("\n=== 装饰器示例 ===")

    @time_and_memory_tracker(use_vllm=False, quantization=False)
    def infer_once(local_prompt: str) -> str:
        return _demo_generation(local_prompt, delay_sec=0.18)

    output2, metrics2 = infer_once("请给出移动到 home 位姿的 action JSON。")
    print("输出:", output2)
    print("指标:", json.dumps(metrics2, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo()
