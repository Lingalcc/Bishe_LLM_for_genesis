# 实验 11_exp7_vllm 使用说明

这个补充实验用于回答一个更具体的问题：
在同样的 `vLLM + bitsandbytes 4bit` 部署栈下，如果把可用显存预算压到 `8GB / 6GB / 4GB / 2GB`，推理速度会分别怎样变化？

## 实验设计

- 固定模型：默认复用仓库已有的 `model/Qwen_Qwen2.5-3B-Instruct`
- 固定后端：`vllm`
- 固定量化：`4bit`
- 固定推理口径：同一批 prompts、同一 `batch_size`、同一 `num_samples`、同一 `max_new_tokens`
- 唯一核心变量：显存预算档位

这里的“显存预算”不是直接给 CUDA 设置一个绝对上限，而是先根据当前可见 GPU 的总显存，换算成 vLLM 的 `gpu_memory_utilization`：

```text
gpu_memory_utilization = clamp(requested_budget_mb / total_gpu_memory_mb, 0.05, 0.99)
```

所以：

- `8GB / 6GB / 4GB / 2GB` 是“近似预算档位”
- 若当前 GPU 总显存不是刚好 `8192 MB`，脚本会自动按比例换算
- `8GB` 档在 8GB 卡上通常会被轻微钳制到 `0.99`，避免把比例直接打满

## 运行方式

默认运行四档预算：

```bash
python experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py
```

指定预算档位：

```bash
python experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py \
  --memory-budgets-gb 8,6,4,2
```

如果当前环境无法自动探测总显存，可以手动指定：

```bash
python experiments/11_exp7_vllm/run_exp7_vllm_benchmark.py \
  --total-gpu-memory-mb 8192
```

## 输出产物

运行完成后会在 `experiments/11_exp7_vllm/reports/` 下生成：

- `exp7_vllm_memory_budget_comparison.csv`
- `exp7_vllm_memory_budget_summary.json`
- `exp7_vllm_budget_report.md`
- `exp7_vllm_latency_bar.png`
- `exp7_vllm_throughput_bar.png`
- `exp7_vllm_memory_bar.png`
- `run_meta.json`
- 每个预算档位对应的 `*_benchmark.json`

## 如何解读

- 如果你关心交互响应，优先看 `Avg Latency (s)` 和 `P95 Latency (s)`
- 如果你关心部署下限，优先看 `Status` 是否变成 `oom`
- 如果显存预算很低，吞吐下降通常说明可用 KV Cache 太小，或者初始化阶段编译与运行时开销已经挤占了预算

## 和 Exp5 的关系

这个实验可以视为 [`09_exp5_engine`](/home/lin/Bishe_LLM_for_genesis/experiments/09_exp5_engine) 的补充切片：

- Exp5 回答的是“不同部署栈谁更快”
- 这里回答的是“同一个 vLLM 部署栈在不同显存预算下还能快到什么程度”
