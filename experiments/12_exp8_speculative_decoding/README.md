# 实验12：Speculative Decoding 推理加速

本实验用于在当前机器人动作生成任务中验证 HuggingFace `transformers` 的投机解码能力。核心思路是使用 `Qwen2.5-3B-Instruct` 作为主模型，使用更小的 `Qwen2.5-0.5B-Instruct` 作为草稿模型，在 `generate()` 中通过 `assistant_model=assistant_model` 启用 speculative decoding，对结构化 JSON Action 输出进行推理加速评测。

## 1. 实验目标

- 对比普通 greedy decoding 与 speculative decoding 的端到端平均延迟。
- 统计生成阶段的 token 吞吐率。
- 记录推理阶段峰值显存占用。
- 通过 `json.loads()` 校验输出是否仍然保持可解析 JSON，避免加速破坏格式稳定性。

这类任务非常适合投机解码，因为输出大量包含固定 JSON 结构 token，例如 `{"commands":[...]}`、`"action"`、`"pos"`、`"quat"` 等，主模型对这些 token 的接受率通常较高。

## 2. 目录结构

```text
experiments/12_exp8_speculative_decoding/
├── README.md
├── configs/
│   └── speculative.yaml
├── reports/
│   └── speculative_benchmark.json
└── run_speculative_benchmark.py
```

## 3. 配置说明

默认配置文件为 `configs/speculative.yaml`：

```yaml
target_model_path: "model/Qwen_Qwen2.5-3B-Instruct"
assistant_model_path: "model/Qwen_Qwen2.5-0.5B-Instruct"
dataset_path: "../../data_prepare/splits/test.json"
num_samples: 50
batch_size: 1
max_new_tokens: 256
temperature: 0.0
warmup_samples: 1
assistant_num_tokens: 8
assistant_confidence_threshold: 0.55
assistant_num_tokens_schedule: "heuristic_transient"
prefer_same_gpu: true
preferred_cuda_device: 0
allow_auto_device_map_fallback: true
report_path: "reports/speculative_benchmark.json"
trust_remote_code: true
prefer_flash_attention_2: true
```

说明：

- `target_model_path`：主模型路径，可以是 Hugging Face Repo ID、本地基础模型目录，或本地 LoRA adapter 目录。
- `assistant_model_path`：草稿模型路径，建议与主模型共享 tokenizer。
- `dataset_path`：测试集路径，默认读取 `data_prepare/splits/test.json`。
- `num_samples`：默认抽取前 50 条样本用于基准评测。
- `batch_size`：建议保持为 `1`，这是 speculative decoding 最容易体现加速收益的设置。
- `temperature`：建议固定为 `0.0`，即 greedy decoding，以最大化草稿验证命中率。
- `warmup_samples`：正式计时前先做预热，减少首次 kernel / cache 初始化对结果的污染。
- `assistant_num_tokens`：每轮 speculative 草稿 token 上限。对当前短 JSON 任务，推荐先从 `8` 起步，而不是库默认的 `20`。
- `assistant_confidence_threshold`：assistant 置信度阈值。适当调高可以减少低质量草稿带来的回退重算。
- `assistant_num_tokens_schedule`：推荐 `heuristic_transient`，让每条样本都从较稳妥的草稿步长重新开始。
- `prefer_same_gpu`：优先把 target 和 assistant 放到同一张 GPU，尽量避免 CPU offload 和跨设备搬运。
- `allow_auto_device_map_fallback`：如果单卡放不下，再回退到 `device_map=auto`，优先保证实验能跑通。

## 4. 运行方式

### 4.1 默认运行：同时输出基线与投机解码对比

```bash
python experiments/12_exp8_speculative_decoding/run_speculative_benchmark.py \
  --config experiments/12_exp8_speculative_decoding/configs/speculative.yaml
```

默认会顺序运行两组实验：

- `baseline_off`：不传 `assistant_model`，作为普通 greedy decoding 基线。
- `speculative_on`：在 `target_model.generate()` 中传入 `assistant_model=assistant_model`，启用投机解码。

新版脚本还会额外打印：

- target / assistant 的主设备位置
- 是否检测到 CPU offload
- assistant speculative 参数

### 4.2 只跑基线消融

```bash
python experiments/12_exp8_speculative_decoding/run_speculative_benchmark.py \
  --config experiments/12_exp8_speculative_decoding/configs/speculative.yaml \
  --disable-assistant
```

这个模式适合单独复现实验基线，或者在调试环境兼容性时先确认普通 `generate()` 正常工作。

### 4.3 覆盖默认模型路径

例如将主模型替换为本地 LoRA 目录：

```bash
python experiments/12_exp8_speculative_decoding/run_speculative_benchmark.py \
  --config experiments/12_exp8_speculative_decoding/configs/speculative.yaml \
  --target-model-path output/qwen2.5-3b-genesis-qlora
```

脚本会自动检测本地目录中是否存在 `adapter_config.json`。如果检测到 LoRA adapter，会先加载基座模型，再挂载 PEFT adapter。

### 4.4 调整 speculative 草稿步长

如果你想进一步压缩回退开销，可以把草稿步长继续调小：

```bash
python experiments/12_exp8_speculative_decoding/run_speculative_benchmark.py \
  --config experiments/12_exp8_speculative_decoding/configs/speculative.yaml \
  --assistant-num-tokens 4 \
  --assistant-confidence-threshold 0.6
```

通常可以按下面顺序试：

1. `assistant_num_tokens=8`
2. 如果仍然慢，试 `assistant_num_tokens=4`
3. 如果仍然出现大量回退，再把 `assistant_confidence_threshold` 提高到 `0.6` 或 `0.65`

## 5. 指标口径

脚本会把结果写入 `reports/speculative_benchmark.json`，主要包含以下指标：

- `avg_latency_sec_per_sample`：平均端到端延迟，单位秒/样本。
- `token_throughput_tps`：输出 token 吞吐率，单位 tokens/s。
- `peak_vram_mb`：推理阶段峰值显存，单位 MB。
- `parse_ok_rate`：使用 `json.loads()` 解析成功的比例。
- `samples`：样本级明细，包含输入、预测结果、输出 token 数、JSON 解析状态等。
- `target_primary_device / assistant_primary_device`：模型实际主设备。
- `target_cpu_offload_detected / assistant_cpu_offload_detected`：是否检测到 CPU / meta / disk offload。
- `assistant_num_tokens / assistant_confidence_threshold / assistant_num_tokens_schedule`：本次 speculative 的关键控制参数。

如果同时跑了基线和投机解码，结果中还会额外生成 `comparison` 字段，用于汇总：

- `latency_speedup_vs_baseline`
- `throughput_gain_pct_vs_baseline`
- `peak_vram_delta_mb`
- `parse_ok_rate_delta`

## 6. 实现细节

### 6.1 模型加载

- 主模型和草稿模型都使用 `AutoModelForCausalLM.from_pretrained(...)` 加载。
- 优先选择 `bfloat16`，若硬件不支持则退回 `float16`。
- 若环境中可用 `flash_attn`，脚本会优先尝试 `attn_implementation="flash_attention_2"`。
- tokenizer 只加载一次，并在两个模型上共享 `pad_token_id / eos_token_id` 配置。
- 新版脚本默认优先尝试“单卡同 GPU 放置”，只有放不下时才回退到 `device_map=auto`。

### 6.2 核心生成逻辑

投机解码核心代码路径在：

```python
output_ids = target_bundle.model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    assistant_model=assistant_bundle.model,
    generation_config=generation_config,
    num_assistant_tokens=8,
    assistant_confidence_threshold=0.55,
    num_assistant_tokens_schedule="heuristic_transient",
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
```

关闭投机解码时则不传 `assistant_model` 参数。

### 6.3 性能监控

脚本复用了仓库现有的 `src/eval_core/performance_monitor.py` 中的 `time_and_memory_tracker`：

- 自动记录批次耗时；
- 自动记录 CUDA 峰值显存；
- 自动统计输出 token 吞吐；
- 最终在样本级与整体级两层汇总。

## 7. 依赖与注意事项

- 需要当前环境的 `transformers` 支持 `generate(..., assistant_model=...)`。
- 若主模型或草稿模型来自 Hugging Face 远程仓库，首次运行需要能访问并下载模型。
- 若加载的是 LoRA adapter，本地环境还需要安装 `peft`。
- 若你希望最大化 speculative decoding 的接受率，建议保持：
  - `temperature = 0.0`
  - `batch_size = 1`
  - `assistant_num_tokens` 不要直接用默认 `20`
  - `target` 和 `assistant` 尽量都驻留在同一张 GPU
  - 输出格式尽量严格受控，只输出 JSON

## 8. 建议的实验解读方式

如果你最终发现：

- `avg_latency_sec_per_sample` 明显下降；
- `token_throughput_tps` 明显上升；
- `parse_ok_rate` 基本不变；

那么可以认为 speculative decoding 在这个“结构化 JSON 机器人动作生成任务”上是有效的。

如果速度提升不明显，通常优先检查这几个方向：

1. 主模型和草稿模型 tokenizer 是否完全一致。
2. 输出是否足够模板化，若自由文本比例过高，接受率会下降。
3. `temperature` 是否误设置为非 0。
4. target 和 assistant 是否有一方被 offload 到 CPU。
5. `assistant_num_tokens` 是否过大，导致大量回退重算。
6. `transformers` 版本是否较旧，或底层 CUDA/flash attention 配置未生效。
7. 草稿模型是否过弱，导致主模型频繁拒绝草稿 token。
