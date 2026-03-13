# 03 — 准确率评测实验 (Accuracy Evaluation)

本实验用于评估模型将自然语言指令转换为 Franka 机械臂 JSON action 的能力，核心输出三类指标：

- `parse_ok_rate`：输出可被解析为合法 JSON tool-call 的比例
- `exact_match_rate`：预测命令与标注命令完全一致（动作与参数都一致）的比例
- `action_match_rate`：动作序列一致（忽略参数差异）的比例

评测入口：

- 脚本入口：`experiments/03_eval_exp/run_accuracy.py`
- 统一 CLI 入口：`cli.py eval accuracy`

## 目录结构

```text
experiments/03_eval_exp/
├── configs/
│   └── accuracy.yaml       # 实验覆盖配置（覆盖 configs/base.yaml 的 test.accuracy_eval）
├── reports/                # 推荐报告输出目录
├── run_accuracy.py         # 评测启动脚本
└── README.md
```

## 1. 快速开始

在仓库根目录执行。

### 1.1 用实验脚本执行（推荐）

```bash
python experiments/03_eval_exp/run_accuracy.py
```

该命令会：

- 读取 `configs/base.yaml`
- 再读取 `experiments/03_eval_exp/configs/accuracy.yaml`（如果存在）
- 合并后执行评测

### 1.2 用统一 CLI 执行

```bash
python cli.py eval accuracy --config experiments/03_eval_exp/configs/accuracy.yaml
```

如果不传 `--config`，会只使用 `configs/base.yaml` 的 `test.accuracy_eval` 配置。

## 2. 评测模式说明

`test.accuracy_eval.mode` 支持：

- `api`：在线调用 OpenAI 兼容接口进行预测
- `local`：本地模型推理（`transformers` 或 `vllm` 后端）

此外，`api` 模式下可通过 `predictions_file` 走离线预测评测（不发请求）。

### 2.1 API 在线评测

典型配置（放到 `experiments/03_eval_exp/configs/accuracy.yaml`）：

```yaml
test:
  accuracy_eval:
    mode: api
    dataset_file: data_prepare/genesis_franka_toolcall_alpaca.json
    report_file: experiments/03_eval_exp/reports/accuracy_report.json
    num_samples: 200
    seed: 42
    api_base: https://api.openai.com/v1
    model: gpt-5
    api_key: ""                 # 留空则读 api_key_env
    api_key_env: OPENAI_API_KEY
    temperature: 0.0
    max_tokens: 1200
    timeout: 120
    max_retries: 3
    sleep_seconds: 0.0
```

设置 API Key（二选一）：

```bash
export OPENAI_API_KEY="你的key"
python experiments/03_eval_exp/run_accuracy.py
```

也可在 yaml 中改 `api_key_env` 指向自定义环境变量名（不建议在 yaml 填真实 `api_key`）。

### 2.2 本地模型评测

典型配置：

```yaml
test:
  accuracy_eval:
    mode: local
    dataset_file: data_prepare/genesis_franka_toolcall_alpaca.json
    report_file: experiments/03_eval_exp/reports/accuracy_report_local.json
    num_samples: 200
    seed: 42

    model_path: model/qwen2.5-3b-genesis-qlora
    backend: transformers       # transformers | vllm
    quantization: null          # transformers: null/4bit/8bit
                               # vllm: 由 vLLM 支持的量化名称（如 awq 等）
    max_new_tokens: 512
    max_model_len: 4096         # 仅 vllm 生效
    gpu_memory_utilization: 0.9 # 仅 vllm 生效
    trust_remote_code: true
    temperature: 0.0
```

运行：

```bash
python experiments/03_eval_exp/run_accuracy.py
```

本地模式报告除准确率外，还会统计：

- `avg_latency_sec`
- `avg_throughput_tps`
- `avg_peak_vram_mb`
- `max_peak_vram_mb`

### 2.3 离线预测文件评测（不请求 API）

当你已经有模型输出时，可直接评测，设置：

```yaml
test:
  accuracy_eval:
    mode: api
    predictions_file: output/predictions.json
```

`predictions_file` 支持两种 JSON 结构：

- `list`：按评测采样顺序逐项对应预测文本/JSON
- `dict`：键为数据集原始索引（字符串），值为预测文本/JSON

注意：

- 使用 `predictions_file` 时不会校验 API Key，也不会发网络请求
- `num_samples` 和 `seed` 仍会影响采样样本集合

## 3. 参数详细说明（`test.accuracy_eval`）

以下参数来自 `configs/base.yaml` 与 `src/eval_core/accuracy.py`。

### 3.1 数据与输出

| 参数 | 默认值 | 说明 |
|---|---|---|
| `dataset_file` | `data_prepare/genesis_franka_toolcall_alpaca.json` | 待评测数据集（Alpaca 风格 JSON 列表） |
| `predictions_file` | `null` | 离线预测文件路径；设置后跳过在线请求 |
| `report_file` | `experiments/03_eval_exp/reports/accuracy_report.json` | 评测报告输出路径 |
| `num_samples` | `200` | 抽样评测条数（超出有效样本会自动截断） |
| `seed` | `42` | 抽样随机种子，保证可复现 |

### 3.2 通用控制

| 参数 | 默认值 | 说明 |
|---|---|---|
| `mode` | `api` | `api` 或 `local` |
| `temperature` | `0.0` | 生成温度，`0` 更稳定 |
| `system_prompt` | 内置中文提示词 | 本地模式强制作为 system 消息注入；API 模式使用数据集每条样本自带 `system` 字段 |

### 3.3 API 模式参数（`mode: api`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `api_base` | `https://api.openai.com/v1` | OpenAI 兼容接口根地址 |
| `model` | `gpt-5` | 调用模型名 |
| `api_key` | `""` | 显式 API Key，优先于环境变量 |
| `api_key_env` | `OPENAI_API_KEY` | 当 `api_key` 为空时读取的环境变量名 |
| `max_tokens` | `1200` | API 端单次生成 token 上限 |
| `timeout` | `120` | 单次请求超时（秒） |
| `max_retries` | `3` | 单样本失败重试次数（指数退避） |
| `sleep_seconds` | `0.0` | 样本间额外 sleep，便于限流 |

### 3.4 本地模式参数（`mode: local`）

| 参数 | 默认值 | 说明 |
|---|---|---|
| `model_path` | `""` | 本地模型路径；为空会退回 API 评测分支，务必设置 |
| `backend` | `transformers` | 推理后端：`transformers` / `vllm` |
| `quantization` | `null` | 量化配置（后端相关） |
| `max_new_tokens` | `512` | 本地生成 token 上限 |
| `max_model_len` | `4096` | `vllm` 上下文长度 |
| `gpu_memory_utilization` | `0.9` | `vllm` 显存利用率上限 |
| `trust_remote_code` | `true` | 是否允许模型仓库自定义代码 |

## 4. 输出与结果解读

终端会打印摘要：

```text
[ok] evaluated samples : 200
[ok] parse ok          : 180 (0.9000)
[ok] exact match       : 120 (0.6000)
[ok] action match      : 150 (0.7500)
```

报告 JSON（`report_file`）核心字段：

- `num_samples_evaluated`：本次实际评测条数
- `num_valid_rows_in_dataset`：数据集中可用于评测的有效样本数
- `parse_ok` / `parse_ok_rate`
- `exact_match` / `exact_match_rate`
- `action_match` / `action_match_rate`
- `details`：逐样本明细（含错误信息、预测预览等）

API 模式额外字段：

- `online_call_failures`：在线请求失败计数
- `exact_match_rate_on_parse_ok`

本地模式额外字段：

- `avg_latency_sec`
- `avg_throughput_tps`
- `avg_peak_vram_mb`
- `max_peak_vram_mb`

## 5. 常见问题与排查

### 5.1 报错 `missing API key`

原因：`mode=api` 且未提供 `predictions_file`，同时 `api_key`/环境变量为空。  
处理：设置 `api_key` 或 `export OPENAI_API_KEY=...`，或改用 `predictions_file`。

### 5.2 报错 `dataset file not found`

检查 `dataset_file` 路径是否存在，路径建议相对仓库根目录填写。

### 5.3 本地模式没有生效，像是在走 API

代码逻辑是：只有 `mode == local` 且 `model_path` 非空时才会走本地分支。  
请确认：

- `mode: local`
- `model_path: model/xxx`（非空且路径有效）

### 5.4 量化参数无效或报错

- `transformers` 后端仅支持 `null` / `4bit` / `8bit`
- `vllm` 的 `quantization` 需使用 vLLM 支持的量化名

### 5.5 `accuracy.yaml` 中 `report_file` 没写到 `reports/` 目录

允许这样做，脚本会自动创建父目录。  
但建议统一写到 `experiments/03_eval_exp/reports/`，便于管理多次实验结果。
