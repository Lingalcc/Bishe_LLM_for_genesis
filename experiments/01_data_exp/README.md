# 01_data_exp — 数据集生成与校验实验

通过高级 LLM API（DeepSeek / OpenAI / SiliconFlow 等）自动生成 Franka Emika Panda 机械臂的
**自然语言指令 → JSON 动作命令** 训练数据，输出格式兼容 LLaMA Factory 微调。

## 目录结构

```
01_data_exp/
├── README.md                          # 本文件
├── run_api_generate.py                # 主入口脚本（生成 + Demo + 动作查看）
└── configs/
    └── api_generate.yaml              # 实验配置（API、生成参数等）
```

## 快速开始

> 所有命令均在项目根目录 (`Bishe_LLM_for_genesis/`) 下执行。

### 1. 查看支持的机械臂动作

```bash
python experiments/01_data_exp/run_api_generate.py --show-actions
```

会打印全部 11 个基本动作及参数说明，无需 API 调用。

### 2. Demo 模式（小批量试生成）

```bash
python experiments/01_data_exp/run_api_generate.py --demo
```

生成 10 条样本并在终端预览 Alpaca / ShareGPT 格式，用于验证 API 连通性和输出质量。

### 3. 完整生成

```bash
python experiments/01_data_exp/run_api_generate.py
```

按 `api_generate.yaml` 中的配置批量生成数据集（默认 2000 条），输出到 `data_prepare/` 目录。

### 4. 数据校验

使用 CLI 统一入口对生成的数据集进行校验：

```bash
python cli.py data calibrate --config experiments/01_data_exp/configs/api_generate.yaml
```

或直接在 Python 中调用：

```python
from src.data_core.calibration import CalibrationConfig, calibrate_dataset

report = calibrate_dataset(CalibrationConfig(
    dataset_file="data_prepare/genesis_franka_toolcall_alpaca.json",
    max_print_errors=20,
    strict=False,
))
print(f"有效率: {report['valid_ratio']:.2%}  ({report['valid_rows']}/{report['total_rows']})")
```

## 配置说明

编辑 `configs/api_generate.yaml` 自定义所有参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `num_samples` | 生成样本总数 | 2000 |
| `batch_size` | 每次 API 调用生成的样本数 | 20 |
| `seed` | 随机种子（可复现） | 42 |
| `simple_ratio` | 单步动作占比 | 0.3 |
| `medium_ratio` | 2-3 步组合动作占比 | 0.4 |
| `complex_ratio` | 4-7 步复杂序列占比 | 0.3 |
| `state_context_ratio` | 含场景状态上下文的样本比例 | 0.7 |
| `api_base` | OpenAI 兼容 API 地址 | SiliconFlow |
| `model` | 模型名称 | DeepSeek-V3 |
| `api_key` | API 密钥（留空则读环境变量） | — |
| `api_key_env` | API 密钥环境变量名 | `DEEPSEEK_API_KEY` |
| `temperature` | 采样温度 | 0.9 |
| `max_tokens` | 单次最大输出 token | 4096 |
| `timeout` | 请求超时（秒） | 120 |
| `max_retries` | 失败重试次数 | 5 |
| `sleep_seconds` | 请求间隔（秒） | 0.5 |

### API 密钥配置

```bash
# 环境变量（推荐且默认）
export DEEPSEEK_API_KEY="sk-xxx"
# 若需自定义变量名，可在 api_generate.yaml 修改 api_key_env
```

## 输出文件

生成完成后在 `data_prepare/` 目录下产出：

| 文件 | 格式 | 用途 |
|------|------|------|
| `genesis_franka_toolcall_alpaca.json` | Alpaca | LLaMA Factory 微调（`instruction` / `input` / `output`） |
| `genesis_franka_toolcall_sharegpt.json` | ShareGPT | LLaMA Factory 微调（`conversations` 多轮格式） |
| `genesis_franka_toolcall_stats.json` | — | 生成统计（样本数、难度分布、API 调用次数等） |

### 样本示例

**Alpaca 格式：**
```json
{
  "instruction": "把机械臂末端移动到桌面中央位置",
  "input": "",
  "output": "{\"commands\": [{\"action\": \"move_ee\", \"pos\": [0.5, 0.0, 0.15], \"quat\": [0, 1, 0, 0]}]}"
}
```

**ShareGPT 格式：**
```json
{
  "conversations": [
    {"from": "system", "value": "你是 Franka 机械臂控制指令生成器..."},
    {"from": "human", "value": "把机械臂末端移动到桌面中央位置"},
    {"from": "gpt", "value": "{\"commands\": [{\"action\": \"move_ee\", \"pos\": [0.5, 0.0, 0.15], \"quat\": [0, 1, 0, 0]}]}"}
  ]
}
```

## 支持的 11 个基本动作

| # | 动作 | 说明 |
|---|------|------|
| 1 | `wait` | 暂停模拟 |
| 2 | `get_state` | 查询机器人状态 |
| 3 | `open_gripper` | 打开夹爪 |
| 4 | `close_gripper` | 关闭夹爪 |
| 5 | `set_qpos` | 设置全部 9 个关节角度 |
| 6 | `set_dofs_position` | 设置指定自由度位置 |
| 7 | `control_dofs_position` | 位置控制模式 |
| 8 | `control_dofs_velocity` | 速度控制模式 |
| 9 | `control_dofs_force` | 力控制模式 |
| 10 | `move_ee` | 末端执行器笛卡尔空间移动（逆运动学） |
| 11 | `reset_scene` | 重置仿真场景 |

## 涉及的源码模块

| 模块 | 职责 |
|------|------|
| `src/data_core/api_client.py` | API 调用与 JSON 提取 |
| `src/data_core/format_utils.py` | Alpaca / ShareGPT 格式转换、样本校验 |
| `src/data_core/generate.py` | 数据集生成核心逻辑（Prompt 构建、批量生成） |
| `src/data_core/calibration.py` | 数据集校验（结构检查 + 动作合法性验证） |
