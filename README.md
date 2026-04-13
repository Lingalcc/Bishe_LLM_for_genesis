# Bishe_LLM_for_genesis

面向 Genesis 仿真场景的 LLM 工程项目：覆盖数据生成、微调、评测、推理性能基准和仿真联调。

Google Colab 使用说明见 [CLOUD_README.md](/home/lin/Bishe_LLM_for_genesis/CLOUD_README.md)，当前云端流程已更新为“Colab 本地高速磁盘运行，Google Drive 仅导出最终产物”。

## 一页速览（答辩演示建议）

```bash
# 1) 安装
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 2) 数据生成（API）
python cli.py data generate --config experiments/01_data_exp/configs/api_generate.yaml

# 3) 数据切分（训练前必做）
python cli.py data split --config experiments/01_data_exp/configs/api_generate.yaml

# 4) 微调（先 dry-run）
python cli.py finetune start --config experiments/02_finetune_exp/configs/train.yaml --dry-run

# 5) 准确率评测
python cli.py eval accuracy --config experiments/03_eval_exp/configs/accuracy.yaml

# 6) 推理 benchmark（本地模型，按实际模型路径替换）
python cli.py eval benchmark --backend transformers --model-path model/Qwen_Qwen2.5-3B-Instruct
```

仿真端到端启动见下文「最小仿真启动」。

## 当前目录结构（与代码一致）

```text
Bishe_LLM_for_genesis/
├── cli.py
├── configs/
│   └── base.yaml
├── experiments/
│   ├── 01_data_exp/
│   │   ├── README.md
│   │   ├── run_api_generate.py
│   │   └── configs/api_generate.yaml
│   ├── 02_finetune_exp/
│   │   ├── README.md
│   │   ├── run_train.py
│   │   ├── run_benchmark.py
│   │   └── configs/
│   ├── 03_eval_exp/
│   │   ├── README.md
│   │   ├── run_accuracy.py
│   │   └── configs/accuracy.yaml
│   ├── 04_sim_exp/
│   │   ├── README.md
│   │   ├── run_e2e_sim.py
│   │   ├── test_genesis_interactive_env.py
│   │   └── configs/e2e_sim.yaml
│   └── finetune_exp/
│       └── README.md
├── scripts/
│   └── bootstrap_sim_assets.sh
├── src/
│   ├── data_core/
│   ├── finetune_core/
│   ├── eval_core/
│   ├── sim_core/
│   ├── app/
│   ├── genesis/
│   ├── protocols/
│   └── utils/
└── tests/
```

## 配置路径规则

统一采用「基础配置 + 实验覆盖配置」模式。

- 基础配置：`configs/base.yaml`
- 数据实验覆盖：`experiments/01_data_exp/configs/api_generate.yaml`
- 微调实验覆盖：`experiments/02_finetune_exp/configs/train.yaml`
- 评测实验覆盖：`experiments/03_eval_exp/configs/accuracy.yaml`
- 仿真实验覆盖：`experiments/04_sim_exp/configs/e2e_sim.yaml`

说明：

- CLI 默认 `--base-config configs/base.yaml`
- 通过 `--config <实验yaml>` 做深度覆盖
- 旧路径 `configs/default.yaml` 仅兼容回退，不作为主文档路径

## 实际 CLI 命令（以 `python cli.py --help` 为准）

### data

- `python cli.py data generate [--base-config ...] [--config ...]`
- `python cli.py data calibrate [--base-config ...] [--config ...]`
- `python cli.py data split [--base-config ...] [--config ...] [--input-file ...] [--out-dir ...] [--train-ratio ...] [--val-ratio ...] [--test-ratio ...] [--seed ...]`

### finetune

- `python cli.py finetune start [--base-config ...] [--config ...] [--dry-run] [--finetune-method {lora,qlora,dora,galore}] [extra_args...]`
- `python cli.py finetune benchmark [--base-config ...] [--config ...] [--dry-run] [--skip-train] [--skip-base-eval] [--eval-only {base,finetuned}]`

### eval

- `python cli.py eval accuracy [--base-config ...] [--config ...]`
- `python cli.py eval benchmark --backend {transformers,vllm,llama.cpp,exllamav2} --model-path <path> [其他可选参数]`

### app

- `python cli.py app run-instruction --instruction "..." [--print-raw] [--disable-sim-state] [--base-config ...] [--config ...]`

## 从零开始最小流程

### 1) 数据生成与校验

```bash
# data generate 已内置去重与自动补采样（尽量达到 num_samples）
python cli.py data generate --config experiments/01_data_exp/configs/api_generate.yaml
python cli.py data calibrate
python cli.py data split --config experiments/01_data_exp/configs/api_generate.yaml
```

### 2) 训练（微调）

```bash
# 训练前会检查 split 文件是否存在，不存在会提示先执行 data split

# 建议先检查命令拼装是否正确
python cli.py finetune start --config experiments/02_finetune_exp/configs/train.yaml --dry-run

# 实际训练
python cli.py finetune start --config experiments/02_finetune_exp/configs/train.yaml
```

### 3) 准确率评测

```bash
python cli.py eval accuracy --config experiments/03_eval_exp/configs/accuracy.yaml
```

### 4) benchmark（推理性能）

```bash
python cli.py eval benchmark \
  --backend vllm \
  --model-path model/Qwen_Qwen2.5-3B-Instruct \
  --num-samples 32 \
  --batch-size 1 \
  --quantization 4bit \
  --require-gpu \
  --output-json experiments/03_eval_exp/reports/inference_benchmark.json
```

### 5) 最小仿真启动（端到端）

```bash
# 一次性拉取/定位 Genesis 资产
bash scripts/bootstrap_sim_assets.sh

# 如需自定义路径
export GENESIS_REPO_DIR="/abs/path/to/Genesis"
export GENESIS_ASSETS_ROOT="/abs/path/to/Genesis/genesis/assets"

# 运行端到端仿真实验（需要 PYTHONPATH）
PYTHONPATH=. python experiments/04_sim_exp/run_e2e_sim.py \
  --instruction "移动到方块上方并张开夹爪"
```

## 功能边界（答辩建议直接引用）

### 已实现并可演示

- API 生成 `instruction -> action JSON` 数据（Alpaca/ShareGPT）
- 数据切分（`data split`，输出 `train/val/test + split_metadata`）
- 基于 LLaMA-Factory 的微调启动与前后对比 benchmark
- 准确率评测（`parse_ok / exact_match / action_match`）
- 本地推理吞吐与延迟 benchmark（HF/vLLM）
- 仿真端到端链路（指令 -> 模型 -> action 执行）

### 实验性

- `finetune_method=dora/galore`（可用但建议在答辩中声明实验性）
- `eval accuracy` 的本地 `vllm` 后端
- `app.inference.mode=local`（依赖本地模型与推理后端环境）

### 计划中 / 未接入统一 CLI

- 数据增强（`data augment`）
- `app` 域下仍有更多展示型子命令可补充；当前统一 CLI 已提供 `run-instruction` 与 `interactive`

## Exp16 Genesis Show（最终演示入口）

这个实验把当前最终部署策略接到 Genesis 交互执行链路上：

- 模型：`model/qwen2.5-3b-top18-rank8-merged-awq`
- 推理后端：`vllm`
- 量化格式：`compressed-tensors`
- 目标：用户直接输入自然语言，机械臂实时执行并显示结果

推荐启动方式：

```bash
# 交互式展示
python experiments/20_exp16_genesis_show/run_exp16_genesis_show.py

# 单条指令直接执行
python experiments/20_exp16_genesis_show/run_exp16_genesis_show.py \
  --instruction "移动到方块上方10厘米处"

# 继续复用统一 CLI 也可以
python cli.py app interactive \
  --config experiments/20_exp16_genesis_show/configs/genesis_show.yaml
```

当前执行器支持的核心动作：

- `move_ee`
- `open_gripper`
- `close_gripper`
- `wait`
- `get_state`
- `reset_scene`
- `set_qpos`
- `set_dofs_position`
- `control_dofs_position`
- `control_dofs_velocity`
- `control_dofs_force`

推荐现场演示的自然语言指令：

1. `移动到方块上方10厘米处`
2. `先张开夹爪，再移动到工作台中央上方30厘米处`
3. `移动到方块上方，下降到方块高度后闭合夹爪，再抬起到更高位置`
4. `读取当前机械臂状态`
5. `将机械臂移动到x=0.5米、y=0.0米、z=0.3米的位置，保持朝下姿态`

更完整的动作 JSON 示例、交互命令和演示建议见：

- [experiments/20_exp16_genesis_show/README.md](/home/lin/Bishe_LLM_for_genesis/experiments/20_exp16_genesis_show/README.md)

## 运行与测试

安装后可使用入口命令：

```bash
genesis-cli --help
```

测试建议：

```bash
pytest -q
```
