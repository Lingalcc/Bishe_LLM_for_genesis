# Bishe_LLM_for_genesis

面向 Genesis 仿真场景的大语言模型工程化项目，覆盖数据构建、微调、评测和交互式应用全流程。

## 项目目标

1. 将自然语言指令转换为可执行的机器人 JSON action。
2. 提供从数据集生成到模型评测的统一工程链路。
3. 通过统一 CLI 降低多脚本分散调用的维护成本。

## 当前目录结构

```text
Bishe_LLM_for_genesis/
├── configs/
│   └── default.yaml               # 统一 YAML 配置
├── src/
│   ├── data/                      # 数据生成、增强、校准
│   ├── finetune/                  # 微调入口与执行逻辑
│   ├── eval/                      # 精度评测与性能统计
│   ├── app/                       # 指令交互与仿真执行
│   └── utils/                     # 通用工具（配置加载等）
├── tests/                         # 回归测试脚本
├── cli.py                         # 统一命令行入口
├── pyproject.toml                 # 打包与依赖配置
└── README.md
```

## 环境要求

1. Python `>=3.10`
2. 建议 Linux + NVIDIA GPU（涉及 `torch`、`vllm`、`bitsandbytes` 时）
3. 已安装 Genesis 运行所需依赖（用于仿真与回归测试）

## 安装

1. 创建虚拟环境并激活。
2. 在项目根目录执行：

```bash
pip install -e .
```

安装后可使用命令 `genesis-cli`。

## 配置说明

统一配置文件路径：

```text
configs/base.yaml
```

主要配置分组：

1. `dataset_prepare`：数据生成、增强、校准参数
2. `finetune.train`：LLaMA-Factory 训练参数
3. `test.accuracy_eval`：评测参数
4. `app.interactive` / `app.inference`：应用运行与模型推理参数（支持 API / 本地模型切换）

如需在线调用 OpenAI 兼容接口，必须通过环境变量提供密钥（不要把真实 key 写入 YAML）：

1. `OPENAI_API_KEY`（默认用于评测与 app API 模式）
2. `DEEPSEEK_API_KEY`（默认用于数据生成）
3. 也可在配置中改 `api_key_env` 指向自定义环境变量名

示例：

```bash
cp .env.example .env
export OPENAI_API_KEY="sk-..."
export DEEPSEEK_API_KEY="sk-..."
```

说明：

1. 配置字段 `api_key` 仅为兼容保留，不再作为真实密钥来源。
2. 缺少环境变量时，程序会抛出明确错误并提示对应变量名。

## 统一 CLI 用法

查看总帮助：

```bash
python cli.py --help
```

或（安装后）：

```bash
genesis-cli --help
```

### 1. 数据生成

```bash
python cli.py data generate
python cli.py data generate -- --num-samples 1000 --seed 123
```

### 2. 数据增强

```bash
python cli.py data augment
python cli.py data augment -- --num-source 200 --aug-per-sample 1
```

### 3. 启动微调

```bash
python cli.py finetune start --dry-run
python cli.py finetune start --gpus 0 -- --num_train_epochs 3
```

### 4. 精度评测

```bash
python cli.py eval accuracy
python cli.py eval accuracy -- --num-samples 50
```

### 5. 应用交互（指令 -> action）

```bash
python cli.py app run-instruction
python cli.py app run-instruction --instruction "打开夹爪并移动到目标点"
```

## 典型工作流

1. 生成基础数据集：`data generate`
2. 使用 API 增强数据：`data augment`
3. 启动微调：`finetune start`
4. 运行精度评测：`eval accuracy`
5. 在仿真中交互验证：`app run-instruction`

## 测试

运行回归测试：

```bash
python tests/run_regression_tests.py --config configs/default.yaml --target all
```

可选分组：

1. `manager`
2. `controller`
3. `basic`

## 开发约定

1. 业务代码统一放在 `src/` 包内。
2. 配置统一通过 `src/utils/config.py` 加载 YAML。
3. 新功能优先接入 `cli.py`，避免新增散落 `run_xxx.py` 顶层入口。
