# 04_sim_exp — 仿真端到端实验

目标：执行 `指令 -> 模型输出 action -> Genesis 执行 action` 端到端流程。

## 目录

```text
experiments/04_sim_exp/
├── README.md
├── run_e2e_sim.py
├── test_genesis_interactive_env.py
└── configs/
    └── e2e_sim.yaml
```

## 前置条件

1. 已安装 Genesis Python 包（可导入 `genesis`）
2. 可访问 Franka 资产文件（默认 `xml/franka_emika_panda/panda.xml`）

推荐先执行：

```bash
bash scripts/bootstrap_sim_assets.sh
```

如使用自定义路径：

```bash
export GENESIS_REPO_DIR="/abs/path/to/Genesis"
export GENESIS_ASSETS_ROOT="/abs/path/to/Genesis/genesis/assets"
```

## 最小启动命令

### 1) 端到端执行

```bash
PYTHONPATH=. python experiments/04_sim_exp/run_e2e_sim.py \
  --instruction "移动到方块上方并张开夹爪"
```

### 2) 交互测试环境

```bash
PYTHONPATH=. python experiments/04_sim_exp/test_genesis_interactive_env.py
```

## 配置路径

- 基础配置：`configs/base.yaml`
- 本实验覆盖：`experiments/04_sim_exp/configs/e2e_sim.yaml`

主要相关字段：

- `app.interactive`
- `app.sim`
- `app.state_injection`
- `app.inference`

## 与统一 CLI 的关系

- `python cli.py app run-instruction`：只做 `instruction -> action`（不执行仿真动作）
- 端到端仿真执行当前通过本实验脚本完成

## 功能边界

- 当前状态：实验性（依赖本地 Genesis 环境与图形/资产可用性）
