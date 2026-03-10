# 仿真应用

本目录专注于第四阶段：仿真执行与交互验证。

你要求的三部分仿真流程已拆分为三个独立入口：

1. 我给模型指令，模型输出 action
2. 我给 action，仿真引擎中的 Franka 运动
3. 我给模型指令，模型输出 action，仿真引擎根据模型输出运动

## 统一配置

默认读取统一配置文件：`configs/default.yaml`  
对应配置段：

- `app.interactive`
- `app.state_injection`
- `app.inference`

可通过 `app.interactive.enabled` 控制是否允许启动应用。
`app.inference.mode` 支持 `api` / `local` 切换。
API Key 建议配置在 `app.inference.api.api_key` 或环境变量 `app.inference.api.api_key_env`。
状态注入开关：

- `app.state_injection.enable_instruction_to_action`
- `app.state_injection.enable_instruction_to_motion`

## 前置依赖

1. 可导入 `Genesis` 及其依赖（包括 `gstaichi`）
2. 可用图形/仿真运行环境（用于 viewer）

## 目录结构

- `run_instruction_to_action.py`：指令 -> Action（仅模型推理）
- `run_action_to_motion.py`：Action -> Franka运动（仅仿真执行）
- `run_instruction_to_motion.py`：指令 -> Action -> Franka运动（端到端）
- `run_interactive_app.py`：兼容旧入口（Action -> Franka运动）

## 1) 指令 -> Action

```bash
python src/app/run_instruction_to_action.py
```

单次调用示例：

```bash
python src/app/run_instruction_to_action.py \
  --instruction "打开夹爪，移动到[0.65,0,0.2]，然后闭合夹爪"
```

默认会读取当前仿真场景状态并注入到模型输入。  
可用 `--disable-sim-state` 关闭状态注入做对照实验。

## 2) Action -> Franka运动

```bash
python src/app/run_action_to_motion.py
```

单次调用示例：

```bash
python src/app/run_action_to_motion.py \
  --action '{"commands":[{"action":"open_gripper"},{"action":"move_ee","pos":[0.65,0,0.2],"quat":[0,1,0,0]}]}'
```

## 3) 指令 -> Action -> Franka运动（端到端）

```bash
python src/app/run_instruction_to_motion.py
```

单次调用示例：

```bash
python src/app/run_instruction_to_motion.py \
  --instruction "先张开夹爪，再移动到盒子上方然后夹住"
```

## 推理配置示例

API 模式（OpenAI 兼容）：

```yaml
app:
  inference:
    mode: api
    api:
      api_base: https://api.openai.com/v1
      model: gpt-5
      api_key: ""
      api_key_env: OPENAI_API_KEY
      generation:
        temperature: 0.0
        max_tokens: 1200
        top_p: 1.0
```

本地模式（可切换推理后端）：

```yaml
app:
  inference:
    mode: local
    local:
      model_path: model/my_lora_merged_model
      backend: auto # auto | vllm | transformers
      quantization: null # null | awq | 4bit | 8bit
      generation:
        temperature: 0.0
        top_p: 1.0
        max_new_tokens: 512
```

每次推理前都会采集场景中各物体状态（名字、类别、坐标/姿态等）并注入模型。

## 交互命令

内置命令：

- `/help`：帮助
- `/state`：查看当前状态摘要
- `/scene`：查看场景全部实体状态
- `/example`：打印可直接执行的样例
- `/quit`：退出

## 适用场景

1. 分离验证模型动作生成质量（Part 1）
2. 分离验证动作执行稳定性（Part 2）
3. 端到端联调与演示（Part 3）
