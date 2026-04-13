# 实验 20 / Exp16：Genesis Show 展示实验

这个实验用于把当前最终部署策略 `Top18Rank8 + vLLM + AWQ` 直接接到 Genesis 交互执行链路上，形成一个适合答辩或现场演示的展示入口。

默认配置如下：

- 模型：`model/qwen2.5-3b-top18-rank8-merged-awq`
- 推理后端：`vllm`
- 量化格式：`compressed-tensors`
- vLLM 兼容性检查：默认跳过，保持与 `exp15` 已验证运行口径一致
- 状态注入：开启
- Viewer：开启

也就是说，用户输入自然语言后，系统会执行这条链路：

`自然语言指令 -> Top18Rank8 模型生成 JSON action -> Genesis 执行动作 -> 实时查看结果`

## 运行方式

交互式展示：

```bash
python experiments/20_exp16_genesis_show/run_exp16_genesis_show.py
```

单次指令执行：

```bash
python experiments/20_exp16_genesis_show/run_exp16_genesis_show.py \
  --instruction "移动到方块上方10厘米处"
```

如果你只想在终端里看结果，不打开 Viewer：

```bash
python experiments/20_exp16_genesis_show/run_exp16_genesis_show.py \
  --instruction "先张开夹爪，再移动到工作台中央上方30厘米处" \
  --hide-viewer
```

也可以继续复用统一 CLI：

```bash
python cli.py app interactive \
  --config experiments/20_exp16_genesis_show/configs/genesis_show.yaml
```

如果你想强制启用严格版本检查：

```bash
python experiments/20_exp16_genesis_show/run_exp16_genesis_show.py \
  --strict-vllm-compat-check
```

## 交互命令

进入交互模式后可用：

- `/help`：显示帮助
- `/state`：查看当前机械臂状态摘要
- `/scene`：查看当前场景实体状态
- `/example`：打印示例 action JSON
- `/examples`：打印推荐自然语言指令
- `/actions`：打印执行器支持的动作示例
- `/raw on|off`：开启或关闭模型原始输出
- `/quit`：退出

## 当前场景

当前默认场景里会创建：

- `franka` 机械臂
- `cube` 方块
- `ground` 地面

因此最适合现场演示的自然语言指令，是那些围绕 `cube` 做定位、抓取、抬升的操作。

## 可执行动作

下面这些动作是当前执行器已经支持的 JSON action：

| 动作名 | 说明 | 示例 |
| --- | --- | --- |
| `move_ee` | 移动末端执行器到目标位置和姿态 | `{"commands":[{"action":"move_ee","pos":[0.65,0.0,0.15],"quat":[0,1,0,0]}]}` |
| `open_gripper` | 张开夹爪 | `{"commands":[{"action":"open_gripper","position":0.04}]}` |
| `close_gripper` | 闭合夹爪 | `{"commands":[{"action":"close_gripper","position":0.0}]}` |
| `wait` | 推进仿真若干步 | `{"commands":[{"action":"wait"}]}` |
| `get_state` | 读取当前机械臂状态 | `{"commands":[{"action":"get_state"}]}` |
| `reset_scene` | 执行轻量级场景复位步进 | `{"commands":[{"action":"reset_scene"}]}` |
| `set_qpos` | 直接设置整条机械臂与夹爪关节位置 | `{"commands":[{"action":"set_qpos","qpos":[0.0,-0.785,0.0,-2.356,0.0,1.571,0.785,0.04,0.04]}]}` |
| `set_dofs_position` | 为指定关节设置目标位置 | `{"commands":[{"action":"set_dofs_position","values":[-0.52,-1.2],"dofs_idx_local":[1,3]}]}` |
| `control_dofs_position` | 按位置控制方式驱动指定关节 | `{"commands":[{"action":"control_dofs_position","values":[0.2,-0.2],"dofs_idx_local":[1,3]}]}` |
| `control_dofs_velocity` | 按速度控制方式驱动指定关节 | `{"commands":[{"action":"control_dofs_velocity","values":[0.1,-0.1],"dofs_idx_local":[1,3]}]}` |
| `control_dofs_force` | 对指定自由度施加力或力矩 | `{"commands":[{"action":"control_dofs_force","values":[0.0,0.0,-2.0],"dofs_idx_local":[0,1,2]}]}` |

现场演示优先推荐：

- `move_ee`
- `open_gripper`
- `close_gripper`
- `wait`
- `get_state`

这些动作最稳定，也最容易让观众看懂“语言 -> 动作 -> 执行”的完整链路。

## 推荐自然语言指令

推荐你直接复制下面这些指令做展示：

1. `移动到方块上方10厘米处`
2. `先张开夹爪，再移动到工作台中央上方30厘米处`
3. `移动到方块上方，下降到方块高度后闭合夹爪，再抬起到更高位置`
4. `读取当前机械臂状态`
5. `将机械臂移动到x=0.5米、y=0.0米、z=0.3米的位置，保持朝下姿态`
6. `移动到方块附近后关闭夹爪`
7. `保持当前姿态并等待一下`

如果你要做更复杂的展示，建议遵循一个经验：

- 先说目标位置或目标物体
- 再说夹爪动作
- 最后说是否抬起、等待或读取状态

例如：

- `移动到方块上方，张开夹爪，下降后夹住，再抬起`
- `移动到桌面中央上方，保持朝下姿态，然后读取当前状态`

## 输出与记录

每次启动脚本都会在：

- `experiments/20_exp16_genesis_show/reports/sessions/`

下创建一个新的会话目录，并把每轮执行结果保存成 `turn_XXX.json`，方便答辩后回看每条指令对应的：

- 模型原始输出
- 规范化后的 action JSON
- 每条命令的执行结果
- 本轮执行耗时

## 注意事项

- 这个实验默认依赖本地 `vllm` 和 `Genesis` 环境。
- 当前默认会跳过 `vllm / compressed-tensors` 的保守版本检查，因为 `exp15` 已经在同仓库里用这套环境成功跑通过。
- 如果显卡较紧张，可以把配置里的 `gpu_memory_utilization` 从 `0.8` 调低到 `0.5` 再试。
- 如果只想做链路演示、不需要图形窗口，可以使用 `--hide-viewer`。
