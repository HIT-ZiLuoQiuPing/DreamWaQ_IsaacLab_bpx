# BPX MuJoCo Sim2sim 关节映射修复说明

日期：2026-06-09

相关提交：

- `40802b4 Fix BPX MuJoCo sim2sim joint mapping`

目标：说明这次为什么 MuJoCo sim2sim 从“一进仿真就乱动、翻倒、抽搐”变成能稳定跟随低速/中速命令。重点是部署链路修正，不是训练 reward 或地形课程修改。

## 结论

这次 sim2sim 变好的核心原因是修正了 BPX 12 个关节在 MuJoCo runner 中的动作/观测顺序。

旧 runner 默认按每条腿分组：

```text
fl_roll, fl_pitch, fl_knee,
fr_roll, fr_pitch, fr_knee,
hl_roll, hl_pitch, hl_knee,
hr_roll, hr_pitch, hr_knee
```

但 06-05 基线策略在 IsaacLab 中实际对应的顺序更接近按关节类型分组：

```text
fl_roll, fr_roll, hl_roll, hr_roll,
fl_pitch, fr_pitch, hl_pitch, hr_pitch,
fl_knee, fr_knee, hl_knee, hr_knee
```

也就是 `type_major`。

这个顺序错了以后，策略输出的某个动作会被送到错误的关节上。例如 policy 以为自己在控制某条腿的 pitch，MuJoCo 却可能把它用到了另一条腿的 roll 或 knee。结果就是：

- 动作语义错位。
- 观测里的 joint pos/vel 也跟 action 对不上。
- CENet history 看到的是错误动力学闭环。
- policy 会迅速进入训练中从未见过的状态。
- 表现为乱动、抽搐、后倒、翻倒。

这类问题不能靠改 reward 根治，因为训练策略本身没有按这个错误映射训练过。

## 修复前的典型现象

用户之前的 MuJoCo sim2sim 日志中能看到：

- `action_mean_abs` 逐渐变大甚至异常。
- `height` 很快降到很低。
- `projected_gravity` 后期接近翻倒状态。
- 即使 `cmd_x=0.3`，机器人也无法稳定前进，经常原地扭动或转向。
- 旧工具会继续打印翻倒后的速度和动作，让日志看起来像还能运行，但实际机器人已经不在有效姿态。

这说明当时的问题不只是“策略走得不好”，而是 sim2sim 执行链路已经失真。

## 排查过程

### 1. 先验证 MuJoCo XML 和 PD 层

先跑 zero-action 站立测试，不加载策略，只让默认关节位置 PD 工作：

```bash
./isaaclab_waq.sh --mujoco-play --zero-action --duration 10 --real-time --debug-obs
```

结果：

- repo 内 `assets/BPX/mujoco/bpx.xml` 能站住。
- `/home/ubuntu/robot_rl/bpx_mjlab/src/bpx_mjlab/bpx/xmls/bpx.xml` 也能站住。
- 高度会从初始高度回落到 MuJoCo 静态站立高度附近，但不会炸、不翻、不出现 NaN。

这说明 MuJoCo XML 和 PD actuator 不是首要根因。

### 2. 对比不同关节顺序

用同一个 `policy_jit.pt` 测试不同顺序：

- `--joint-order metadata`
- `--joint-order leg_major`
- `--joint-order type_major`
- `--joint-order alphabetical`

结果显示：

- 旧 metadata/leg-major 容易快速翻倒。
- type-major 能稳定运行。
- 仅靠 `--flip-knee` 只能产生局部假象，不能解释完整行为。

最终判断：核心问题是 joint order，不是 knee sign。

### 3. 确认动作符号

这次没有把 knee sign 作为默认修复。

当前结论是：MuJoCo deployment 使用和 IsaacLab 一致的动作符号：

```text
hip_roll:  +1
hip_pitch: +1
knee:      +1
```

真正关键的是 type-major 顺序。

## 具体改动

### 1. `scripts/sim2sim/mujoco_play.py`

主要改动：

- 默认 joint order 改为 `type_major`。
- 新增 `leg_major` 选项，方便保留旧顺序做 A/B 测试。
- `metadata` 和 `alphabetical` 仍保留为诊断选项。
- 从 metadata 或 `bpx_constants.py` 读取 `action_sign`。
- 默认 action sign 全部为 `+1`。
- 增加安全 reset 原因打印：
  - `base_height_low`
  - `base_height_high`
  - `orientation_bad`
  - `qvel_high`
  - `non_finite_state`
- reset 后日志使用累计控制时间，避免 reset 后 `sim_t` 清零导致观察混乱。
- 支持加载 mjlab 的 MuJoCo XML：
  - 保留 XML 自己的 `meshdir`。
  - 如果 XML 没有 floor，runner 自动补 floor。
  - 如果 XML 没有 actuator，runner 自动补 12 个 motor actuator。
  - 统一设置 collision/friction/condim，便于部署测试。

当前默认 joint order：

```python
DEFAULT_JOINT_NAMES = [
    "fl_hip_roll_joint", "fr_hip_roll_joint", "hl_hip_roll_joint", "hr_hip_roll_joint",
    "fl_hip_pitch_joint", "fr_hip_pitch_joint", "hl_hip_pitch_joint", "hr_hip_pitch_joint",
    "fl_knee_joint", "fr_knee_joint", "hl_knee_joint", "hr_knee_joint",
]
```

### 2. `scripts/waq/export_policy.py`

主要改动：

- 导出 `policy_jit.json` 时写入 type-major 的 `joint_names`。
- 新增 metadata 字段：

```json
"joint_order": "type_major"
```

- 写入 `action_sign`：

```json
"action_sign": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

这样以后重新导出的 checkpoint 自带正确部署信息，不需要每次靠命令行覆盖。

### 3. `bpx_constants.py`

新增共享常量：

```python
BPX_MUJOCO_ACTION_SIGN = {
    ".*_hip_roll_joint": 1.0,
    ".*_hip_pitch_joint": 1.0,
    ".*_knee_joint": 1.0,
}
```

这个常量的意义是明确记录：当前 MuJoCo 部署不需要默认翻转 knee/pitch/roll 符号。

### 4. `README.md`

更新 MuJoCo sim2sim 使用说明：

- 明确默认 `--joint-order type_major`。
- 明确 `--flip-knee` / `--flip-hip-pitch` 只是诊断开关，不是默认修复路径。
- 增加直接使用 mjlab XML 的示例。
- 删除把 action clip / action scale multiplier 当常规解决方案的表述。

## 验证结果

### 1. 编译检查

```bash
/home/ubuntu/miniconda3/envs/isaaclab_bpx/bin/python -m compileall source/isaaclab_waq scripts
```

结果：通过。

### 2. 06-05 `model_44000.pt` 重新导出

```bash
./isaaclab_waq.sh --waq-export \
  --checkpoint logs/waq/bpx_waq_rough/2026-06-05_17-44-04/model_44000.pt \
  --control-profile legacy_mjlab_10hz
```

导出后 `policy_jit.json` 已确认：

```text
joint_order = type_major
action_sign = 全部 1.0
```

### 3. 平地 `cmd_x=0.3`

命令：

```bash
./isaaclab_waq.sh --mujoco-play \
  --policy logs/waq/bpx_waq_rough/2026-06-05_17-44-04/policy_jit.pt \
  --headless \
  --duration 10 \
  --print-interval 2
```

结果摘要：

```text
resets = 0
height ≈ 0.427-0.430
vel_x ≈ 0.19-0.22
```

这说明低速前进不再原地乱扭或翻倒。

### 4. 平地 `cmd_x=0.6`

命令：

```bash
./isaaclab_waq.sh --mujoco-play \
  --policy logs/waq/bpx_waq_rough/2026-06-05_17-44-04/policy_jit.pt \
  --command-x 0.6 \
  --headless \
  --duration 5 \
  --print-interval 1
```

结果摘要：

```text
resets = 0
height ≈ 0.420-0.422
vel_x ≈ 0.50-0.55
```

这说明中速前进链路也基本稳定。

### 5. 低台阶 smoke test

命令：

```bash
./isaaclab_waq.sh --mujoco-play \
  --policy logs/waq/bpx_waq_rough/2026-06-05_17-44-04/policy_jit.pt \
  --terrain stairs \
  --step-height 0.06 \
  --command-x 0.4 \
  --headless \
  --duration 5 \
  --print-interval 1
```

结果摘要：

```text
resets = 0
height 随台阶上升
vel_x ≈ 0.28-0.36
```

这不是证明台阶能力已经完全可靠，只说明 MuJoCo 接触场景下部署链路不再立即失真。

### 6. 使用 mjlab XML

命令：

```bash
./isaaclab_waq.sh --mujoco-play \
  --policy logs/waq/bpx_waq_rough/2026-06-05_17-44-04/policy_jit.pt \
  --xml /home/ubuntu/robot_rl/bpx_mjlab/src/bpx_mjlab/bpx/xmls/bpx.xml \
  --headless \
  --duration 3
```

结果和 repo XML 基本一致：

```text
resets = 0
height ≈ 0.424-0.429
vel_x ≈ 0.22-0.26 at cmd_x=0.3
```

这说明这次问题不是主要由 XML 路径或 meshdir 引起，而是 joint order 引起。

## 为什么这能让 sim2sim 变好

四足策略的动作向量不是普通的 12 个无关数字。每一维都绑定一个特定关节：

- 第 0 维如果训练时是 `fl_hip_roll_joint`，部署时也必须是 `fl_hip_roll_joint`。
- 第 4 维如果训练时是 `fl_hip_pitch_joint`，部署时不能被送到 `fr_hip_pitch_joint` 或 `fr_knee_joint`。
- joint pos / joint vel 的观测顺序也必须与策略训练时一致。

旧 runner 的问题是 action 和 observation 的排列都偏了。这样 policy 的闭环控制结构被打乱：

```text
policy 输出 A_i -> 错误关节执行
错误关节状态 -> 下一帧 observation
history 继续累积错误动力学
CENet 估计进入错误分布
actor 输出进一步异常
```

因此机器人会很快从稳定站立进入翻倒状态。

修复 joint order 后，策略终于在 MuJoCo 里看到了接近训练时语义的 joint pos/vel，并把动作送回正确关节。这个变化直接恢复了闭环控制结构，所以 sim2sim 表现明显改善。

## 当前还不能说明什么

这次修复只能说明：基础部署链路中的关节映射问题被修掉了。

它不能说明：

- 当前策略已经适合真机。
- 高台阶能力已经可靠。
- MuJoCo 与 IsaacLab/PhysX 完全一致。
- 所有 actuator/contact/friction 差异已经解决。

后续仍然需要逐项验证：

- 真实 motor kp/kd/effort limit。
- joint friction / armature。
- 接触摩擦。
- 足端碰撞体。
- IMU/关节编码器观测延迟和噪声。
- policy action rate 与真机控制频率。

## 推荐后续流程

### 1. 每个 checkpoint 都先重新导出

```bash
./isaaclab_waq.sh --waq-export \
  --checkpoint logs/waq/bpx_waq_rough/<run>/model_<iter>.pt \
  --control-profile legacy_mjlab_10hz
```

然后检查：

```bash
python -c "import json; d=json.load(open('logs/waq/bpx_waq_rough/<run>/policy_jit.json')); print(d['joint_order']); print(d['joint_names']); print(d['action_sign'])"
```

应该看到：

```text
type_major
all action_sign = 1.0
```

### 2. 先平地，再低台阶，再高台阶

平地低速：

```bash
./isaaclab_waq.sh --mujoco-play \
  --policy logs/waq/bpx_waq_rough/<run>/policy_jit.pt \
  --command-x 0.3 \
  --real-time \
  --interactive
```

平地中速：

```bash
./isaaclab_waq.sh --mujoco-play \
  --policy logs/waq/bpx_waq_rough/<run>/policy_jit.pt \
  --command-x 0.6 \
  --real-time \
  --interactive
```

低台阶：

```bash
./isaaclab_waq.sh --mujoco-play \
  --policy logs/waq/bpx_waq_rough/<run>/policy_jit.pt \
  --terrain stairs \
  --step-height 0.06 \
  --command-x 0.4 \
  --real-time \
  --interactive
```

高台阶不要直接上 12-16 cm。建议先确认：

```text
0.06 -> 0.08 -> 0.10 -> 0.12 -> 0.14 -> 0.16 m
```

每一步只变一个参数。

### 3. 如果又出现翻倒，先看 reset reason

新版 runner 会打印原因，例如：

```text
orientation_bad:gravity_z=...
base_height_low:...
qvel_high:...
```

不要直接改 reward。先判断是哪类问题：

- `orientation_bad`：姿态失稳，优先看 joint order、action sign、观测 frame、history layout。
- `base_height_low`：可能是策略本身低身位，也可能是 action scale/control profile 不匹配。
- `qvel_high`：可能是 actuator/contact 太激进，先查 PD/effort/contact。
- `non_finite_state`：通常是 MuJoCo 数值爆炸，先查 XML/contact/actuator。

## 注意事项

1. `--flip-knee` 不是默认修复方案。

   它只能用于验证 sign 假设。当前 BPX sim2sim 默认 action sign 全部为 `+1`。

2. `--joint-order metadata` 不一定安全。

   旧导出的 metadata 是 leg-major，已经证明会导致错误部署。新导出的 metadata 才是 type-major。

3. 不建议用降低 `action_scale_multiplier` 或 `clip_actions` 掩盖映射错误。

   这类参数会让机器人看起来不炸，但可能只是把错误动作压小。根因修正后，再讨论 action scale 才有意义。

4. 这次没有修改训练核心。

   `rough_env_cfg.py`、reward、terrain curriculum 没有被这次 sim2sim 修复改动。这一点很重要，因为它保证了我们是在验证部署链路，而不是又把训练问题和部署问题混在一起。

## 一句话总结

这次 sim2sim 变好，不是因为把狗“训得更保守”，也不是因为把动作强行裁小，而是因为 MuJoCo runner 终于按 IsaacLab 策略实际训练时的 type-major 关节顺序执行动作和构造观测，恢复了 policy 的正确闭环控制语义。
