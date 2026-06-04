# BPX DreamWaQ IsaacLab 训练记录与当前判断

日期：2026-06-04

## 背景

目标是在独立工程 `/home/ubuntu/isaaclab_waq` 中实现 BPX 四足机器人 DreamWaQ IsaacLab 粗糙地形行走训练，不直接修改 `/home/ubuntu/IsaacLab`、`/home/ubuntu/unitree_rl_lab`。

已经完成的主线：

- 创建独立 IsaacLab extension：`isaaclab_waq`。
- 使用 BPX USD 资产注册任务：`Isaac-BPX-WAQ-Rough-v0`、`Isaac-BPX-WAQ-Rough-Play-v0`。
- 实现 DreamWaQ 风格 CENet + PPO runner。
- 添加 terrain curriculum、velocity curriculum、奖励项日志、termination 日志、console.log 保存。
- 修复 play 在显卡/驱动问题解决后可运行，并添加交互控制。
- 修复训练日志波动问题：随机化 episode length、原始 reward 日志、增大 episode buffer、添加 `mean_reward_per_step` 和 `completed_episode_count`。

## 最新训练数据

最新 run：

`logs/waq/bpx_waq_rough/2026-06-03_17-08-17/events.out.tfevents.1780477702.ubuntu-System-Product-Name.1650742.0`

最后 200 个点的主要统计：

| 指标 | 末值 | 尾部均值 | 判断 |
| --- | ---: | ---: | --- |
| `Train/mean_reward` | 38.59 | 49.47 | 仍有波动，但比旧日志更可信 |
| `Train/mean_episode_length` | 609.25 | 723.24 | 不是完全稳定走满 |
| `Train/mean_reward_per_step` | 0.0633 | 0.0680 | 每步质量一般 |
| `Episode_Termination/time_out` | 0.599 | 0.612 | 约 60% episode 能超时结束 |
| `Episode_Termination/base_height` | 0.401 | 0.388 | 约 39% 因高度过低失败 |
| `Curriculum/terrain_level_mean` | 3.238 | 3.245 | 已到中等地形 |
| `Curriculum/terrain_level_max` | 9 | 9 | 有环境已经到最高等级 |
| `Rollout/command_x` | 0.828 | 0.839 | 目标速度已经较高 |
| `Rollout/velocity_x` | 0.618 | 0.643 | 实际速度明显偏低 |
| `Metrics/base_velocity/error_vel_xy` | 0.356 | 0.389 | 速度误差偏大 |
| `Policy/mean_noise_std` | 0.206 | 0.206 | 探索已较低 |
| `Loss/learning_rate` | 1e-5 | 约 1.1e-5 | 学习率已压到下限 |

粗略结论：

- 这版策略不是完全不会走，但在 terrain mean 约 3.24 时仍有较高 `base_height` 失败。
- 实际速度约 `0.64 m/s`，命令约 `0.84 m/s`，说明策略用某种不够稳的方式追速度。
- 最后学习率已经贴到 `1e-5`，继续从 `model_40000.pt` 直接训练，改变步态的概率不高。

## 关于小跳步态

当前判断：`leg_symmetry` 不太像主因，但它也不能完全排除。

原因：

- `leg_symmetry` 的贡献尾部均值约 `-0.00046`，数值非常小。
- mjlab 里也有类似 `leg_symmetry`，权重同样是 `-0.25`，但 mjlab 效果更好。
- 这个项约束的是左右镜像，例如左前和右前、左后和右后，不直接要求前后脚同步。

更可疑的因素：

- 我们 IsaacLab 版此前加入了固定相位 `feet_gait`，但 policy observation 里没有 episode phase。
- 训练启动现在会随机化 episode length，这让固定相位 gait reward 的目标相位对 policy 不可观测。
- `FEET_BODY_NAMES` 原来使用 regex，gait offset 依赖 body order。如果解析顺序不是 `FL, FR, HL, HR`，固定 gait 可能奖励错误腿对。
- `air_time_variance` 只惩罚四只脚 air/contact 时间方差，不区分相位，可能偏好四脚同步。

## 与 mjlab 版本的主要差异

mjlab 路径：

`/home/ubuntu/robot_rl/bpx_mjlab`

关键差异：

- mjlab 没有固定相位 `feet_gait`、`all_feet_air`、`feet_contact_count` 这一组我们后来添加的 IsaacLab 步态项。
- mjlab 初始姿态更舒展：`base z=0.42`，`hip_pitch=0.6`，`knee=-1.2`。
- IsaacLab 当前初始姿态更深蹲：`base z=0.45`，`hip_pitch=0.8`，`knee=-1.5`。
- mjlab action scale 约为 `0.38`，IsaacLab 当前为 `0.20`。
- mjlab PPO 配置更接近 `entropy_coef=0.01`、`init_std=1.0`；IsaacLab 此前偏保守。

这些差异不一定每个都是问题。当前先改最确定的奖励逻辑，不同时大改姿态、动作尺度和地形，避免难以判断原因。

## 本次代码改动

已修改：

- 固定脚顺序：`FL, FR, HL, HR`。
- 禁用固定相位 `feet_gait`。
- 禁用 `air_time_variance`。
- 添加 phase-free 接触奖励：
  - `diagonal_trot_contact_reward`：奖励对角两脚支撑。
  - `bad_two_foot_contact_pattern`：惩罚前两脚、后两脚、左侧两脚、右侧两脚这类 bounding/pacing 两脚支撑。
- 稍微提高探索：
  - `init_noise_std: 0.55 -> 0.65`
  - `min_noise_std: 0.15 -> 0.18`
  - `entropy_coef: 0.002 -> 0.006`
  - `desired_kl: 0.01 -> 0.02`
- 新增 resume 参数：
  - `--no_load_optimizer`：只加载模型权重，不加载旧 optimizer/LR。
  - `--reset_policy_std`：把加载模型的 action std 重置为当前配置值。

## 推荐训练方式

不建议从 `model_40000.pt` 直接继续。

理由：

- 末尾学习率已到 `1e-5`。
- `base_height` 失败约 39%。
- 小跳步态可能已经固化。

更建议从较早 checkpoint 分支验证：

- 稳妥分支：`model_6000.pt`
- 保留更多速度能力的分支：`model_16000.pt`

短测试：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 256 \
  --resume \
  --checkpoint logs/waq/bpx_waq_rough/2026-06-03_17-08-17/model_6000.pt \
  --reset_curriculum_on_resume \
  --no_load_optimizer \
  --reset_policy_std \
  --max_iterations 1000 \
  --run_name gait_v2_smoke_from_6000 \
  --headless
```

正式分支：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint logs/waq/bpx_waq_rough/2026-06-03_17-08-17/model_6000.pt \
  --reset_curriculum_on_resume \
  --no_load_optimizer \
  --reset_policy_std \
  --max_iterations 30000 \
  --run_name gait_v2_from_6000 \
  --headless
```

如果 `model_6000.pt` 太慢，可以试 `model_16000.pt`，但它更可能已经带有小跳倾向。

## 观察重点

训练时重点看：

- `Reward/step/diagonal_trot_contact` 是否逐渐上升。
- `Reward/step/bad_two_foot_contact` 是否下降。
- `Reward/step/all_feet_air` 是否降低。
- `Episode_Termination/base_height` 是否低于旧 run。
- `Rollout/velocity_x` 是否接近 `Rollout/command_x`。
- `Policy/mean_noise_std` 和 `Loss/learning_rate` 是否过早贴底。

play 时重点看：

- 是否仍然前后脚一起跳。
- 是否出现对角腿交替。
- 在 stairs/rough 地形上是否为了保命降低速度。

## 不确定项

- 需要一次短训练确认新接触奖励没有把策略推向过慢或拖脚。
- 需要 play 观察确认“显式脚顺序”是否和实际 contact sensor 顺序一致。
- IsaacLab 与 mjlab 的物理、actuator、初始姿态差异可能仍然重要，但这次没有同时大改。
- 如果新奖励改善步态但速度仍慢，再考虑动作尺度、初始姿态和速度课程。

## 2026-06-04 10:10 gait_v2_from_6000 复查

Run：

`logs/waq/bpx_waq_rough/2026-06-04_10-10-19_gait_v2_from_6000`

它从 `2026-06-03_17-08-17/model_6000.pt` 续训，使用了：

- `--reset_curriculum_on_resume`
- `--no_load_optimizer`
- `--reset_policy_std`

console 中确认：

- policy std 重置到 `0.650`
- curriculum step offset 为 `0`

训练到 `model_8000.pt`，等价于从新奖励分支继续了约 2000 iteration。

最后 200 个点：

| 指标 | 尾部均值 | 判断 |
| --- | ---: | --- |
| `Train/mean_reward_per_step` | 0.0808 | 比旧 run 末尾 0.068 好 |
| `Train/mean_episode_length` | 925 | 平地存活较好 |
| `Episode_Termination/time_out` | 0.866 | 大多数 episode 能跑满 |
| `Episode_Termination/base_height` | 0.133 | 明显低于旧 run 末尾约 0.388 |
| `Curriculum/terrain_level_mean` | 0.0 | 还没进入地形课程 |
| `Rollout/command_x` | 0.421 | 当前仍是初始速度段 |
| `Rollout/velocity_x` | 0.345 | 平地速度跟踪尚可但不满 |
| `Policy/mean_noise_std` | 0.512 | 探索仍在 |
| `Loss/learning_rate` | 约 1e-4 | 没有贴死到 1e-5 |
| `Reward/step/diagonal_trot_contact` | 0.0626 | 约等于 39% 对角两脚支撑 |
| `Reward/step/bad_two_foot_contact` | -0.0742 | 约等于 21% 错误两脚支撑 |
| `Reward/step/all_feet_air` | -0.0129 | 约等于 0.6% 四脚离地，纯跳倾向下降 |

当前判断：

- 不建议立刻改代码。
- 不建议从头训练。
- 这条分支值得继续，但至少要训到地形课程开始，即从当前 `model_8000.pt` 再训 3000-4000 iteration。
- 由于当前 checkpoint 的 `iter=8000`，但实际新课程进度只有约 2000 iteration，如果重新启动训练，必须手动设置 `--curriculum_offset_iterations 2000`。否则默认 offset 会按 8000 iteration 计算，地形和速度课程会过快；如果再次使用 `--reset_curriculum_on_resume`，课程又会从 0 重新开始。

建议短续训：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint logs/waq/bpx_waq_rough/2026-06-04_10-10-19_gait_v2_from_6000/model_8000.pt \
  --curriculum_offset_iterations 2000 \
  --max_iterations 4000 \
  --run_name gait_v2_continue_terrain_check \
  --headless
```

不要加：

- `--reset_curriculum_on_resume`
- `--no_load_optimizer`
- `--reset_policy_std`

这次是继续同一分支，应保留当前 optimizer 和 policy std。
