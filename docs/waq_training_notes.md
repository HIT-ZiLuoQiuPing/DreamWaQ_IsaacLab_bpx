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

## 2026-06-04 12:23 mjlab_parity_v1 复查

Run：

`logs/waq/bpx_waq_rough/2026-06-04_12-23-27_mjlab_parity_v1`

事件文件：

`events.out.tfevents.1780547014.ubuntu-System-Product-Name.2272110.0`

训练到约 `7315` iteration。尾部 200 个点的核心情况：

| 指标 | 尾部均值 | 判断 |
| --- | ---: | --- |
| `Train/mean_reward` | 158.65 | 数值较高，但不能单独说明能过复杂地形 |
| `Train/mean_episode_length` | 941.05 | 大多数 episode 能跑很久 |
| `Curriculum/terrain_level_mean` | 1.32 | 平均地形等级仍很低 |
| `Curriculum/terrain_level_max` | 9.00 | 有环境已经到最高等级 |
| `Curriculum/terrain_move_up_rate` | 0.147 | 有升级 |
| `Curriculum/terrain_move_down_rate` | 0.203 | 降级更频繁 |
| `Curriculum/terrain_mean_distance` | 3.67 | 明显低于固定升级阈值 6m |
| `Curriculum/terrain_promotion_distance` | 6.00 | 原 mjlab 风格绝对距离阈值过硬 |
| `Rollout/velocity_x` | 0.272 | 实际前进速度偏低 |
| `Metrics/base_velocity/error_vel_xy` | 0.297 | 速度误差仍偏大 |

## 2026-06-05 gait_speed_v3 反蠕动分支

触发原因：

- `2026-06-04_16-04-43` 的真实 play 观察显示，机器人更多是在低身位蠕动，而不是形成清晰四足步态。
- 训练日志中地形等级稳步上升，但速度跟随变差：terrain level 与 `velocity_x` 呈明显负相关，`action_mean_abs` 随地形等级升高而增大。
- 约 15% episode 因 `base_height` 终止，说明策略在困难地形上更偏向保命和低姿态，而不是稳定跨越。

本次改动不是继续纯 mjlab parity，而是在 parity 基础上做针对性分支：

- 新增 `ForwardBiasedVelocityCommandCfg`，实现 mjlab 风格 `rel_forward_envs=0.75`，其中 75% 非站立环境使用独立前进命令范围。
- 初始前进命令提高到 `forward_lin_vel_x=(0.30, 0.90)`，避免早期长期低速导致策略学会蠕动。
- 速度课程提前：
  - `0`: forward x `(0.30, 0.90)`
  - `3000*16`: `(0.40, 1.10)`
  - `7000*16`: `(0.50, 1.25)`
  - `12000*16`: `(0.60, 1.45)`
  - 后续逐步到 `(0.80, 1.80)`
- 启用速度跟踪约束：
  - `forward_velocity_error`
  - `no_forward_motion`
  - `crawl_penalty`
- 加强前进精细速度奖励：`track_forward_velocity_fine` 从 `1.4/std=0.25` 改为 `2.0/std=0.22`。
- 加强脚部抬高奖励：`feet_clearance` 从 `0.08/0.12m` 改为 `0.14/0.14m`。
- 放慢地形课程并提高升级门槛：
  - `level_step_interval: 600*16 -> 900*16`
  - `promotion_distance_ratio: 0.75 -> 0.80`
  - `promotion_command_ratio: 0.60 -> 0.72`
  - `minimum_promotion_distance: 3.0 -> 3.5`
  - `min_level_hold_steps: 100*16 -> 150*16`
- 日志新增 `Forward cmd x range`，用于确认前进命令课程是否真正生效。

预期现象：

- 前 1000-3000 轮地形平均等级可能比上一版上升更慢，这是有意的。
- `Rollout/cmd_x vs vel_x` 的差距应该比上一版缩小。
- `Rollout/action |mean|` 不应继续随 terrain level 明显上升；如果仍然上升，说明 crawl penalty 仍不够或动作尺度/物理配置还存在问题。
- `Reward terms` 中新增的 `forward_velocity_error`、`no_forward_motion`、`crawl_penalty` 应该在早期明显，后期下降。

训练建议：

- 最稳妥：从 0 训练 `3000-5000` 轮短测，先 play 看是否形成步态。
- 可选：从 `2026-06-04_16-04-43/model_3200.pt` 分支续训，但该 checkpoint 已有蠕动倾向，建议使用 `--no_load_optimizer` 并把课程 offset 调低。

从 0 短测：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --max_iterations 5000 \
  --run_name gait_speed_v3_from_scratch_smoke \
  --headless
```

从旧 checkpoint 分支：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint logs/waq/bpx_waq_rough/2026-06-04_16-04-43/model_3200.pt \
  --no_load_optimizer \
  --curriculum_offset_iterations 1500 \
  --max_iterations 5000 \
  --run_name gait_speed_v3_branch_from_3200 \
  --headless
```

## 2026-06-05 mjlab parity foot-observation 修正

触发原因：

- `2026-06-05_09-55-36` 的真实 play 仍然表现为小跳/拖行，不会主动抬腿跨楼梯。
- 日志中 terrain level 能升到中高等级，但 `velocity error xy` 随等级升高明显变大，`action |mean|` 接近 `4-5`，说明策略在用大动作硬蹭。
- 图像观察显示当前 `leg_symmetry` 只保证左右镜像，并不保证对角 trot 相位。

本次改动：

- `clip_actions` 从 `1.0` 改为 `None`，对齐 mjlab 的 `clip_actions: null`。
- command 恢复 mjlab 语义：
  - `heading_command=True`
  - `heading_control_stiffness=0.5`
  - `rel_heading_envs=0.3`
  - `rel_forward_envs=0.75`
  - 主速度课程恢复 mjlab 阶段表。
- 地形课程恢复更接近 mjlab：
  - 固定距离升级，`promotion_distance_ratio=0.75`
  - `demote_only_early_termination=False`
  - 不再使用额外 command-ratio 升级门槛和等级开放节奏。
- 去掉 gait_speed_v3 额外项：
  - `forward_velocity_error`
  - `no_forward_motion`
  - `crawl_penalty`
- 去掉 `base_height` termination，避免它把策略推向低速保守/拖行局部最优。
- bad orientation 改为 mjlab 的约 `70deg` 限制。
- 补齐 CENet estimator 和 critic 的脚部 privileged observation：
  - `foot_height`
  - `foot_air_time`
  - `foot_contact`
  - `foot_contact_forces`
- 将 `feet_clearance` 改为脚相对地形高度的 L2 penalty，并补 `feet_swing_height`。

重要影响：

- estimator/critic 输入维度已经变化，旧 checkpoint 不兼容。
- 这版必须从 0 训练，不能从 `2026-06-05_09-55-36` 或更早 checkpoint 继续。

建议从 0 短测：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --max_iterations 5000 \
  --run_name mjlab_parity_footobs_v1_smoke \
  --headless
```

如果 3000-5000 轮 play 仍然前后腿同步/小跳，则下一步才应该加入显式对角 trot contact 约束，而不是继续调速度奖励。
| `Episode_Termination/base_height` | 0.110 | 仍有约 11% 高度失败 |
| `Rollout/action_mean_abs` | 4.515 | policy raw mean 远超 `clip_actions=1.0`，动作大量饱和 |

结论：

- 不是“完全不升级”，而是升级后更频繁降级，所以均值卡在约 1.3。
- 原始 `promotion_distance_ratio=0.75` 对 8m patch 等价于 6m。当前平均速度约 `0.27m/s`，20 秒只能走约 `5.4m`，很难稳定满足 6m timeout。
- `allowed_max_level=9` 从一开始就完全放开，导致局部环境能冲到高等级，但平均等级随降级逻辑掉回低等级。
- 当前 CENet estimator target 与 mjlab 不完全一致：mjlab 是 `base_lin_vel + height_scan`，当前版本曾额外加入脚高度、触地和接触力，增加了估计器重构负担。

本次修正：

- estimator target 收窄为 `base_lin_vel + height_scan`，更接近 mjlab 的 DreamWaQ 监督目标。
- 地形升级增加命令完成率判据：除了绝对 6m，也允许 timeout 且移动距离达到 `0.60 * command_distance`，最低仍需 3m。
- 地形降级改为只在早退失败时触发，避免“跑满但速度低”的 episode 反复降级。
- 地形允许等级改为逐步开放：`level_step_interval = 600 * 16`，避免刚开始就允许冲到 9 级。
- 单个 env 升降级后至少保持 `100 * 16` step，减少等级来回震荡。
- console/TensorBoard 增加课程诊断：
  - `terrain_mean_command_distance`
  - `terrain_distance_success_rate`
  - `terrain_command_success_rate`
  - console 中显示 `Terrain dist/cmd_dist` 和 `Terrain abs/cmd success`

训练建议：

- 不建议继续 `2026-06-04_12-23-27_mjlab_parity_v1` 的旧 checkpoint，因为 estimator 维度已经从 `62` 改成 `38`，checkpoint 和新网络结构不匹配。
- 这次改动后应从头训练，或至少从结构匹配的新 checkpoint 开始。

建议短测：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 256 \
  --max_iterations 1000 \
  --run_name terrain_curriculum_fix_smoke \
  --headless
```

正式训练：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --max_iterations 50000 \
  --run_name terrain_curriculum_fix_v1 \
  --headless
```

## 2026-06-05 hind calf drag fix

现象：

- `mjlab_parity_footobs_v1_smoke` 已经出现可用步态，并且能跨较高台阶。
- 真实 play 中两条后腿仍有明显拖地，后小腿/膝部会触碰地面。

本次只做局部 reward 调整，不改命令、地形、观测维度、网络结构或 PPO：

- 保留全体 calf 触地惩罚，并将 `undesired_contacts` 权重从 `-0.10` 小幅提高到 `-0.15`。
- 新增 `hind_calf_contacts`，只惩罚 `hl_calf_link` 和 `hr_calf_link` 触地，权重 `-0.55`。
- 新增 `hind_feet_swing_height`，只对后脚摆动高度给轻微约束：
  - target height `0.14m`
  - weight `-0.18`

训练建议：

- 这次没有改 observation/model shape，可以从 `mjlab_parity_footobs_v1_smoke` 的 checkpoint 分支继续训。
- 建议先从最近 checkpoint 分支续训 1000-3000 轮观察 play，不建议从 0 重训。

## 2026-06-05 remove leg symmetry and constrain hind pose

现象：

- `leg_symmetry` 约束的是左右镜像姿态，不是 `FL+HR / FR+HL` 的 trot 相位。
- play 中前腿步态已经较正常，但后腿有明显内收，后小腿/膝部拖地。
- 这个问题应优先解决后腿姿态，再解决后腿触地。

本次修改：

- 禁用 `leg_symmetry`，避免继续奖励“左右镜像但前后腿动作不协调”的局部解。
- 新增 `hind_hip_roll_pose`：
  - 只约束 `hl_hip_roll_joint`、`hr_hip_roll_joint` 接近默认站姿。
  - 权重 `-0.35`。
- 新增 `hind_leg_pose`：
  - 轻微约束两条后腿全部 6 个关节接近默认姿态。
  - 权重 `-0.05`。
- 加强后腿拖地相关项：
  - `hind_feet_swing_height`: `-0.18 -> -0.24`
  - `hind_calf_contacts`: `-0.55 -> -0.75`

说明：

- 这次仍然不加显式 trot 相位奖励，避免破坏当前已经出现的自然步态。
- 这次没有改 observation/model shape，checkpoint 结构兼容。
- 如果要重新训练，建议 run name 使用 `no_leg_symmetry_hind_pose_v1`。
