# mjlab_parity_v1 对齐说明

日期：2026-06-04

目标：把 IsaacLab 版 BPX DreamWaQ 尽量改成 `/home/ubuntu/robot_rl/bpx_mjlab` 的对齐版本，用于判断问题来自框架差异还是策略/配置差异。

## 已对齐项目

### BPX 控制与初始姿态

参考：

`/home/ubuntu/robot_rl/bpx_mjlab/src/bpx_mjlab/bpx/bpx_constants.py`

IsaacLab 已改为：

- `EFFORT_LIMIT = 30.0`
- `ARMATURE = 0.005`
- `NATURAL_FREQ = 10 * 2pi`
- `DAMPING_RATIO = 2.0`
- `STIFFNESS = ARMATURE * NATURAL_FREQ ** 2`
- `DAMPING = 2 * DAMPING_RATIO * ARMATURE * NATURAL_FREQ`
- `ACTION_SCALE = 0.25 * EFFORT_LIMIT / STIFFNESS`
- 初始姿态：
  - `base z = 0.42`
  - `hip_roll = 0.0`
  - `hip_pitch = 0.6`
  - `knee = -1.2`

### PPO / WAQ 配置

参考：

`/home/ubuntu/robot_rl/bpx_mjlab/src/bpx_mjlab/rl_cfg.py`

IsaacLab 已改为：

- `num_steps_per_env = 16`
- `num_learning_epochs = 5`
- `learning_rate = 1e-3`
- `entropy_coef = 0.01`
- `desired_kl = 0.01`
- `init_noise_std = 1.0`
- `latent_dim = 24`
- encoder `(512, 256)`
- decoder `(256, 512)`
- actor/critic `(512, 256, 128)`
- `save_interval = 200`

### Actor 输出

mjlab 使用 rsl_rl Gaussian distribution 的 raw mean。IsaacLab 原先对 actor mean 做了 `tanh`，现在已移除，让动作由 env wrapper clip。

### Terrain

参考：

`/home/ubuntu/robot_rl/bpx_mjlab/src/bpx_mjlab/env_cfgs.py`

IsaacLab 已改为：

- `flat: 0.12`
- `pyramid_stairs: 0.20`
- `pyramid_stairs_inv: 0.08`
- `hf_pyramid_slope: 0.25`
- `hf_pyramid_slope_inv: 0.20`
- `random_rough: 0.08`
- `wave_terrain: 0.07`
- stairs:
  - `step_width = 0.35`
  - `step_height_range = (0.04, 0.16)`
- slopes:
  - `slope_range = (0.0, 0.85)`
- `max_init_terrain_level = 1`

### Command Curriculum

IsaacLab 已改为 mjlab 阶段：

- step `0`: `lin_vel_x=(-0.20, 0.75)`, `lin_vel_y=(-0.10, 0.10)`, `ang_vel_z=(-0.25, 0.25)`
- step `6000 * 16`: `(-0.30, 0.95)`, `(-0.18, 0.18)`, `(-0.40, 0.40)`
- step `12000 * 16`: `(-0.35, 1.10)`, `(-0.22, 0.22)`, `(-0.48, 0.48)`
- step `18000 * 16`: `(-0.45, 1.25)`, `(-0.30, 0.30)`, `(-0.60, 0.60)`
- step `26000 * 16`: `(-0.50, 1.40)`, `(-0.30, 0.30)`, `(-0.60, 0.60)`
- step `36000 * 16`: `(-0.55, 1.60)`, `(-0.32, 0.32)`, `(-0.65, 0.65)`
- step `45000 * 16`: `(-0.60, 1.80)`, `(-0.35, 0.35)`, `(-0.70, 0.70)`

### Terrain Curriculum

IsaacLab 已取消此前额外 warmup/hold/streak gate，改为接近 mjlab：

- `promotion_distance_ratio = 0.75`
- `demotion_command_ratio = 0.5`
- `consecutive_successes = 1`
- `warmup_steps = 0`
- `min_level_hold_steps = 0`
- `demote_only_early_termination = False`

### Reward

IsaacLab 已回退到更接近 mjlab 的结构：

- `track_lin_vel_xy`: weight `3.5`, std `0.35`
- `track_ang_vel_z`: weight `3.2`, std `0.40`
- `track_forward_velocity_fine`: weight `1.4`, std `0.25`
- `track_lateral_velocity_fine`: weight `1.2`, std `0.16`
- `track_yaw_velocity_fine`: weight `1.4`, std `0.25`
- `forward_lateral_drift`: weight `-1.5`
- `forward_yaw_drift`: weight `-1.2`
- `leg_symmetry`: weight `-0.25`
- `ang_vel_xy`: weight `-0.05`
- `action_rate`: weight `-0.12`
- `feet_air_time`: weight `0.2`
- `feet_clearance target_height = 0.12`
- `undesired/calf contact`: weight `-0.1`
- `termination`: weight `-25.0`

已从训练配置中移除/禁用之前额外加入的 gait shaping：

- `no_forward_motion`
- `forward_velocity_error`
- `diagonal_trot_contact`
- `bad_two_foot_contact`
- `all_feet_air`
- `feet_contact_count`
- `air_time_variance`

### Critic Privileged Observation

mjlab critic 里包含：

- `base_lin_vel`
- `height_scan`
- `foot_height`
- `foot_air_time`
- `foot_contact`
- `foot_contact_forces`

IsaacLab 已补充等价项：

- `foot_height_body`
- `foot_air_time`
- `foot_contact`
- `foot_contact_forces`

## 仍非完全一致的地方

这些差异需要在结果分析时保留：

- mjlab 使用 MuJoCo，IsaacLab 使用 Isaac Sim/PhysX，接触模型不同。
- mjlab contact sensor 是 geom/site 粒度；IsaacLab 当前主要是 body/contact sensor 近似。
- mjlab 使用 `MjlabOnPolicyRunner` 和 rsl_rl 原生结构；IsaacLab 版仍是手写 DreamWaQ runner。
- mjlab 的 base velocity task 默认 reward/termination 细节可能还有隐藏差异；当前只按可见配置对齐。
- IsaacLab USD 资产与 mjlab XML 资产可能存在惯量、碰撞体、关节轴、接触几何差异。

## 训练建议

这是控制、reward、critic obs 和 actor 输出都变化后的新 baseline，不建议加载旧 checkpoint。

建议从头训练：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --max_iterations 50000 \
  --run_name mjlab_parity_v1 \
  --headless
```

先做短测：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 128 \
  --max_iterations 5 \
  --run_name mjlab_parity_v1_smoke \
  --headless
```

判断重点：

- 初始几百轮是否能避免立刻摔倒。
- `Rollout/action |mean|` 是否还长期接近 1.0。
- `Rollout/cmd_x vs vel_x` 是否比旧版更接近。
- `Episode_Termination/base_height` 是否快速下降。
- `Terrain level mean/max` 是否像 mjlab 一样能稳定推进。

## 本地验证

已执行：

```bash
/home/ubuntu/miniconda3/envs/isaaclab_bpx/bin/python scripts/waq/train.py \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 4 \
  --max_iterations 1 \
  --run_name mjlab_parity_v1_smoke_cpu_local \
  --headless \
  --device cpu
```

结果：

- 环境创建成功。
- terrain curriculum 初始化成功，`Terrain level mean/max: 0.25/1`。
- DreamWaQ observation/critic/estimator 维度能通过 runner 初始化。
- 完成 1 个 rollout 和 PPO update。
- CUDA/GPU 相关报错来自当前工具环境没有可见 GPU；CPU smoke 足以验证配置没有直接的 body/sensor/reward 名称错误。
