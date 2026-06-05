# IsaacLab 与 mjlab 对齐差异总结

日期：2026-06-05

目标：总结 BPX DreamWaQ 从 `/home/ubuntu/robot_rl/bpx_mjlab` 迁移到 `/home/ubuntu/isaaclab_waq` 的过程中，发现的关键差异、这些差异对训练效果的影响，以及目前已经修正和仍需排查的部分。

## 结论

IsaacLab 版本并不是 mjlab 的一比一等价复制。最开始只对齐了部分可见配置，例如机器人控制参数、PPO/WAQ 网络结构、粗糙地形比例和部分奖励。后来在训练和 play 中发现，步态差异主要来自几类问题：

- 动作执行语义不同：IsaacLab 版曾经存在 actor 输出 `tanh` 和 `clip_actions=1.0`，导致 policy raw action 严重饱和。
- command 采样不同：mjlab 中有 `heading_command=True`、`rel_heading_envs=0.3`、`rel_forward_envs=0.75`，IsaacLab 初版没有完整复现。
- 脚部观测和奖励语义不同：mjlab 的 foot height/clearance 更贴近脚相对地形高度，IsaacLab 初版一度使用 toe 世界高度近似。
- termination 不同：IsaacLab 版曾加入 `base_height` termination，mjlab 没有同等项，这会把策略推向保守低身位或拖行。
- contact/collision 物理不同：MuJoCo XML 和 IsaacLab USD/PhysX 不是同一套接触系统，即使 reward 名字相同，实际优化目标也可能不同。

当前 `mjlab_parity_footobs_v1` 之后，步态已经明显改善，并且能跨较高台阶；这说明主要问题不是 IsaacLab 完全不能训练，而是此前训练语义没有真正对齐。

## 差异总表

| 类别 | mjlab | IsaacLab 曾经的问题 | 当前状态 | 影响 |
| --- | --- | --- | --- | --- |
| 物理引擎 | MuJoCo | Isaac Sim / PhysX | 仍不同 | 接触、摩擦、solver、碰撞体都可能影响步态 |
| 资产格式 | XML | USD | 仍不同 | 惯量、碰撞体、自碰撞、contact body 粒度可能不一致 |
| Actor 输出 | raw Gaussian mean | 曾经对 mean 做 `tanh` | 已移除 | `tanh` 会限制动作表达，和 rsl_rl 不一致 |
| Action clipping | `clip_actions: null` | 曾经 `clip_actions=1.0` | 已改为 `None` | 旧版 raw action 到 `3-5`，环境执行被裁剪，PPO 学习和真实执行不一致 |
| Command heading | `heading_command=True` | 曾经关闭 | 已恢复 | 影响直行/yaw 稳定性 |
| 前进命令比例 | `rel_forward_envs=0.75` | IsaacLab 原生 command 没有这个字段 | 已实现 forward-biased command | 影响是否集中学习向前行走 |
| 脚部 privileged obs | foot height/contact/force 等 | 初版缺失或不完整 | 已补 estimator/critic foot obs | 影响跨台阶、脚部抬高和接触判断 |
| Feet clearance | 脚相对地形语义 | 曾经近似 toe 世界高度 | 已改 terrain-relative | 旧版可能奖励错误的“抬脚” |
| Feet swing height | mjlab 有 | 初版没有等价项 | 已补全，并增加后脚专用项 | 帮助脚真正离地跨越 |
| Base height termination | 无同等硬终止 | 曾经加入 `base_height` termination | 已移除 | 旧版大量 base_height 终止，容易导致低姿态保命 |
| 反蠕动 reward | 无 | 曾经加入 `forward_velocity_error/no_forward_motion/crawl_penalty` | 已禁用 | 额外 reward 会破坏 parity，对比不干净 |
| 后腿拖地 | mjlab 中表现较好 | IsaacLab footobs 后仍出现后 calf/knee 拖地 | 已新增后 calf 触地惩罚 | 当前最新局部修正项 |
| Runner | rsl_rl/mjlab runner | 手写 DreamWaQ runner | 仍不同 | 优化器、日志、scheduler、advantage 细节可能有差异 |
| 训练速度 | 轻量 mjlab | IsaacLab/PhysX 更慢 | 仍不同 | 主要是仿真采样和脚部 ray/contact 开销 |

## 关键差异细节

### 1. 动作裁剪与 action 饱和

mjlab 保存配置中 `clip_actions: null`。IsaacLab 版曾经使用 `clip_actions=1.0`，同时早期 actor mean 还做过 `tanh`。

这造成一个严重问题：policy 计算出来的 raw action 可以很大，但实际送进环境时被 wrapper 裁剪。训练日志里曾出现：

- `Rollout/action |mean|` 到 `3.0-5.0`
- 实际速度仍跟不上命令
- play 中表现为小跳、拖脚或蠕动

这说明 PPO 学到的是“把动作打满”，不是细腻的关节控制。后来将 `clip_actions` 改为 `None` 后，`action |mean|` 降到约 `1.0`，步态明显改善。

### 2. Command 采样不一致

mjlab 中 command 有几个关键字段：

- `heading_command=True`
- `rel_heading_envs=0.3`
- `rel_forward_envs=0.75`
- `heading_control_stiffness=0.5`

IsaacLab 原生 `UniformVelocityCommandCfg` 没有完全一样的 `rel_forward_envs` 逻辑。早期 IsaacLab 版没有真正复现这个采样结构，导致训练样本中“向前正常行走”的比例和 mjlab 不一致。

后续实现了 `ForwardBiasedVelocityCommandCfg`，并恢复 heading command，才更接近 mjlab。

### 3. 脚部观测缺失或语义不一致

mjlab 中 critic/estimator 使用了脚部相关 privileged 信息，例如：

- `foot_height`
- `foot_air_time`
- `foot_contact`
- `foot_contact_forces`
- `height_scan`

IsaacLab 初版的 estimator/critic 主要依赖 base velocity 和 height scan，脚部信息不完整。对于楼梯和粗糙地形，策略需要知道脚相对地形的高度、触地状态和接触力，否则容易学到“身体往前蹭”而不是“抬脚跨越”。

当前已补充：

- `foot_height`
- `foot_air_time`
- `foot_contact`
- `foot_contact_forces`

并且将 `feet_clearance` 改为相对地形高度计算。

### 4. Feet clearance 从世界高度改为地形相对高度

旧 IsaacLab 版的 `feet_clearance` 更像使用 toe body 的世界 z 高度。这个在平地上可能还行，但在台阶/坡地上语义会偏：

- 台阶上脚的世界高度高，不等于脚已经抬过局部障碍。
- 身体姿态变化也可能改变 toe 世界高度，让 reward 被误导。

当前使用 height scanner 找脚附近的地形高度，计算：

```text
foot_height_above_terrain = foot_world_z - nearest_terrain_z
```

这更接近“脚有没有真正离开地面/跨过台阶”的目标。

### 5. Base height termination 的影响

IsaacLab 旧版加入过 `base_height` termination。训练数据中这个终止项一度达到 10%-30% 甚至更高。

这个终止项的问题是：它会把“低身位通过复杂地形”直接判死，但同时又可能让策略为了避免失败学得过于保守，尤其是在台阶和坡地上。mjlab 没有这个完全等价的硬终止项。

当前已移除 `base_height` termination，只保留 `base_height` reward。这样避免了大量 episode 因短时低身位直接终止，也让策略有机会通过 reward 学姿态。

### 6. 旧左右镜像姿态约束不是 trot 相位约束

早期使用过一个左右镜像姿态约束，它主要约束：

- 左前和右前镜像
- 左后和右后镜像

它不保证：

- 左前和右后同相
- 右前和左后同相
- 两组对角腿反相

因此它不是严格的 trot 相位奖励。此前观察到“前两条腿同步、后两条腿拖行”时，这种动作仍可能部分满足左右对称。该项已经从源码和训练配置中移除。后续没有继续加显式 trot 约束，是因为 footobs 版本已经自然出现了较好的步态，继续大改会破坏当前有效方向。

### 7. 后腿拖地问题

当前 `mjlab_parity_footobs_v1` 已经出现步态，也能跨高台阶，但 play 观察到两条后腿小腿/膝部拖地。

这说明主步态和跨越能力已经开始建立，但 reward 对后腿 calf 触地的惩罚仍偏弱。旧的 `undesired_contacts` 只对所有 calf 做轻微惩罚，权重较小。

最新局部修正：

- 增加 `hind_calf_contacts`
  - body: `hl_calf_link`, `hr_calf_link`
  - weight: `-0.55`
- `undesired_contacts`
  - `-0.10 -> -0.15`
- 增加 `hind_feet_swing_height`
  - target height: `0.14m`
  - weight: `-0.18`

这次没有改 observation/model shape，因此可以从 `mjlab_parity_footobs_v1_smoke` checkpoint 分支续训。

### 8. Terrain curriculum 的差异

mjlab 和 IsaacLab 的地形课程实现细节不同。IsaacLab 版曾经加入过：

- allowed max level gate
- command-distance success ratio
- hold steps
- early termination only demotion

这些逻辑虽然有助于稳定训练，但会让 parity 对比变得不干净。后来为了对齐 mjlab，又把 terrain curriculum 拉回更接近 mjlab 的版本。

当前需要注意：地形 mean 上升不等于策略真的掌握了地形。必须结合：

- play 观察是否抬腿
- `velocity error xy`
- `action_mean_abs`
- calf/knee contact
- timeout/failure composition

一起判断。

### 9. MuJoCo 与 PhysX 的物理差异

即使所有 reward 名字和权重一样，MuJoCo 和 PhysX 也不会完全等价：

- 接触求解器不同
- 摩擦模型不同
- solver iteration 不同
- mesh/primitive collision 不同
- 自碰撞处理不同
- timestep/substep 细节不同

四足步态对接触非常敏感。一个 calf 或 toe 的碰撞体形状差异，就足够让 policy 学出不同动作。

因此不能把所有差异都归因于代码没对齐，但也不能把所有问题都归因于 IsaacLab 框架。当前最合理的做法是：先保证训练语义对齐，再逐项检查资产和物理。

## 已经修正的关键问题

截至最新提交，已经做过这些修正：

- 去掉 actor mean 的 `tanh`。
- `clip_actions` 改为 `None`。
- 恢复 heading command。
- 实现 forward-biased command。
- 移除额外 anti-crawl reward，回到更干净的 parity。
- 去掉 `base_height` termination。
- 补齐脚部 privileged obs。
- `feet_clearance` 改为地形相对高度。
- 增加 `feet_swing_height`。
- 针对后腿拖地增加 `hind_calf_contacts` 和 `hind_feet_swing_height`。

## 仍需进一步确认的地方

这些还没有完全确认：

- BPX USD 与 mjlab XML 的惯量是否一致。
- toe/calf/thigh collision shape 是否一致。
- PhysX contact sensor 的 body 粒度是否能等价 mjlab geom/site contact。
- `height_scanner` 的采样点是否足够覆盖脚附近地形。
- 后腿拖地是否来自 reward 权重不足，还是来自 USD 后腿碰撞/惯量/关节轴差异。
- 手写 WAQ runner 与 mjlab rsl_rl runner 在 advantage、KL、scheduler、normalization 上是否仍有差异。

## 目前建议

当前不要再大改。因为 `mjlab_parity_footobs_v1` 已经比之前明显更接近 mjlab：

- 有步态；
- 可以跨更高台阶；
- action 不再严重饱和；
- base_height termination 已移除；
- 速度跟踪早期更健康。

下一步只建议做局部验证：

1. 从 `mjlab_parity_footobs_v1_smoke/model_1200.pt` 续训后腿拖地修正版。
2. 观察 `hind_calf_contacts` 是否下降。
3. 观察 play 中后膝/后小腿是否离地。
4. 如果仍拖地，只调 `hind_calf_contacts` 权重，不要同时动 command、terrain、actor 或主 reward。

建议命令：

```bash
./isaaclab_waq.sh --waq-train \
  --task Isaac-BPX-WAQ-Rough-v0 \
  --num_envs 1024 \
  --resume \
  --checkpoint logs/waq/bpx_waq_rough/2026-06-05_15-03-48_mjlab_parity_footobs_v1_smoke/model_1200.pt \
  --no_load_optimizer \
  --max_iterations 3000 \
  --run_name hind_calf_fix_from_1200 \
  --headless
```

## 简短判断

训练效果差距不是单一原因。早期 IsaacLab 版确实没有完整按 mjlab 的训练语义构建，尤其是动作裁剪、command、脚部观测/奖励和 termination。修正这些后，效果已经明显接近 mjlab。

剩下的差异更可能集中在：

- 后腿局部 reward 权重；
- USD/XML 资产和碰撞体差异；
- PhysX 与 MuJoCo 的接触求解差异；
- runner 细节差异。

因此后续应该继续小步排查，不应再一次性大改多个训练机制。
