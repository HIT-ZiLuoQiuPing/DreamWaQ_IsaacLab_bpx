# IsaacLab WAQ BPX

Independent IsaacLab extension for BPX quadruped rough-terrain DreamWaQ training.

## Quick Start

```bash
conda activate isaaclab_bpx
./isaaclab_waq.sh --install
./isaaclab_waq.sh --waq-train --task Isaac-BPX-WAQ-Rough-v0 --num_envs 64 --max_iterations 2 --headless
./isaaclab_waq.sh --waq-play --task Isaac-BPX-WAQ-Rough-Play-v0 --checkpoint <model.pt>
```

Training writes TensorBoard events, checkpoints, config snapshots, and `console.log` under
`logs/waq/bpx_waq_rough/<run_name>/`.

For faster iteration during reward tuning:

```bash
./isaaclab_waq.sh --waq-train --task Isaac-BPX-WAQ-Rough-v0 --num_envs 1024 --num_steps_per_env 8 --ppo_epochs 3 --height_scan_resolution 0.25 --height_scan_update_stride 3 --headless
```

Interactive play with a fixed terrain cap and follow camera:

```bash
./isaaclab_waq.sh --waq-play --task Isaac-BPX-WAQ-Rough-Play-v0 --checkpoint <model.pt> --gui --interactive --follow_camera --terrain_profile rough --terrain_level 5 --command_x 0.6
```

During interactive play use `W/S` for forward speed, `A/D` for lateral velocity, `Q/E` for yaw, `F` for fast forward, `C` for crawl, `Space` to stop, and `R` to reset to a 0.6 m/s forward command.

Use `--terrain_profile stairs --terrain_level 5` or `--terrain_profile slope --terrain_level 7` to force a specific play terrain instead of randomly landing on flat terrain.

The rough-terrain curriculum starts from easy rows, but the final rows include 12-16 cm stairs, taller boxes, stronger roughness, and steeper slopes. To continue the faster/harder curriculum from a trained checkpoint:

```bash
./isaaclab_waq.sh --waq-train --task Isaac-BPX-WAQ-Rough-v0 --num_envs 1024 --resume --reset_curriculum_on_resume --checkpoint logs/waq/bpx_waq_rough/2026-06-03_11-20-22/model_15000.pt --max_iterations 25000 --run_name harder_terrain_speedup_curriculum_reset --headless
```

## MuJoCo Sim2sim

Export a trained DreamWaQ checkpoint to a TorchScript deployment policy:

```bash
./isaaclab_waq.sh --waq-export --checkpoint logs/waq/bpx_waq_rough/<run>/model_44000.pt
```

This writes `policy_jit.pt` and `policy_jit.json` next to the checkpoint. Run it in MuJoCo on flat terrain:

```bash
./isaaclab_waq.sh --mujoco-play --policy logs/waq/bpx_waq_rough/<run>/policy_jit.pt --real-time --interactive
```

Use `W/S` for forward speed, `A/D` for lateral velocity, `Q/E` for yaw, `Space` to stop, and `R` to reset the command. The MuJoCo runner uses the BPX IsaacLab action order by default: all hip-roll joints, all hip-pitch joints, then all knee joints (`--joint-order type_major`). Re-export old checkpoints after pulling this change so `policy_jit.json` records the same type-major metadata.

If the robot stays stable but immediately falls over or crouches, first verify the MuJoCo model and PD layer without the policy:

```bash
./isaaclab_waq.sh --mujoco-play --zero-action --duration 10 --real-time --debug-obs
```

Then inspect policy observations and actions:

```bash
./isaaclab_waq.sh --mujoco-play --policy logs/waq/bpx_waq_rough/<run>/policy_jit.pt --command-x 0.3 --debug-obs --real-time
```

The MuJoCo runner has safety resets for deployment debugging. If the policy flips over, the log prints the reset reason instead of continuing to report meaningless velocities from an upside-down state.

The MuJoCo player defaults to `--actuator-mode position`, which matches mjlab's `BuiltinPositionActuatorCfg`: the policy outputs position targets and MuJoCo applies the built-in actuator force limit. The old external torque-PD path is still available for comparison:

```bash
./isaaclab_waq.sh --mujoco-play --zero-action --actuator-mode torque_pd --duration 10 --real-time --debug-obs
```

For ablation only, compare `--joint-order metadata`, `--joint-order leg_major`, `--joint-order type_major`, and `--joint-order alphabetical`, or test one sign hypothesis at a time with `--flip-hip-pitch` / `--flip-knee`. These flags are diagnostic switches, not the default deployment path.

The BPX MuJoCo XML from the mjlab tree can also be used directly; missing floor/actuators are injected by the runner:

```bash
./isaaclab_waq.sh --mujoco-play --policy logs/waq/bpx_waq_rough/<run>/policy_jit.pt --xml /home/ubuntu/robot_rl/bpx_mjlab/src/bpx_mjlab/bpx/xmls/bpx.xml --real-time --interactive
```

The CENet history layout defaults to `--history-layout term_major`, matching IsaacLab's observation manager: each observation term is flattened over history first, then the terms are concatenated. `--history-layout frame_major` is kept only as an ablation switch.

A simple stair scene is also available:

```bash
./isaaclab_waq.sh --mujoco-play --policy logs/waq/bpx_waq_rough/<run>/policy_jit.pt --terrain stairs --step-height 0.08 --real-time --interactive
```
