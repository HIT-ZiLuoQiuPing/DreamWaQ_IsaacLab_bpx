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
