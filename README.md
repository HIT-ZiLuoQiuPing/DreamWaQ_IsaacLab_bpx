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
