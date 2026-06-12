# BPX Sim2Real Deployment Export

> Current status: this package was exported from the old 15-frame checkpoint and
> is not the final artifact for the fixed 5-frame BPX upper deploy stack. Retrain
> with `history_length = 5`, then re-export so `bpx_dwaq_v2.pt` accepts `[1, 225]`.
>
> The training/export source now targets the fixed upper stack contract:
> `history_length=5`, `encoder_input_dim=225`, `current_plus_history=270`,
> `action_scale=0.25`, command scale `[2.0, 2.0, 0.25]`, and default pose
> hip pitch `0.8`, knee `-1.5`. The files in this folder remain old artifacts
> until a new checkpoint is trained and exported.

This directory contains SDK-ready TorchScript policies exported from:

`logs/waq/bpx_waq_rough/2026-06-09_18-19-51/model_42000.pt`

## Files

- `bpx_dwaq_v2.pt`
  - Single-input TorchScript policy for SDK/UI `history_only` mode.
  - Input: `[1, 675]` = `15 * 45` frame-major observation history.
  - Output: `[1, 12]` action vector in BPX type-major joint order.
- `bpx_dwaq_v2_current_plus_history.pt`
  - Single-input TorchScript policy for SDK/UI `current_plus_history` mode.
  - Input: `[1, 720]` = `45 + 15 * 45`.
  - Output: `[1, 12]`.
- `bpx_dwaq_v2_simreal_metadata.json`
  - Full export metadata: joint order, default pose, action scale, observation scale, history layout.
- `real_config_policy_patch.yaml`
  - Old 15-frame contract values for the old checkpoint only. Do not copy this
    patch into the fixed 5-frame `bpx_simreal_v6` stack.

## Old Artifact Contract

The old exported policy files currently in this directory use:

- `history_length = 15`
- `actor_obs_dim = 45`
- `history_only_input_dim = 675`
- `current_plus_history_input_dim = 720`
- `history input layout = frame_major_oldest_to_newest`
- internal CENet layout converted to `term_major_oldest_to_newest`
- `action_scale = 0.3799544386804864`
- `base_ang_vel` observation scale `0.2`
- velocity command observation scale `[1.0, 1.0, 1.0]`
- default joint pose: hip roll `0.0`, hip pitch `0.6`, knee `-1.2`

Do not run this old 15-frame model with the fixed upper-stack contract values `history_length=5`,
`input_dim=225/270`, `action_scale=0.25`, or command scale `[2.0, 2.0, 0.25]`.

## Verification

The exported wrappers were dry-run checked:

```bash
bpx_dwaq_v2.pt                      input=(1,675) output=(1,12)
bpx_dwaq_v2_current_plus_history.pt input=(1,720) output=(1,12)
```

The `history_only` wrapper was also compared against the original dual-input
`policy_jit.pt` with random inputs after layout conversion; `max_abs_diff = 0.0`.
