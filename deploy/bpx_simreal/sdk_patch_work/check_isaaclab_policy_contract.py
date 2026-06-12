#!/usr/bin/env python3
"""Check SDK config and TorchScript signature against IsaacLab WAQ export metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

try:
    import torch
except Exception:
    torch = None


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _nearly(a: object, b: object, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def _check_value(name: str, got: object, expected: object, ok: list[bool]):
    mark = "OK" if _nearly(got, expected) else "ERR"
    if mark != "OK":
        ok[0] = False
    print(f"[{mark}] {name}: got={got} expected={expected}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", default=None)
    parser.add_argument("--poses", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--policy", default=None)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    config_path = Path(args.config).expanduser() if args.config else root / "configs" / "real_config_working.yaml"
    poses_path = Path(args.poses).expanduser() if args.poses else root / "configs" / "poses.yaml"
    metadata_path = Path(args.metadata).expanduser() if args.metadata else root / "policy" / "bpx_dwaq_v2_simreal_metadata.json"
    policy_path = Path(args.policy).expanduser() if args.policy else root / "policy" / "bpx_dwaq_v2.pt"

    cfg = _load_yaml(config_path)
    poses = _load_yaml(poses_path) if poses_path.exists() else {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    contract = cfg.get("isaaclab_policy_contract", {})
    if not isinstance(contract, dict):
        raise RuntimeError(f"Missing isaaclab_policy_contract in {config_path}")

    ok = [True]
    print("=== ISAACLAB WAQ SIM2REAL CONTRACT ===")
    print(f"root    : {root}")
    print(f"config  : {config_path}")
    print(f"metadata: {metadata_path}")
    print(f"policy  : {policy_path}")
    print()

    _check_value("action_scale", contract.get("action_scale"), metadata["action_scale_scalar"], ok)
    _check_value("dwaq_history_length", contract.get("dwaq_history_length"), metadata["history_length"], ok)
    _check_value("actor_obs_single_frame_dim", contract.get("actor_obs_single_frame_dim"), metadata["num_actor_obs"], ok)
    _check_value("encoder_input_dim", contract.get("encoder_input_dim"), metadata["num_history_obs"], ok)
    dims = contract.get("supported_policy_input_dims", {})
    _check_value("supported.history_only", dims.get("history_only"), metadata["history_only_input_dim"], ok)
    _check_value("supported.current_plus_history", dims.get("current_plus_history"), metadata["current_plus_history_input_dim"], ok)

    scales = contract.get("obs_scales", {})
    command_scale = contract.get("command_scale", {})
    _check_value("obs_scales.ang_vel", scales.get("ang_vel"), 0.2, ok)
    _check_value("obs_scales.projected_gravity", scales.get("projected_gravity"), 1.0, ok)
    _check_value("obs_scales.joint_pos", scales.get("joint_pos"), 1.0, ok)
    _check_value("obs_scales.joint_vel", scales.get("joint_vel"), 0.05, ok)
    _check_value("obs_scales.last_action", scales.get("last_action"), 1.0, ok)
    _check_value("command_scale.vx", command_scale.get("vx"), 1.0, ok)
    _check_value("command_scale.vy", command_scale.get("vy"), 1.0, ok)
    _check_value("command_scale.wz", command_scale.get("wz"), 1.0, ok)

    print("\n[default_joint_pos]")
    default_cfg = contract.get("default_joint_pos", {})
    stand_pose = poses.get("poses", {}).get("stand_standard", {}).get("joints", {})
    for name, expected in metadata["default_joint_pos_by_name"].items():
        _check_value(f"contract.{name}", default_cfg.get(name), expected, ok)
        if stand_pose:
            _check_value(f"stand_standard.{name}", stand_pose.get(name), expected, ok)

    print("\n[torchscript]")
    if torch is None:
        ok[0] = False
        print("[ERR] torch is not importable in this Python environment")
    elif not policy_path.exists():
        ok[0] = False
        print(f"[ERR] policy file does not exist: {policy_path}")
    else:
        model = torch.jit.load(str(policy_path), map_location="cpu").eval()
        dim = int(metadata["history_only_input_dim"])
        x = torch.zeros(1, dim, dtype=torch.float32)
        with torch.no_grad():
            y = model(x)
        if isinstance(y, (tuple, list)):
            y = y[0]
        shape = tuple(y.detach().cpu().shape)
        finite = bool(torch.isfinite(y.detach().cpu()).all().item())
        mark = "OK" if shape == (1, int(metadata["num_actions"])) and finite else "ERR"
        if mark != "OK":
            ok[0] = False
        print(f"[{mark}] history_only dry-run: input=(1,{dim}) output={shape} finite={finite}")

    print("\nRESULT:", "OK" if ok[0] else "FAILED")
    return 0 if ok[0] else 2


if __name__ == "__main__":
    raise SystemExit(main())
