#!/usr/bin/env python3
"""
XWK BPX safe DWAQ model runner skeleton.

This file documents the same policy-input logic used by the UI.  Do not run
policy on the ground until:
1) zero/stand/sit PD tests pass,
2) IMU calibration is completed,
3) q_raw <-> q_sim mapping is verified by hand,
4) robot is hanging/supporting for first policy test.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "real_config_working.yaml"


def _policy_contract() -> dict:
    with CONFIG.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    contract = cfg.get("isaaclab_policy_contract", {})
    return contract if isinstance(contract, dict) else {}


POLICY_CONTRACT = _policy_contract()
OBS_DIM = int(POLICY_CONTRACT.get("actor_obs_single_frame_dim", 45))
HISTORY_LEN = int(POLICY_CONTRACT.get("dwaq_history_length", 5))
HIST_DIM = OBS_DIM * HISTORY_LEN
CURRENT_PLUS_HIST_DIM = OBS_DIM + HIST_DIM


def build_policy_input(current_obs: List[float], obs_history_flat: List[float], mode: str) -> List[float]:
    """Build TorchScript input for the two supported DWAQ export signatures."""
    if len(current_obs) != OBS_DIM:
        raise ValueError(f"current_obs must be {OBS_DIM} dims, got {len(current_obs)}")
    if len(obs_history_flat) != HIST_DIM:
        raise ValueError(f"obs_history_flat must be {HIST_DIM} dims, got {len(obs_history_flat)}")

    if mode == "history_only":
        return list(obs_history_flat)
    if mode == "current_plus_history":
        return list(current_obs) + list(obs_history_flat)
    raise ValueError(f"unknown policy mode: {mode}")


def main():
    print("XWK BPX model runner skeleton")
    print(f"Expected config: {CONFIG}")
    print("Supported DWAQ TorchScript signatures:")
    print(f"  history_only:          {HIST_DIM} dims")
    print(f"  current_plus_history:  {CURRENT_PLUS_HIST_DIM} dims")
    print("Use app/xwk_joint_lab_ui.py or scripts/07_check_policy_signature.sh to auto-detect the loaded policy.")


if __name__ == "__main__":
    main()
