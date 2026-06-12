#!/usr/bin/env python3
"""Check the exported BPX sim2real TorchScript policy signature."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", nargs="?", default="bpx_dwaq_v2.pt")
    parser.add_argument("--metadata", default="bpx_dwaq_v2_simreal_metadata.json")
    parser.add_argument("--mode", choices=("auto", "history_only", "current_plus_history", "all"), default="auto")
    args = parser.parse_args()

    policy_path = Path(args.policy).expanduser()
    metadata_arg = Path(args.metadata).expanduser()
    if not policy_path.is_absolute():
        policy_path = Path.cwd() / policy_path
    if metadata_arg.is_absolute():
        metadata_path = metadata_arg
    elif args.metadata == parser.get_default("metadata"):
        metadata_path = policy_path.parent / metadata_arg
    else:
        metadata_path = Path.cwd() / metadata_arg

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    all_expected = [
        ("history_only", metadata["history_only_input_dim"]),
        ("current_plus_history", metadata["current_plus_history_input_dim"]),
    ]
    if args.mode == "all":
        expected = all_expected
    elif args.mode == "history_only":
        expected = [all_expected[0]]
    elif args.mode == "current_plus_history":
        expected = [all_expected[1]]
    elif "current_plus_history" in policy_path.name:
        expected = [all_expected[1]]
    else:
        expected = [all_expected[0]]

    print(f"Checking policy: {policy_path}")
    print(f"Metadata: {metadata_path}")
    model = torch.jit.load(str(policy_path), map_location="cpu").eval()
    for name, dim in expected:
        x = torch.zeros(1, int(dim), dtype=torch.float32)
        try:
            with torch.no_grad():
                y = model(x)
            y = y[0] if isinstance(y, (tuple, list)) else y
            y = y.detach().cpu()
            print(f"OK   {name:22s} input={dim:4d} -> output={tuple(y.shape)} finite={torch.isfinite(y).all().item()}")
        except Exception as exc:
            msg = str(exc).splitlines()
            print(f"FAIL {name:22s} input={dim:4d} -> {msg[-1] if msg else exc}")


if __name__ == "__main__":
    main()
