"""Train Original LeWM, Semigroup-JEPA, or DINO-WM from YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sg_jepa.baselines.training import run_dino_training
from sg_jepa.config import load_experiment_config
from sg_jepa.training import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--dinov2-root",
        type=Path,
        help="pinned DINOv2 checkout and weights (required by DINO-WM configs)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="cap optimizer steps for a bounded integration run",
    )
    args = parser.parse_args()
    payload = yaml.safe_load(args.config.read_text())
    is_dino = isinstance(payload, dict) and "predictor" in payload
    if is_dino:
        if args.dinov2_root is None:
            parser.error("DINO-WM training requires --dinov2-root")
        report = run_dino_training(
            args.config,
            args.data,
            args.output_dir,
            dinov2_root=args.dinov2_root,
            device=args.device,
            resume=args.resume,
            max_steps=args.max_steps,
        )
    else:
        report = run_training(
            load_experiment_config(args.config),
            args.data,
            args.output_dir,
            device=args.device,
            resume=args.resume,
            max_steps=args.max_steps,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
