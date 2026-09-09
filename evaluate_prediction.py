"""Evaluate latent prediction with a fitted or released state probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from sg_jepa.evaluation import (
    run_frozen_approach_evaluation,
    run_planar_probe_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--probe-config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--method", default="Semigroup-JEPA")
    parser.add_argument("--max-episodes", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=44)
    parser.add_argument("--dinov2-root", type=Path)
    parser.add_argument(
        "--task",
        choices=("right_triangle", "square", "approach_ball"),
        default="approach_ball",
    )
    parser.add_argument("--normalization-stats", type=Path)
    parser.add_argument(
        "--episode-manifest",
        type=Path,
        help="source-episode cohort manifest for planar paper evaluation",
    )
    parser.add_argument("--spin-dt", type=float, default=1.0 / 16.0)
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text())
    model_kind = (
        "dino"
        if isinstance(raw, dict) and ("predictor" in raw or raw.get("kind") == "dino_world_model")
        else "native"
    )
    if args.task == "approach_ball":
        report = run_frozen_approach_evaluation(
            method=args.method,
            model_kind=model_kind,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            probe_path=args.probe,
            probe_config_path=args.probe_config,
            dataset_path=args.data,
            output_dir=args.output_dir,
            device=args.device,
            max_episodes=args.max_episodes,
            horizon=args.horizon,
            dinov2_root=args.dinov2_root,
        )
    else:
        report = run_planar_probe_evaluation(
            task=args.task,
            method=args.method,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            probe_path=args.probe,
            probe_config_path=args.probe_config,
            dataset_path=args.data,
            output_dir=args.output_dir,
            device=args.device,
            max_episodes=args.max_episodes,
            horizon=args.horizon,
            dinov2_root=args.dinov2_root,
            normalization_path=args.normalization_stats,
            episode_manifest_path=args.episode_manifest,
            spin_dt=args.spin_dt,
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
