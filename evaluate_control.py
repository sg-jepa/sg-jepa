"""Run bounded closed-loop evaluation for a released control policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("franka_basket", "arm_paddle_ball", "arm_catcher_ball"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--encoder-config", type=Path)
    parser.add_argument("--encoder-checkpoint", type=Path)
    parser.add_argument("--dinov2-root", type=Path)
    parser.add_argument("--menagerie-root", type=Path, required=True)
    parser.add_argument("--method", default="Semigroup-JEPA")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--render-size", type=int, default=256)
    parser.add_argument(
        "--paper-cohort",
        action="store_true",
        help="mark a complete held-out cohort run as a paper-metric reproduction attempt",
    )
    args = parser.parse_args()
    common = {
        "method": args.method,
        "policy_checkpoint": args.checkpoint,
        "policy_config": args.config,
        "world_model_checkpoint": args.encoder_checkpoint,
        "world_model_config": args.encoder_config,
        "dataset_path": args.data,
        "menagerie_root": args.menagerie_root,
        "output_dir": args.output_dir,
        "device": args.device,
        "dinov2_root": args.dinov2_root,
        "max_episodes": args.max_episodes,
        "episode_start": args.episode_start,
        "seed": args.seed,
        "render_size": args.render_size,
        "paper_cohort": args.paper_cohort,
    }
    if args.task == "franka_basket":
        from sg_jepa.evaluation.frozen_franka import run_frozen_franka_evaluation

        report = run_frozen_franka_evaluation(**common)
    elif args.task == "arm_paddle_ball":
        from sg_jepa.evaluation.frozen_paddle import run_frozen_paddle_evaluation

        report = run_frozen_paddle_evaluation(**common)
    else:
        from sg_jepa.evaluation.frozen_catcher import run_frozen_catcher_evaluation

        report = run_frozen_catcher_evaluation(**common)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
