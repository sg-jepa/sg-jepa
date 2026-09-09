"""Train a diffusion policy on frozen Semigroup-JEPA or DINOv2 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sg_jepa.control.config import UNIFORM_BATCH_SAMPLING
from sg_jepa.control.training import run_policy_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--world-model-config", type=Path)
    parser.add_argument("--world-model-checkpoint", type=Path)
    parser.add_argument("--dinov2-root", type=Path)
    parser.add_argument("--split", type=Path, help="world-model train_val_split.json")
    parser.add_argument("--resume", type=Path, help="schema-v2 policy training checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument(
        "--batch-sampling-override",
        choices=[UNIFORM_BATCH_SAMPLING],
        help=(
            "use uniform sampling for a bounded smoke cohort instead of a "
            "source-ID-keyed paper sampler"
        ),
    )
    args = parser.parse_args()
    report = run_policy_training(
        args.config,
        args.data,
        args.output_dir,
        world_model_config=args.world_model_config,
        world_model_checkpoint=args.world_model_checkpoint,
        dinov2_root=args.dinov2_root,
        split_path=args.split,
        resume=args.resume,
        device=args.device,
        max_steps=args.max_steps,
        max_episodes=args.max_episodes,
        batch_sampling_override=args.batch_sampling_override,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
