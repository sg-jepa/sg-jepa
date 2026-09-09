"""Train the paper MLP state probe on a frozen native or DINO encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sg_jepa.evaluation.state_probe import ProbeTrainConfig, run_probe_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("right_triangle", "square", "approach_ball"),
        required=True,
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", type=Path, help="world-model train_val_split.json")
    parser.add_argument("--dinov2-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--temporal-window", type=int)
    parser.add_argument("--max-train-windows", type=int, default=50_000)
    parser.add_argument("--max-val-windows", type=int, default=50_000)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    config = ProbeTrainConfig(
        temporal_window=args.temporal_window,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        feature_batch_size=args.feature_batch_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    report = run_probe_training(
        task=args.task,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        dataset_path=args.data,
        output_dir=args.output_dir,
        device=args.device,
        dinov2_root=args.dinov2_root,
        split_path=args.split,
        resume=args.resume,
        config=config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
