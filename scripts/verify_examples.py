"""Validate GPU, dataset, and checkpoint contracts for the example workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from sg_jepa.config import load_experiment_config
from sg_jepa.data import validate_trajectory_dataset
from sg_jepa.models import build_world_model
from sg_jepa.train_utils import load_training_checkpoint


def _print(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))


def _gpu(_args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("the example workflow requires an NVIDIA CUDA GPU")
    torch.cuda.init()
    _print(
        {
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "pytorch": torch.__version__,
            "status": "pass",
        }
    )


def _data(args: argparse.Namespace) -> None:
    report = validate_trajectory_dataset(
        args.path,
        expected_task=args.task,
        expected_split="all",
        sample_episodes=args.sample_episodes,
    )
    expected = {
        "episodes": args.episodes,
        "train_episodes": args.train_episodes,
        "test_episodes": args.test_episodes,
        "frames_min": 64,
        "frames_max": 64,
    }
    mismatches = {
        key: {"actual": report[key], "expected": value}
        for key, value in expected.items()
        if report[key] != value
    }
    if mismatches:
        raise ValueError(f"dataset contract mismatch: {mismatches}")
    manifest_path = args.path / "generation_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("task") != args.task or int(manifest.get("episodes", -1)) != args.episodes:
        raise ValueError("generation manifest task or episode count differs")
    report["manifest"] = str(manifest_path.resolve())
    _print(report)


def _world_model(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    report_path = output / "train_report.json"
    checkpoint_path = output / "checkpoint.pt"
    report = json.loads(report_path.read_text())
    config = load_experiment_config(args.config)
    checkpoint = load_training_checkpoint(checkpoint_path)
    if report.get("status") != "pass" or report.get("strict_reload") != "pass":
        raise ValueError("world-model report did not pass")
    if not str(report.get("device", "")).startswith("cuda"):
        raise ValueError("world-model report was not produced on CUDA")
    if int(report.get("steps_after", -1)) < args.minimum_steps:
        raise ValueError("world-model checkpoint has too few optimizer steps")
    if args.completed_epochs is not None and int(report.get("completed_epochs", -1)) != (
        args.completed_epochs
    ):
        raise ValueError("world-model completed epoch count differs")
    if checkpoint.get("config") != config.to_dict():
        raise ValueError("world-model checkpoint configuration differs")
    if "scheduler" not in checkpoint:
        raise ValueError("world-model checkpoint lacks scheduler state")
    model = build_world_model(config.model).to("cuda")
    model.load_state_dict(checkpoint["model"], strict=True)
    if not all(
        torch.isfinite(value).all()
        for value in model.state_dict().values()
        if torch.is_floating_point(value)
    ):
        raise ValueError("world-model checkpoint contains non-finite tensors")
    _print(
        {
            "checkpoint": str(checkpoint_path),
            "completed_epochs": report["completed_epochs"],
            "config": config.name,
            "lr_schedule": report.get("lr_schedule"),
            "steps": report["steps_after"],
            "strict_cuda_load": "pass",
            "status": "pass",
        }
    )


def _status_report(args: argparse.Namespace) -> None:
    payload = json.loads(args.path.read_text())
    if payload.get("status") != "pass":
        raise ValueError(f"report did not pass: {args.path}")
    if args.minimum_steps is not None and int(payload.get("steps_after", -1)) < (
        args.minimum_steps
    ):
        raise ValueError(f"report has fewer than {args.minimum_steps} optimizer steps")
    _print(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    gpu = commands.add_parser("gpu")
    gpu.set_defaults(func=_gpu)

    data = commands.add_parser("data")
    data.add_argument("--path", type=Path, required=True)
    data.add_argument("--task", required=True)
    data.add_argument("--episodes", type=int, required=True)
    data.add_argument("--train-episodes", type=int, required=True)
    data.add_argument("--test-episodes", type=int, required=True)
    data.add_argument("--sample-episodes", type=int, default=8)
    data.set_defaults(func=_data)

    world_model = commands.add_parser("world-model")
    world_model.add_argument("--output", type=Path, required=True)
    world_model.add_argument("--config", type=Path, required=True)
    world_model.add_argument("--minimum-steps", type=int, default=1)
    world_model.add_argument("--completed-epochs", type=int)
    world_model.set_defaults(func=_world_model)

    report = commands.add_parser("report")
    report.add_argument("--path", type=Path, required=True)
    report.add_argument("--minimum-steps", type=int)
    report.set_defaults(func=_status_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
