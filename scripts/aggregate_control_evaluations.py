"""Aggregate five no-plot control evaluations from the subset workflow."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sg_jepa.train_utils import atomic_json

_SUCCESS_FIELD_BY_REPORT_KIND = {
    "franka_frozen_checkpoint_qualitative_evaluation": "success",
    "paddle_frozen_checkpoint_qualitative_evaluation": "paper_success",
    "catcher_frozen_checkpoint_qualitative_evaluation": "captured",
}


def aggregate_control_evaluations(input_root: Path, *, expected_seeds: int = 5) -> dict[str, Any]:
    """Aggregate task-specific paper success metrics across rollout seeds."""

    if expected_seeds <= 0:
        raise ValueError("expected_seeds must be positive")

    reports = []
    for path in sorted(input_root.glob("seed_*/evaluation.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "pass":
            raise ValueError(f"evaluation did not pass: {path}")
        reports.append((path, payload))
    if len(reports) != expected_seeds:
        raise ValueError(f"expected {expected_seeds} seed reports, found {len(reports)}")

    report_kinds = {str(report.get("kind")) for _, report in reports}
    if len(report_kinds) != 1:
        raise ValueError(f"evaluation reports have inconsistent kinds: {sorted(report_kinds)}")
    report_kind = report_kinds.pop()
    try:
        success_field = _SUCCESS_FIELD_BY_REPORT_KIND[report_kind]
    except KeyError as exc:
        raise ValueError(f"unsupported evaluation report kind: {report_kind}") from exc

    per_seed: list[dict[str, Any]] = []
    per_gravity: dict[float, list[bool]] = defaultdict(list)
    all_success: list[bool] = []
    seen_seeds: set[int] = set()
    for path, report in reports:
        records = report.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(f"evaluation has no records: {path}")
        missing = [index for index, record in enumerate(records) if success_field not in record]
        if missing:
            raise ValueError(
                f"evaluation records in {path} are missing {success_field!r}; "
                f"first missing index: {missing[0]}"
            )
        seed = int(report["evaluation"]["seed"])
        if seed in seen_seeds:
            raise ValueError(f"duplicate evaluation seed {seed}: {path}")
        seen_seeds.add(seed)

        successes = [bool(record[success_field]) for record in records]
        all_success.extend(successes)
        for record, success in zip(records, successes, strict=True):
            per_gravity[float(record["gravity"])].append(success)
        per_seed.append(
            {
                "path": str(path.resolve()),
                "seed": seed,
                "episodes": len(records),
                "success_rate": sum(successes) / len(successes),
            }
        )

    return {
        "schema_version": 1,
        "kind": "paired_control_seed_aggregate",
        "claim": "paper_subset_reproduction_attempt",
        "evaluation_report_kind": report_kind,
        "success_field": success_field,
        "metric_name": reports[0][1].get("paper_metric_name", success_field),
        "seed_count": len(reports),
        "episodes": len(all_success),
        "success_rate": sum(all_success) / len(all_success),
        "per_seed": per_seed,
        "per_gravity": [
            {
                "gravity": gravity,
                "episodes": len(values),
                "success_rate": sum(values) / len(values),
            }
            for gravity, values in sorted(per_gravity.items())
        ],
        "status": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, default=5)
    args = parser.parse_args()

    payload = aggregate_control_evaluations(args.input_root, expected_seeds=args.expected_seeds)
    atomic_json(payload, args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
