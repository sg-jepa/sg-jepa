"""Paddle contact bookkeeping and paper policy-evaluation metrics."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SOURCE_CONTACT_GAP_FRAMES = 6
DEFAULT_MAX_TILT_THETA = math.pi / 6.0
TILT_VECTOR_SCALE = 0.5

# Final Figure 3(d) criterion used by the paper-level five-seed aggregator.
# The historical simulator gate stored post-warmup flight cycles using a 0.30 m
# diagnostic threshold.  The final paper aggregator deliberately rescored those
# raw cycles at 0.28 m and required one qualifying bounce.
PAPER_METRIC_NAME = "relaxed_one_bounce_apex_0p28"
PAPER_WARMUP_FRAMES = 16
PAPER_CONTACT_DEBOUNCE_FRAMES = 3
PAPER_MINIMUM_FLIGHT_GAP_FRAMES = 5
PAPER_MINIMUM_VERTICAL_CLEARANCE_M = 0.15
PAPER_MINIMUM_APEX_M = 0.28
PAPER_SUCCESS_CRITERION: dict[str, Any] = {
    "name": PAPER_METRIC_NAME,
    "minimum_qualified_bounces": 1,
    "minimum_flight_gap_frames": PAPER_MINIMUM_FLIGHT_GAP_FRAMES,
    "minimum_vertical_clearance_m": PAPER_MINIMUM_VERTICAL_CLEARANCE_M,
    "minimum_apex_m": PAPER_MINIMUM_APEX_M,
    "require_interior_apex": True,
    "require_full_horizon": True,
    "require_no_floor_contact": True,
}


@dataclass
class ContactCounter:
    """Count distinct paddle contacts exactly like the production simulator.

    A new contact requires a false-to-true interval transition and a gap of at
    least six recorded frame indices from the previous counted event.
    """

    min_gap_frames: int = SOURCE_CONTACT_GAP_FRAMES
    previous_interval_paddle_contact: bool = False
    last_paddle_contact_frame: int = -1000
    paddle_contact_frames: list[int] = field(default_factory=list)
    floor_contact_frame: int | None = None
    wall_contact_intervals: int = 0
    interval_count: int = 0

    def observe(
        self,
        frame_index: int,
        *,
        paddle_contact: bool,
        floor_contact: bool,
        wall_contact: bool,
    ) -> bool:
        frame_index = int(frame_index)
        paddle_contact = bool(paddle_contact)
        floor_contact = bool(floor_contact)
        wall_contact = bool(wall_contact)
        counted = bool(
            paddle_contact
            and not self.previous_interval_paddle_contact
            and frame_index - self.last_paddle_contact_frame >= self.min_gap_frames
        )
        if counted:
            self.paddle_contact_frames.append(frame_index)
            self.last_paddle_contact_frame = frame_index
        if floor_contact and self.floor_contact_frame is None:
            self.floor_contact_frame = frame_index
        if wall_contact:
            self.wall_contact_intervals += 1
        self.previous_interval_paddle_contact = paddle_contact
        self.interval_count += 1
        return counted

    @property
    def paddle_contact_count(self) -> int:
        return len(self.paddle_contact_frames)

    @property
    def floor_failure(self) -> bool:
        return self.floor_contact_frame is not None

    @property
    def source_success(self) -> bool:
        return self.paddle_contact_count >= 2 and not self.floor_failure


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _rms(vectors: Sequence[Sequence[float]]) -> float:
    values = [float(value) for vector in vectors for value in vector]
    return math.sqrt(_mean([value * value for value in values])) if values else 0.0


def _vector_l2(vector: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in vector))


def _subtract(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(left) - float(right) for left, right in zip(a, b, strict=False)]


def contact_onsets(
    contact: Sequence[bool],
    *,
    debounce_frames: int = PAPER_CONTACT_DEBOUNCE_FRAMES,
) -> list[int]:
    """Return debounced false-to-true contact indices used by the paper gate."""

    if debounce_frames <= 0:
        raise ValueError("debounce_frames must be positive")
    values = [bool(value) for value in contact]
    kept: list[int] = []
    for frame, value in enumerate(values):
        rising = value and (frame == 0 or not values[frame - 1])
        if rising and (not kept or frame - kept[-1] >= debounce_frames):
            kept.append(frame)
    return kept


def score_paper_true_flights(
    *,
    ball_z: Sequence[float],
    paddle_z: Sequence[float],
    paddle_contact: Sequence[bool],
    floor_contact: Sequence[bool],
    requested_transitions: int,
) -> dict[str, Any]:
    """Apply the final Figure 3(d) Paddle success criterion to one rollout.

    State traces contain the initial state plus one state after every observed
    transition. Contact traces contain one boolean per observed transition.
    This matches the source gate's indexing and the final paper aggregator's
    relaxed one-bounce rethresholding.
    """

    requested_transitions = int(requested_transitions)
    if requested_transitions <= 0:
        raise ValueError("requested_transitions must be positive")
    contacts = [bool(value) for value in paddle_contact]
    floors = [bool(value) for value in floor_contact]
    observed = len(contacts)
    if len(floors) != observed:
        raise ValueError("paddle_contact and floor_contact must have equal lengths")
    if len(ball_z) != observed + 1 or len(paddle_z) != observed + 1:
        raise ValueError("ball_z and paddle_z traces must each contain T+1 values")
    ball = [float(value) for value in ball_z]
    paddle = [float(value) for value in paddle_z]
    if not all(math.isfinite(value) for value in (*ball, *paddle)):
        raise ValueError("ball_z and paddle_z traces must contain only finite values")

    onsets = contact_onsets(contacts)
    cycles: list[dict[str, Any]] = []
    for left, right in zip(onsets, onsets[1:], strict=False):
        if left < PAPER_WARMUP_FRAMES:
            continue
        z = ball[left : right + 1]
        clearance = [ball[index] - paddle[index] for index in range(left, right + 1)]
        apex_offset = max(range(len(z)), key=z.__getitem__)
        apex_frame = left + apex_offset
        gap = right - left
        maximum_clearance = max(clearance)
        apex = z[apex_offset]
        failures: list[str] = []
        if gap < PAPER_MINIMUM_FLIGHT_GAP_FRAMES:
            failures.append("flight_gap_below_5_frames")
        if maximum_clearance < PAPER_MINIMUM_VERTICAL_CLEARANCE_M:
            failures.append("vertical_clearance_below_0p15m")
        if not left < apex_frame < right:
            failures.append("apex_not_interior_to_contacts")
        if apex < PAPER_MINIMUM_APEX_M:
            failures.append("apex_below_0p28m")
        cycles.append(
            {
                "start_contact_frame": left,
                "end_contact_frame": right,
                "gap_frames": gap,
                "maximum_vertical_clearance_m": maximum_clearance,
                "apex_frame": apex_frame,
                "apex_z_m": apex,
                "qualified": not failures,
                "failure_reasons": failures,
            }
        )

    qualified = [cycle for cycle in cycles if cycle["qualified"]]
    full_horizon = observed == requested_transitions
    floor_failure = any(floors)
    paper_success = bool(full_horizon and not floor_failure and qualified)
    failures: list[str] = []
    if not full_horizon:
        failures.append("incomplete_horizon")
    if floor_failure:
        failures.append("floor_contact")
    if not qualified:
        failures.append("no_qualified_paper_true_flight_cycle")
    return {
        "metric_name": PAPER_METRIC_NAME,
        "criterion": dict(PAPER_SUCCESS_CRITERION),
        "requested_transitions": requested_transitions,
        "observed_transitions": observed,
        "full_horizon": full_horizon,
        "floor_contact": floor_failure,
        "floor_contact_frames": [index for index, value in enumerate(floors) if value],
        "contact_onsets": onsets,
        "completed_post_warmup_cycles": len(cycles),
        "qualified_true_flight_cycles": len(qualified),
        "qualified_ratio": len(qualified) / max(1, len(cycles)),
        "minimum_qualified_apex_m": (
            None if not qualified else min(float(cycle["apex_z_m"]) for cycle in qualified)
        ),
        "cycles": cycles,
        "paper_success": paper_success,
        "passed": paper_success,
        "failure_reasons": failures,
    }


def action_metric_vector(action: Sequence[float]) -> list[float]:
    """Map a raw simulator action to the wrap-continuous policy coordinates."""

    if len(action) != 6:
        raise ValueError(f"Expected raw [g,dx,dy,dz,phi,theta], got {len(action)} values")
    _gravity, dx, dy, dz, phi, theta = (float(value) for value in action)
    return [
        dx,
        dy,
        dz,
        math.sin(theta) * math.cos(phi) / TILT_VECTOR_SCALE,
        math.sin(theta) * math.sin(phi) / TILT_VECTOR_SCALE,
    ]


def action_smoothness_metrics(
    actions: Iterable[Sequence[float]],
    *,
    max_tilt_theta: float = DEFAULT_MAX_TILT_THETA,
) -> dict[str, float | int]:
    raw = [[float(value) for value in action] for action in actions]
    vectors = [action_metric_vector(action) for action in raw]
    first = [_subtract(vectors[index], vectors[index - 1]) for index in range(1, len(vectors))]
    second = [_subtract(first[index], first[index - 1]) for index in range(1, len(first))]
    saturated = [
        any(abs(value) >= 1.0 - 1e-6 for value in action[1:4])
        or action[5] >= float(max_tilt_theta) - 1e-6
        for action in raw
    ]
    first_norms = [_vector_l2(vector) for vector in first]
    second_norms = [_vector_l2(vector) for vector in second]
    return {
        "executed_action_count": len(raw),
        "action_first_difference_rms": _rms(first),
        "action_second_difference_rms": _rms(second),
        "mean_action_change_l2": _mean(first_norms),
        "mean_action_acceleration_l2": _mean(second_norms),
        "max_action_change_l2": max(first_norms, default=0.0),
        "action_saturation_rate": _mean([float(value) for value in saturated]),
    }


def episode_metrics(
    *,
    counter: ContactCounter,
    actions: Iterable[Sequence[float]],
    frames_executed: int,
    max_frames: int,
    fps: float,
    ball_z: Sequence[float],
    paddle_z: Sequence[float],
    paddle_contact: Sequence[bool],
    floor_contact: Sequence[bool],
    max_tilt_theta: float = DEFAULT_MAX_TILT_THETA,
) -> dict[str, Any]:
    """Return one finite metric record for a closed-loop episode."""

    frames_executed = int(frames_executed)
    max_frames = int(max_frames)
    fps = float(fps)
    if frames_executed < 0 or max_frames <= 0 or fps <= 0.0:
        raise ValueError(f"Invalid episode lengths/fps: {frames_executed=}, {max_frames=}, {fps=}")
    contact_frames = list(counter.paddle_contact_frames)
    first_contact = contact_frames[0] if contact_frames else None
    last_contact = contact_frames[-1] if contact_frames else None
    end_frame = min(frames_executed, max_frames)
    juggling_duration = (
        max(0.0, (end_frame - first_contact) / fps) if first_contact is not None else 0.0
    )
    contact_span = (
        max(0.0, (last_contact - first_contact) / fps)
        if first_contact is not None and last_contact is not None
        else 0.0
    )
    dropped = counter.floor_failure
    full_horizon = bool(not dropped and frames_executed >= max_frames)
    true_flight_gate = score_paper_true_flights(
        ball_z=ball_z,
        paddle_z=paddle_z,
        paddle_contact=paddle_contact,
        floor_contact=floor_contact,
        requested_transitions=max_frames,
    )
    metrics: dict[str, Any] = {
        "distinct_paddle_contacts": counter.paddle_contact_count,
        # Evaluation stops at the first floor contact, so every distinct event
        # counted here belongs to the uninterrupted pre-drop juggling streak.
        "consecutive_paddle_contacts_before_drop": counter.paddle_contact_count,
        "first_paddle_contact_frame": first_contact,
        "last_paddle_contact_frame": last_contact,
        "floor_contact_frame": counter.floor_contact_frame,
        "wall_contact_intervals": counter.wall_contact_intervals,
        "survival_frames": end_frame,
        "survival_duration_seconds": end_frame / fps,
        "juggling_duration_seconds": juggling_duration,
        "contact_span_seconds": contact_span,
        "floor_contact": dropped,
        "drop": dropped,
        "full_horizon_no_drop": full_horizon,
        "paper_success": true_flight_gate["paper_success"],
        "paper_metric_name": PAPER_METRIC_NAME,
        "true_flight_gate": true_flight_gate,
        # Retained as a diagnostic only; this is not the paper success metric.
        "source_compatible_success": counter.source_success,
        "evaluation_terminated_early_on_floor_contact": bool(
            dropped and frames_executed < max_frames
        ),
    }
    metrics.update(action_smoothness_metrics(actions, max_tilt_theta=max_tilt_theta))
    return metrics


RATE_FIELDS = (
    "floor_contact",
    "drop",
    "full_horizon_no_drop",
    "paper_success",
    "source_compatible_success",
    "evaluation_terminated_early_on_floor_contact",
)

MEAN_FIELDS = (
    "distinct_paddle_contacts",
    "consecutive_paddle_contacts_before_drop",
    "wall_contact_intervals",
    "survival_frames",
    "survival_duration_seconds",
    "juggling_duration_seconds",
    "contact_span_seconds",
    "executed_action_count",
    "action_first_difference_rms",
    "action_second_difference_rms",
    "mean_action_change_l2",
    "mean_action_acceleration_l2",
    "max_action_change_l2",
    "action_saturation_rate",
)


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"episodes": len(records)}
    for field_name in RATE_FIELDS:
        values = [float(bool(record[field_name])) for record in records]
        summary[f"{field_name}_rate"] = _mean(values)
        summary[f"{field_name}_count"] = int(sum(values))
    for field_name in MEAN_FIELDS:
        summary[f"mean_{field_name}"] = _mean([float(record[field_name]) for record in records])
    return summary
