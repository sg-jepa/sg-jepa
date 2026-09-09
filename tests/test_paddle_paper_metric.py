from __future__ import annotations

import pytest

from sg_jepa.envs.paddle_metrics import (
    PAPER_METRIC_NAME,
    contact_onsets,
    score_paper_true_flights,
)


def _paper_cycle(
    *,
    observed: int = 32,
    left: int = 16,
    right: int = 24,
    apex_frame: int = 20,
    apex_z: float = 0.35,
    paddle_apex_z: float = 0.10,
    requested: int | None = None,
    floor_frame: int | None = None,
) -> dict[str, object]:
    ball_z = [0.10] * (observed + 1)
    paddle_z = [0.10] * (observed + 1)
    ball_z[apex_frame] = apex_z
    paddle_z[apex_frame] = paddle_apex_z
    paddle_contact = [False] * observed
    paddle_contact[left] = True
    paddle_contact[right] = True
    floor_contact = [False] * observed
    if floor_frame is not None:
        floor_contact[floor_frame] = True
    return score_paper_true_flights(
        ball_z=ball_z,
        paddle_z=paddle_z,
        paddle_contact=paddle_contact,
        floor_contact=floor_contact,
        requested_transitions=observed if requested is None else requested,
    )


def test_paper_metric_accepts_one_post_warmup_true_flight() -> None:
    result = _paper_cycle(apex_z=0.28, paddle_apex_z=0.13)

    assert result["metric_name"] == PAPER_METRIC_NAME
    assert result["paper_success"] is True
    assert result["passed"] is True
    assert result["qualified_true_flight_cycles"] == 1
    assert result["cycles"][0]["qualified"] is True


@pytest.mark.parametrize(
    ("overrides", "failure"),
    [
        ({"right": 20, "apex_frame": 18}, "flight_gap_below_5_frames"),
        (
            {"apex_z": 0.29, "paddle_apex_z": 0.15},
            "vertical_clearance_below_0p15m",
        ),
        ({"apex_frame": 16}, "apex_not_interior_to_contacts"),
        ({"apex_z": 0.279}, "apex_below_0p28m"),
    ],
)
def test_paper_metric_rejects_cycles_below_each_threshold(
    overrides: dict[str, object], failure: str
) -> None:
    result = _paper_cycle(**overrides)

    assert result["paper_success"] is False
    assert failure in result["cycles"][0]["failure_reasons"]


def test_paper_metric_requires_full_horizon_and_no_floor_contact() -> None:
    incomplete = _paper_cycle(requested=33)
    floor = _paper_cycle(floor_frame=31)

    assert incomplete["paper_success"] is False
    assert "incomplete_horizon" in incomplete["failure_reasons"]
    assert floor["paper_success"] is False
    assert "floor_contact" in floor["failure_reasons"]


def test_paper_metric_ignores_cycles_starting_before_warmup() -> None:
    result = _paper_cycle(left=0, right=8, apex_frame=4)

    assert result["paper_success"] is False
    assert result["completed_post_warmup_cycles"] == 0


def test_contact_onsets_match_three_frame_source_debounce() -> None:
    contacts = [False] * 25
    contacts[16] = True
    contacts[18] = True
    contacts[24] = True

    assert contact_onsets(contacts) == [16, 24]


def test_paper_metric_validates_trace_shapes() -> None:
    with pytest.raises(ValueError, match=r"T\+1"):
        score_paper_true_flights(
            ball_z=[0.0],
            paddle_z=[0.0, 0.0],
            paddle_contact=[False],
            floor_contact=[False],
            requested_transitions=1,
        )
