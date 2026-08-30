import math

import pytest

import aiter_benchmark_protocol as protocol


def test_policy_defaults_are_the_campaign_contract() -> None:
    policy = protocol.AdaptiveTimingPolicy()
    assert policy.min_samples == 16
    assert policy.max_samples == 128
    assert policy.sample_step == 16
    assert policy.confidence == 0.90
    assert policy.epsilon_pct == 2.0
    assert policy.stable_rounds == 2
    assert policy.noise_floor_pct == 0.5


def test_timing_summary_reports_log_space_statistics() -> None:
    summary = protocol.timing_summary([1.0, 2.0, 4.0], 8_000_000, 0.0)
    assert summary["median_log_ms"] == pytest.approx(math.log(2.0))
    assert summary["median_ms"] == pytest.approx(2.0)
    assert summary["median_tflops"] == pytest.approx(0.004)
    assert summary["outlier_count"] == 0


def test_adaptive_pair_uses_route_prefix_and_alternates_order(monkeypatch) -> None:
    monkeypatch.setattr(protocol.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(
        protocol, "_event_time", lambda function, launches: function() or 1.0
    )
    selected = []
    calls = []

    def select(index: int) -> None:
        selected.append(index)

    def baseline() -> None:
        calls.append("baseline")

    def candidate() -> None:
        calls.append("candidate")

    summaries, report = protocol.adaptive_pair_timings(
        {"baseline": baseline, "candidate": candidate},
        policy=protocol.AdaptiveTimingPolicy(
            min_samples=2, max_samples=6, sample_step=2, stable_rounds=2
        ),
        warmup=1,
        launches_per_sample=1,
        flops=1,
        select_sample=select,
        sample_capacity=6,
    )
    assert report["sample_count"] == 6
    assert report["stop_reason"] == "stable_confidence_and_prefix"
    assert report["sample_order"] == [
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
        ["candidate", "baseline"],
        ["baseline", "candidate"],
        ["candidate", "baseline"],
    ]
    assert [record["sample_index"] for record in report["samples"]] == [
        0,
        1,
        2,
        3,
        4,
        5,
    ]
    assert selected.count(0) >= 2
    assert selected.count(3) >= 2
    assert summaries["baseline"]["median_log_ms"] == pytest.approx(0.0)


def test_launches_per_sample_stays_on_one_route(monkeypatch) -> None:
    monkeypatch.setattr(protocol.torch.cuda, "synchronize", lambda: None)
    selected = []
    calls = []

    def select(index: int) -> None:
        selected.append(index)

    def baseline() -> None:
        calls.append("baseline")

    def candidate() -> None:
        calls.append("candidate")

    _, report = protocol.adaptive_pair_timings(
        {"baseline": baseline, "candidate": candidate},
        policy=protocol.AdaptiveTimingPolicy(
            min_samples=1, max_samples=1, sample_step=1
        ),
        warmup=0,
        launches_per_sample=3,
        flops=1,
        select_sample=select,
        sample_capacity=1,
    )
    assert report["sample_count"] == 1
    assert report["sample_unit"] == "route_vector"
    assert report["samples"][0]["sample_index"] == 0
    assert calls == ["baseline"] * 3 + ["candidate"] * 3
    assert set(selected) == {0}
