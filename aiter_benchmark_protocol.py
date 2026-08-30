"""Adaptive paired timing protocol shared by AITER route-bank tuners."""

import math
import statistics
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

MEDIAN_SE_FACTOR = 1.2533141373155001


@dataclass(frozen=True)
class AdaptiveTimingPolicy:
    min_samples: int = 16
    max_samples: int = 128
    sample_step: int = 16
    confidence: float = 0.90
    epsilon_pct: float = 2.0
    stable_rounds: int = 2
    noise_floor_pct: float = 0.5

    def __post_init__(self) -> None:
        if self.min_samples <= 0 or self.max_samples < self.min_samples:
            raise ValueError("timing sample bounds are invalid")
        if self.sample_step <= 0:
            raise ValueError("timing sample step must be positive")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("timing confidence must be between 0 and 1")
        if not math.isfinite(self.epsilon_pct) or self.epsilon_pct <= 0.0:
            raise ValueError("timing epsilon must be positive")
        if self.stable_rounds <= 0:
            raise ValueError("timing stable rounds must be positive")
        if not math.isfinite(self.noise_floor_pct) or self.noise_floor_pct < 0.0:
            raise ValueError("timing noise floor must be nonnegative")

    @property
    def epsilon_log(self) -> float:
        return math.log1p(self.epsilon_pct / 100.0)

    @property
    def noise_floor_log(self) -> float:
        return math.log1p(self.noise_floor_pct / 100.0)

    @property
    def z_value(self) -> float:
        return statistics.NormalDist().inv_cdf(0.5 + self.confidence / 2.0)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "min_samples": self.min_samples,
            "max_samples": self.max_samples,
            "sample_step": self.sample_step,
            "confidence": self.confidence,
            "epsilon_pct": self.epsilon_pct,
            "stable_rounds": self.stable_rounds,
            "noise_floor_pct": self.noise_floor_pct,
        }


def _quantile(values: list[float], fraction: float) -> float:
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


def timing_summary(
    samples_ms: list[float], flops: int, noise_floor_log: float
) -> dict[str, Any]:
    values = [float(value) for value in samples_ms]
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("timing samples must be finite and positive")
    ordered = sorted(values)
    logs = [math.log(value) for value in values]
    ordered_logs = sorted(logs)
    median_log = statistics.median(logs)
    mad_log = statistics.median(abs(value - median_log) for value in logs)
    iqr_log = _quantile(ordered_logs, 0.75) - _quantile(ordered_logs, 0.25)
    stddev_log = statistics.stdev(logs) if len(logs) >= 2 else 0.0
    robust_sigma = max(
        stddev_log,
        1.4826 * mad_log,
        iqr_log / 1.349,
        noise_floor_log,
    )
    stderr = MEDIAN_SE_FACTOR * robust_sigma / math.sqrt(len(values))
    high_fence = (
        _quantile(ordered_logs, 0.75) + 1.5 * iqr_log if iqr_log > 0.0 else float("inf")
    )
    return {
        "samples_ms": values,
        "median_ms": statistics.median(values),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "stdev_ms": statistics.pstdev(values),
        "median_tflops": flops / (statistics.median(values) * 1.0e9),
        "mean_log_ms": statistics.fmean(logs),
        "median_log_ms": median_log,
        "stddev_log_ms": stddev_log,
        "robust_sigma_log": robust_sigma,
        "stderr_median_log": stderr,
        "mad_log": mad_log,
        "iqr_log": iqr_log,
        "p10_ms": _quantile(ordered, 0.10),
        "p90_ms": _quantile(ordered, 0.90),
        "outlier_count": sum(value > high_fence for value in logs),
    }


def _event_time(function: Callable[[], Any], launches: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(launches):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / launches


def adaptive_pair_timings(
    functions: Mapping[str, Callable[[], Any]],
    *,
    policy: AdaptiveTimingPolicy,
    warmup: int,
    launches_per_sample: int,
    flops: int,
    select_sample: Callable[[int], None] | None = None,
    sample_capacity: int | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Measure a matched pair until its log-time estimate is stable."""
    names = list(functions)
    if len(names) != 2:
        raise ValueError("paired AITER timing requires exactly two functions")
    if warmup < 0 or launches_per_sample <= 0:
        raise ValueError("warmup must be nonnegative and launches must be positive")
    capacity = policy.max_samples
    if sample_capacity is not None:
        if sample_capacity < policy.min_samples:
            raise ValueError("route bank is smaller than the timing minimum")
        capacity = min(capacity, sample_capacity)

    def order(index: int) -> list[str]:
        return names[index % 2 :] + names[: index % 2]

    samples = {name: [] for name in names}
    records: list[dict[str, object]] = []
    rounds: list[dict[str, object]] = []
    orders: list[list[str]] = []
    target = min(policy.min_samples, capacity)
    warmed = measured = 0
    previous: float | None = None
    streak = 0
    stop_reason = "max_samples"
    while True:
        if select_sample is not None:
            for index in range(warmed, target):
                select_sample(index)
                for _ in range(warmup):
                    for name in order(index):
                        for _ in range(launches_per_sample):
                            functions[name]()
            warmed = target
        elif warmed == 0:
            for index in range(warmup):
                for name in order(index):
                    for _ in range(launches_per_sample):
                        functions[name]()
            warmed = target
        torch.cuda.synchronize()
        for index in range(measured, target):
            if select_sample is not None:
                select_sample(index)
            current_order = order(index)
            values = {
                name: _event_time(functions[name], launches_per_sample)
                for name in current_order
            }
            for name, value in values.items():
                samples[name].append(value)
            orders.append(current_order)
            records.append({"sample_index": index, **values, "order": current_order})
        measured = target
        summaries = {
            name: timing_summary(values, flops, policy.noise_floor_log)
            for name, values in samples.items()
        }
        log_gap = float(summaries[names[1]]["median_log_ms"]) - float(
            summaries[names[0]]["median_log_ms"]
        )
        gap_se = math.sqrt(
            sum(float(summaries[name]["stderr_median_log"]) ** 2 for name in names)
        )
        ci_half_log = policy.z_value * gap_se
        speedup = math.exp(log_gap)
        change_log = None if previous is None else abs(log_gap - previous)
        change_pct = (
            None
            if previous is None
            else abs(speedup / math.exp(previous) - 1.0) * 100.0
        )
        precise = ci_half_log <= policy.epsilon_log
        stable = precise and change_log is not None and change_log <= policy.epsilon_log
        streak = streak + 1 if stable else 0
        rounds.append(
            {
                "sample_count": measured,
                "speedup": speedup,
                "log_speedup": log_gap,
                "ci_low_speedup": math.exp(log_gap - ci_half_log),
                "ci_high_speedup": math.exp(log_gap + ci_half_log),
                "ci_half_width_pct": (math.exp(ci_half_log) - 1.0) * 100.0,
                "estimate_change_pct": (change_pct),
                "precise": precise,
                "stable": stable,
                "stable_streak": streak,
            }
        )
        if streak >= policy.stable_rounds:
            stop_reason = "stable_confidence_and_prefix"
            break
        if target >= capacity:
            break
        previous = log_gap
        target = min(capacity, target + policy.sample_step)
    return summaries, {
        "sample_unit": "route_vector"
        if select_sample is not None
        else "ordinary_measurement",
        "sample_count": measured,
        "sample_capacity": capacity,
        "policy": policy.to_mapping(),
        "stop_reason": stop_reason,
        "stable": stop_reason == "stable_confidence_and_prefix",
        "sample_order": orders,
        "samples": records,
        "rounds": rounds,
    }
