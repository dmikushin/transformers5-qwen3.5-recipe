#!/usr/bin/env python3
"""Tune one exact AITER GMM/PTGMM key with coefficient-only route priors.

The script deliberately keeps the search local: it evaluates one field at a time
from the current exact-key config. Group sizes are sampled from the fitted
model-level learned priors and the capture-free DeepSeek hash prior. No route
captures, production-frequency weights, or Cartesian candidate products are
used.
"""

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from aiter.ops.triton.gmm import gmm, ptgmm

from aiter_benchmark_protocol import AdaptiveTimingPolicy, adaptive_pair_timings
from expert_distribution_prior import (
    EXPERT_PRIOR_NAMES,
    EXPERTS,
    RouteVector,
    expert_prior_metadata,
    route_bank_for_routed_rows,
)
from moe_gmm_configs import gmm_config, ptgmm_config

FIELDS = (
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_K",
    "BLOCK_SIZE_N",
    "GROUP_SIZE",
    "GRID_DIM",
    "num_warps",
    "num_stages",
)

FAMILIES = {
    "deepseek": {
        "rows": {1: 12288, 4: 49152, 16: 196608},
        "gmm_base": (
            (4096, 2048, True),
            (2048, 4096, True),
            (2048, 4096, False),
            (4096, 2048, False),
        ),
        "gmm_lora": (
            (4096, 4, True),
            (2048, 4, True),
            (4, 4096, True),
            (4, 4096, False),
            (4, 2048, False),
            (4096, 4, False),
        ),
        "ptgmm_base": (
            (4096, 2048),
            (2048, 4096),
        ),
        "ptgmm_lora": (
            (4096, 4),
            (2048, 4),
            (4, 4096),
        ),
    },
    "qwen": {
        "rows": {1: 16384, 4: 65536, 16: 262144},
        "gmm_base": (
            (2048, 512, True),
            (512, 2048, True),
            (512, 2048, False),
            (2048, 512, False),
        ),
        "gmm_lora": (
            (2048, 4, True),
            (512, 4, True),
            (4, 1024, True),
            (4, 2048, True),
            (4, 2048, False),
            (4, 512, False),
            (1024, 4, False),
            (2048, 4, False),
        ),
        "ptgmm_base": (
            (2048, 512),
            (512, 2048),
        ),
        "ptgmm_lora": (
            (2048, 4),
            (512, 4),
            (4, 1024),
            (4, 2048),
        ),
    },
}


@dataclass
class LaunchCase:
    routes: tuple[RouteVector, ...]
    groups: tuple[torch.Tensor, ...]
    lhs: torch.Tensor
    rhs: torch.Tensor
    out: torch.Tensor
    op: str
    selected_route: int = 0

    def select_sample(self, index: int) -> None:
        if not 0 <= index < len(self.routes):
            raise IndexError(f"route vector index {index} is outside the route bank")
        self.selected_route = index

    def launch(self, config: dict[str, int]) -> torch.Tensor:
        group_sizes = self.groups[self.selected_route]
        if self.op == "gmm":
            return gmm(
                self.lhs,
                self.rhs,
                group_sizes,
                preferred_element_type=torch.bfloat16,
                existing_out=self.out,
                config=config,
            )
        return ptgmm(
            self.lhs,
            self.rhs,
            group_sizes,
            preferred_element_type=torch.bfloat16,
            existing_out=self.out,
            config=config,
        )


def make_route_bank(
    expert_prior: str,
    rows: int,
    count: int,
    seed: int | None,
) -> tuple[RouteVector, ...]:
    return route_bank_for_routed_rows(expert_prior, rows, count, seed=seed)


def config_values(
    op: str,
    field: str,
    k: int,
    n: int,
    m: int,
    current: int,
) -> list[int]:
    if field == "BLOCK_SIZE_M":
        values = [16, 32, 64, 128, 256]
        if op == "ptgmm" and m // EXPERTS >= 512:
            values.append(512)
    elif field == "BLOCK_SIZE_K":
        if k <= 16:
            values = [16, 32]
        elif op == "gmm":
            values = [32, 64, 128, 256]
        else:
            values = [64, 128, 256, 512]
    elif field == "BLOCK_SIZE_N":
        if n <= 16:
            values = [16, 32]
        elif op == "gmm" and k > 16 and n > 16:
            values = [32, 64, 128, 256]
        else:
            values = [32, 64, 128, 256, 512]
    elif field == "GROUP_SIZE":
        values = [1, 2, 4, 8]
    elif field == "GRID_DIM":
        values = [20, 40, 80, 160, 256]
    elif field == "num_warps":
        values = [1, 2, 4, 8]
    elif field == "num_stages":
        values = [1, 2, 3]
    else:
        raise KeyError(field)
    if current not in values:
        values = [current, *values]
    return list(dict.fromkeys(values))


def valid_config(op: str, config: dict[str, int]) -> bool:
    if config["BLOCK_SIZE_K"] * config["BLOCK_SIZE_N"] > 65536:
        return False
    return not (op == "gmm" and config["BLOCK_SIZE_M"] * config["BLOCK_SIZE_N"] > 65536)


def build_case(
    op: str,
    m: int,
    k: int,
    n: int,
    route_bank: tuple[RouteVector, ...],
    transposed: bool = False,
) -> LaunchCase:
    if not route_bank:
        raise ValueError("route bank must not be empty")
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(20260824)
    groups = tuple(
        torch.tensor(route.group_sizes, device=device, dtype=torch.int32)
        for route in route_bank
    )
    if op == "gmm":
        lhs = torch.randn((m, k), generator=generator, device=device, dtype=dtype)
        if transposed:
            rhs_storage = torch.randn(
                (EXPERTS, n, k), generator=generator, device=device, dtype=dtype
            )
            rhs = rhs_storage.transpose(1, 2)
        else:
            rhs = torch.randn(
                (EXPERTS, k, n), generator=generator, device=device, dtype=dtype
            )
        out = torch.empty((m, n), device=device, dtype=dtype)
        return LaunchCase(route_bank, groups, lhs, rhs, out, op)
    lhs_storage = torch.randn((m, k), generator=generator, device=device, dtype=dtype)
    lhs = lhs_storage.transpose(0, 1)
    rhs = torch.randn((m, n), generator=generator, device=device, dtype=dtype)
    out = torch.empty((EXPERTS, k, n), device=device, dtype=dtype)
    return LaunchCase(route_bank, groups, lhs, rhs, out, op)


def benchmark_pair(
    case: LaunchCase,
    baseline: dict[str, int],
    candidate: dict[str, int],
    policy: AdaptiveTimingPolicy,
    warmup: int,
    launches_per_sample: int,
) -> dict[str, Any]:
    if not valid_config(case.op, candidate):
        return {"status": "pruned", "score_ms": math.inf}
    try:
        timing, adaptive = adaptive_pair_timings(
            {
                "baseline": lambda: case.launch(baseline),
                "candidate": lambda: case.launch(candidate),
            },
            policy=policy,
            warmup=warmup,
            launches_per_sample=launches_per_sample,
            flops=2 * case.lhs.shape[0] * case.rhs.shape[-1] * case.lhs.shape[1],
            select_sample=case.select_sample,
            sample_capacity=len(case.routes),
        )
        log_speedup = float(timing["baseline"]["median_log_ms"]) - float(
            timing["candidate"]["median_log_ms"]
        )
        return {
            "status": "ok",
            "score_ms": math.exp(float(timing["candidate"]["median_log_ms"])),
            "speedup": math.exp(log_speedup),
            "timing": timing,
            "adaptive": adaptive,
        }
    except Exception as error:  # noqa: BLE001 - reject candidate and continue search
        torch.cuda.empty_cache()
        return {
            "status": "error",
            "score_ms": math.inf,
            "error": f"{type(error).__name__}: {error}",
        }


def target_kind_for(family: str, op: str, target: tuple[int, ...]) -> str:
    for target_kind in ("base", "lora"):
        if target in FAMILIES[family][f"{op}_{target_kind}"]:
            return target_kind
    raise ValueError(f"unknown {op} target for {family}: {target}")


def correctness_check(
    op: str,
    baseline: dict[str, int],
    selected: dict[str, int],
    k: int,
    n: int,
    transposed: bool = False,
) -> dict[str, Any]:
    device = "cuda"
    dtype = torch.bfloat16
    generator = torch.Generator(device=device).manual_seed(2037)
    small_groups = torch.zeros(EXPERTS, device=device, dtype=torch.int32)
    small_groups[0] = 2
    small_groups[7] = 3
    if op == "gmm":
        lhs = torch.randn((5, k), generator=generator, device=device, dtype=dtype)
        if transposed:
            rhs_storage = torch.randn(
                (EXPERTS, n, k), generator=generator, device=device, dtype=dtype
            )
            rhs = rhs_storage.transpose(1, 2)
        else:
            rhs = torch.randn(
                (EXPERTS, k, n), generator=generator, device=device, dtype=dtype
            )
        out_baseline = gmm(
            lhs,
            rhs,
            small_groups,
            preferred_element_type=dtype,
            config=baseline,
        )
        out_selected = gmm(
            lhs,
            rhs,
            small_groups,
            preferred_element_type=dtype,
            config=selected,
        )
    else:
        lhs_storage = torch.randn(
            (5, k), generator=generator, device=device, dtype=dtype
        )
        lhs = lhs_storage.transpose(0, 1)
        rhs = torch.randn((5, n), generator=generator, device=device, dtype=dtype)
        out_baseline = ptgmm(
            lhs, rhs, small_groups, preferred_element_type=dtype, config=baseline
        )
        out_selected = ptgmm(
            lhs, rhs, small_groups, preferred_element_type=dtype, config=selected
        )
    equal = bool(torch.equal(out_baseline, out_selected))
    delta = (out_baseline.float() - out_selected.float()).abs()
    result = {"bitwise_equal": equal, "max_abs": float(delta.max())}
    del lhs, rhs, out_baseline, out_selected, small_groups
    if "rhs_storage" in locals():
        del rhs_storage
    if "lhs_storage" in locals():
        del lhs_storage
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    family = "qwen" if args.expert_prior == "qwen-learned" else "deepseek"
    rows = FAMILIES[family]["rows"][args.batch]
    final_repeats = args.final_repeats or args.repeats
    final_max_samples = args.final_max_samples or args.max_samples
    final_sample_step = args.final_sample_step or args.sample_step
    route_bank = make_route_bank(
        args.expert_prior,
        rows,
        max(args.max_samples, final_max_samples),
        args.route_seed,
    )
    route_seed = route_bank[0].profile.seed
    if args.op == "gmm":
        target = (args.k, args.n, args.trans)
        current = gmm_config(rows, args.k, args.n, args.trans, args.expert_prior)
    else:
        target = (args.k, args.n)
        current = ptgmm_config(rows, args.k, args.n, args.expert_prior)
    target_kind = args.target_kind or target_kind_for(family, args.op, target)
    if target not in FAMILIES[family][f"{args.op}_{target_kind}"]:
        raise ValueError(f"unknown {args.op} {target_kind} target {target}")

    launch_case = build_case(
        args.op,
        rows,
        args.k,
        args.n,
        route_bank,
        args.trans if args.op == "gmm" else False,
    )
    screen_policy = AdaptiveTimingPolicy(
        min_samples=args.repeats,
        max_samples=args.max_samples,
        sample_step=args.sample_step,
        confidence=args.adaptive_confidence,
        epsilon_pct=args.adaptive_epsilon_pct,
        stable_rounds=args.adaptive_stable_rounds,
        noise_floor_pct=args.adaptive_noise_floor_pct,
    )
    final_policy = AdaptiveTimingPolicy(
        min_samples=final_repeats,
        max_samples=final_max_samples,
        sample_step=final_sample_step,
        confidence=args.adaptive_confidence,
        epsilon_pct=args.adaptive_epsilon_pct,
        stable_rounds=args.adaptive_stable_rounds,
        noise_floor_pct=args.adaptive_noise_floor_pct,
    )
    evaluated: dict[tuple[tuple[str, int], ...], dict[str, Any]] = {}
    baseline = dict(current)

    def key(config: dict[str, int]) -> tuple[tuple[str, int], ...]:
        return tuple(sorted(config.items()))

    def evaluate(config: dict[str, int]) -> float:
        config = dict(config)
        config_key = key(config)
        if config_key not in evaluated:
            measured = benchmark_pair(
                launch_case,
                baseline,
                config,
                screen_policy,
                args.warmup,
                args.launches_per_sample,
            )
            evaluated[config_key] = {"config": config, **measured}
            print(json.dumps(evaluated[config_key], sort_keys=True), flush=True)
        return float(evaluated[config_key]["score_ms"])

    evaluate(baseline)
    rounds = []
    for round_index in range(args.rounds):
        round_start = dict(current)
        for field in args.fields:
            choices = []
            for value in config_values(
                args.op, field, args.k, args.n, rows, current[field]
            ):
                candidate = dict(current)
                candidate[field] = value
                choices.append((evaluate(candidate), value, candidate))
            _, _, current = min(choices, key=lambda item: (item[0], item[1]))
        current_score = evaluate(current)
        rounds.append(
            {
                "round": round_index + 1,
                "start": round_start,
                "winner": dict(current),
                "score_ms": current_score,
            }
        )
        if current == round_start:
            break

    final = benchmark_pair(
        launch_case,
        baseline,
        current,
        final_policy,
        args.warmup,
        args.launches_per_sample,
    )
    accepted = (
        current != baseline
        and final["status"] == "ok"
        and float(final["speedup"]) > 1.0 + args.min_gain
    )
    selected = dict(current) if accepted else dict(baseline)
    correctness = correctness_check(
        args.op,
        baseline,
        selected,
        args.k,
        args.n,
        args.trans if args.op == "gmm" else False,
    )
    if accepted and not correctness["bitwise_equal"]:
        accepted = False
        selected = dict(baseline)
        correctness = correctness_check(
            args.op,
            baseline,
            selected,
            args.k,
            args.n,
            args.trans if args.op == "gmm" else False,
        )

    selected_name = "candidate" if accepted else "baseline"
    selected_score = (
        float(final["timing"][selected_name]["median_log_ms"])
        if final["status"] == "ok"
        else math.inf
    )
    prior = {
        **expert_prior_metadata(args.expert_prior),
        "kind": "coefficient_only",
        "route_seed": route_seed,
        "vector_count": len(route_bank),
        "vectors": [route.to_mapping() for route in route_bank],
        "mixed_deepseek_learned_hash": False,
    }
    report = {
        "target": {
            "family": family,
            "target_kind": target_kind,
            "expert_prior": args.expert_prior,
            "batch": args.batch,
            "op": args.op,
            "rows": rows,
            "k": args.k,
            "n": args.n,
            "transposed_rhs": args.trans if args.op == "gmm" else None,
            "search_fields": list(args.fields),
            "rounds_requested": args.rounds,
            "min_gain": args.min_gain,
        },
        "prior": prior,
        "protocol": {
            "sample_unit": "route_vector",
            "route_bank_generated_before_timing": True,
            "one_timing_sample_per_route_vector": True,
            "launches_per_sample": args.launches_per_sample,
            "warmup": args.warmup,
            "adaptive": final_policy.to_mapping(),
            "screen_adaptive": screen_policy.to_mapping(),
            "final_adaptive": final_policy.to_mapping(),
            "route_order": "alternating implementation order",
            "numeric_inputs_fixed": True,
        },
        "baseline": {
            "config": baseline,
            "screen": evaluated[key(baseline)],
            "final": {
                "timing": final.get("timing"),
                "adaptive": final.get("adaptive"),
            },
        },
        "rounds": rounds,
        "screen_winner": {
            "config": dict(current),
            "score_ms": float(evaluated[key(current)]["score_ms"]),
        },
        "final_candidate": {"config": dict(current), **final},
        "accepted": accepted,
        "selected": {"config": selected, "score_ms": math.exp(selected_score)},
        "correctness": correctness,
        "evaluations": list(evaluated.values()),
    }
    gc.collect()
    torch.cuda.empty_cache()
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert-prior", choices=EXPERT_PRIOR_NAMES, required=True)
    parser.add_argument("--target-kind", choices=("base", "lora"), default=None)
    parser.add_argument("--batch", choices=(1, 4, 16), type=int, required=True)
    parser.add_argument("--op", choices=("gmm", "ptgmm"), required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--trans", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--fields", default=",".join(FIELDS))
    parser.add_argument("--route-seed", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--sample-step", type=int, default=16)
    parser.add_argument("--final-repeats", type=int, default=None)
    parser.add_argument("--final-max-samples", type=int, default=None)
    parser.add_argument("--final-sample-step", type=int, default=None)
    parser.add_argument("--adaptive-confidence", type=float, default=0.90)
    parser.add_argument("--adaptive-epsilon-pct", type=float, default=2.0)
    parser.add_argument("--adaptive-stable-rounds", type=int, default=2)
    parser.add_argument("--adaptive-noise-floor-pct", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--launches-per-sample", type=int, default=1)
    parser.add_argument("--min-gain", type=float, default=0.01)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.fields = tuple(
        field.strip() for field in args.fields.split(",") if field.strip()
    )
    if not args.fields or any(field not in FIELDS for field in args.fields):
        parser.error(f"--fields must contain only: {', '.join(FIELDS)}")
    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.repeats <= 0 or args.max_samples < args.repeats:
        parser.error("--max-samples must be at least positive --repeats")
    if args.sample_step <= 0:
        parser.error("--sample-step must be positive")
    if args.final_repeats is not None and args.final_repeats <= 0:
        parser.error("--final-repeats must be positive")
    if args.final_max_samples is not None and args.final_max_samples <= 0:
        parser.error("--final-max-samples must be positive")
    if args.final_sample_step is not None and args.final_sample_step <= 0:
        parser.error("--final-sample-step must be positive")
    final_repeats = args.final_repeats or args.repeats
    final_max_samples = args.final_max_samples or args.max_samples
    final_sample_step = args.final_sample_step or args.sample_step
    if final_max_samples < final_repeats:
        parser.error("final max samples must be at least final repeats")
    if final_sample_step > final_max_samples:
        parser.error("final sample step cannot exceed final max samples")
    if args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.launches_per_sample <= 0:
        parser.error("--launches-per-sample must be positive")
    if args.min_gain < 0.0:
        parser.error("--min-gain must be nonnegative")
    torch.cuda.set_device(0)
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "target": report["target"],
                "accepted": report["accepted"],
                "selected": report["selected"],
                "correctness": report["correctness"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
