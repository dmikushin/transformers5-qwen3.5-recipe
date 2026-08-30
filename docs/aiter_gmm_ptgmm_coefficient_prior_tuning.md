# AITER GMM and PTGMM coefficient-prior tuning

This document defines the reproducible tuning protocol for the AITER `gmm` and `ptgmm` configurations in this repository. The production authority remains `moe_gmm_configs.py`. AITER source and operator contracts are not modified by the tuner.

The tuner searches exact supported shape keys with bounded coordinate descent. It is a benchmark tool, not a production route dispatcher and not a claim about model quality or training-wide route frequency.

## One prior law per run

Every tuner invocation selects exactly one value of `--expert-prior`:

| Prior | Family | Top-k | Law |
|---|---|---:|---|
| `qwen-learned` | Qwen | 8 | fitted learned support and rank curve |
| `deepseek-learned` | DeepSeek | 6 | fitted learned support and rank curve |
| `deepseek-hash` | DeepSeek | 6 | fitted persistent-head plus Dirichlet body |

The coefficient definitions are implemented in `expert_distribution_prior.py` and numerically match `torch-ggml-ops/bench/workload_prior.py`. A DeepSeek learned draw and a DeepSeek hash draw are separate campaigns. They are never averaged, pooled, or combined with a `40/43` versus `3/43` production weighting. The campaign runner defaults to one Qwen learned campaign and two separate DeepSeek campaigns.

The only workload input to a fitted law is the physical token count `T = physical_batch * 2048`. The sampler uses the fitted residual laws, exact constrained largest-remainder rounding, and a deterministic expert-identity permutation. It preserves exactly `top_k * T` routed rows and keeps every hash expert active. Seeds are recorded in each report.

DeepSeek V4 uses both route laws in one model. `model.layers[0]`, `model.layers[1]`, and `model.layers[2]` are `hash_moe` layers with `DeepseekV4HashRouter`, so their routed expert modules use `deepseek-hash`. `model.layers[3]` through `model.layers[42]` are ordinary `moe` layers with `DeepseekV4TopKRouter`, so their routed expert modules use `deepseek-learned`. The two prior values select configurations for the same B1/B4/B16 row counts and expert matrix geometries. They must not be pooled or assigned by a model-wide compromise. The dense `shared_experts` MLP and the router's own linear are outside this GMM/PTGMM route-prior table.

The current GGUF training path uses packed MMQ for frozen base expert projections and input gradients, while AITER GMM/PTGMM handles the rank-4 LoRA factors. Base-shaped entries are retained for future non-packed base-kernel tuning and are not evidence that the current packed path invokes those entries.

The routed row counts are:

| Family | B1 | B4 | B16 |
|---|---:|---:|---:|
| DeepSeek top-6 | 12,288 | 49,152 | 196,608 |
| Qwen top-8 | 16,384 | 65,536 | 262,144 |

No captured histogram, checkpoint, layer identity, training step, or route corpus is consumed by the coefficient prior.

## Route bank

`route_bank_for_routed_rows()` creates the complete route bank before any timed AITER call. The default bank has 128 vectors. The first route seed is `8,314,159 + physical_batch * 104,729`, with the DeepSeek hash offset applied by the canonical prior module. An explicit `--route-seed` is also supported.

A `RouteVector` records:
- the selected prior, seed, token count, top-k, row total, and row digest.
- compact active `expert_indices`.
- cumulative `expert_offsets` over the compact active experts.
- `group_sizes`, a full 256-entry tuple indexed by physical expert ID.

AITER currently receives the full `group_sizes` tensor because its GMM/PTGMM calls own all 256 RHS experts. The compact indices and offsets are retained as route provenance and make the active-order interpretation explicit. They are not silently substituted for the AITER group-size contract.

A route vector is one complete workload: every expert group size is selected as a unit. The route bank is generated once for a target and reused for every configuration comparison in that target. Numeric input tensors are also created once per target with a deterministic CUDA generator and are reused by both configurations. Route generation, tensor allocation, validation, and kernel compilation are outside measured CUDA events.

## Timing protocol

The defaults are:

| Setting | Default |
|---|---:|
| initial samples (`--repeats`) | 16 |
| maximum samples (`--max-samples`) | 128 |
| expansion (`--sample-step`) | 16 |
| confidence | 90% |
| estimate epsilon | 2% |
| stable rounds | 2 |
| noise floor | 0.5% |
| warmup launches per route | 2 |
| timed launches per sample | 1 |

The sample unit is a route vector, not a profile aggregate. For route index `i`, exactly one baseline timing sample and one candidate timing sample are recorded. The second launch is first on odd indices and the baseline is first on even indices. This alternating order is part of the report and reduces systematic position effects.

`--launches-per-sample N` runs `N` launches for each implementation inside the same timed event pair and stores the average milliseconds as the one sample for that route vector. The route, inputs, and output contract remain fixed across those launches. It does not create `N` independent route samples and does not permit mixing route vectors.

For every implementation, raw route samples are retained. Summaries are computed in log-time space with a robust scale estimate using standard deviation, MAD, IQR, and the configured noise floor. Adaptive expansion stops only after the configured number of consecutive rounds satisfies both:
- the paired log-speedup confidence interval is within epsilon.
- the cumulative log-speedup estimate changed by no more than epsilon from the previous round.

The primary paired speedup is:

`exp(median(log(baseline_ms)) - median(log(candidate_ms)))`

The report contains raw samples, per-round confidence bounds, stop reason, route order, sample count, policy values, and the full deterministic route bank. A candidate is accepted only when the final paired result exceeds `1 + --min-gain` and the existing correctness check passes.

## Search and correctness

Each exact target starts from its current production configuration. The tuner changes only the supported fields `BLOCK_SIZE_M`, `BLOCK_SIZE_K`, `BLOCK_SIZE_N`, `GROUP_SIZE`, `GRID_DIM`, `num_warps`, and `num_stages`. It uses bounded coordinate descent. Invalid shared-memory products and launch errors reject only that candidate.

Correctness compares the baseline and selected configuration on a deterministic sparse boundary route with group sizes `[2, 0, 3]`. This check is separate from performance route-bank sampling and does not change the selected prior law.

## Commands

A single target can be run as follows. The expert prior determines the route family:

```bash
python tune_coefficient_prior_gmm.py \
  --expert-prior deepseek-learned \
  --batch 4 --op ptgmm --k 4096 --n 4 \
  --output results/deepseek-learned-ptgmm.json
```

The campaign runner passes the same protocol arguments to each target. Its output names include the prior, so learned and hash results cannot overwrite or be mistaken for one another:

```bash
python run_coefficient_prior_campaign.py \
  --output-dir results/gmm_campaign --expert-prior deepseek-learned
```

Confirmation replays the source report's one prior law, route seed, complete route bank, warmup, launch multiplicity, and adaptive policy. It refuses to time if regenerated route metadata does not exactly match the source report:

```bash
python confirm_coefficient_prior_gmm.py \
  --input-dir results/gmm_campaign \
  --output results/gmm_campaign/confirmation.json
```

## Evidence boundary

The fitted laws are workload priors. Deterministic route banks make paired configuration comparisons reproducible, but they do not exhaust the residual distributions or establish production frequency. The bounded coordinate search does not prove global optimality. Historical result documents in `torch-ggml-ops` remain unchanged and are not rewritten by this protocol.
