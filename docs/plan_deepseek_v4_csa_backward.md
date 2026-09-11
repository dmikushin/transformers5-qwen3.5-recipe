# DeepSeek V4 CSA backward plan for gfx1151

## Status

Accepted and optimization-exhausted for batches 1/4/16 at S2048 on `gfx1151`. Both post-HCA ownership experiments are complete and accepted: deterministic group-2 compressed-gradient partials at every batch, and B16-only two-phase dS/D256 dQ ownership. Custom backward covers the fused two-region attention and rate-4 producer. Exact component, non-reentrant checkpoint, compiler, allocation, and prior real GGUF update gates pass.

Reopened for the production contiguous-BSHD `dO` layout. The tables below were measured while the component benchmark fed a BHSD-backed `dO` view. The full model now feeds a contiguous BSHD `dO` after the post-attention rotation moved to BSHD (`deepseek_v4_attention.py`, commit `5ec3113`), and the key-owner dV kernels are up to 2x slower with that layout. Stage 1 retuning results are in "Production dO layout and Stage 1 key-owner retune" at the end of this document.

## Gradient contract

Attention backward returns:
- BF16 `dQ [B,64,2048,512]`.
- one combined BF16 local shared-KV gradient `[B,1,2048,512]` because local K=V is one activation.
- one combined BF16 compressed shared-KV gradient `[B,1,512,512]` because compressed K=V is one activation.
- FP32 `dSink [64]`.

Producer backward consumes combined `dCompressedKV` and returns BF16 gradients for compressor KV and gate projections `[B,2048,1024]`. Frozen position bias, RMS scale, and RoPE tables receive no gradient. Calls requesting those gradients fail closed.

Model-native Q/KV storage is contiguous BHSD. Internal attention views are BSHD. Production `dO` may be a non-contiguous view when its last dimension is contiguous. Layouts requiring a copy fail closed. All input and output-gradient offsets are checked for signed-int32 fit.

## Accepted attention schedule

Forward stores separate FP16 local and compressed raw-score states. B1/B4 use direct dense compressed rows. B16 uses exact arithmetic-packed triangular compressed rows. Backward uses one in-place handoff:
- local key-owned `dV` reconstructs probabilities from local scores and FP32 LSE.
- group-2 compressed key owners reconstruct probabilities into a bounded reusable FP32 partial workspace, then one deterministic reducer writes compressed `dV`.
- at B1/B4, one grouped query owner computes `Delta = sum_d O*dO`, reconstructs both probability regions, accumulates one `dQ`, computes deterministic sink partials, and overwrites both score regions with FP16 `dS`.
- at B16, that owner forms Delta, sink partials, and both dS regions without a D512 dQ accumulator. A second D256 feature-sliced owner consumes stored dS and writes disjoint final BF16 dQ slices.
- local key-owned one-dot `dK` consumes local `dS` and adds into local `dV`.
- group-2 compressed key owners reuse the FP32 workspace for dK, and the deterministic reducer adds compressed dK into dV.
- one deterministic FP32 sink reduction combines query-tile partials.

Local, compressed, and sink logits share one softmax. Neither region is normalized independently. There are no atomics, per-head KV outputs, repeated KV heads, dense probabilities, persistent Delta, second score/gradient state, or query-sized partial-gradient workspace. The reusable compressed workspace is `[B,32,512,512]` FP32, or `32/128/512 MiB` at B1/B4/B16.

## Accepted launch tables

Grouped query owner:

| Batch | Query M | Head group | Local N | Compressed N | Warps | waves/EU |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8 | 8 | 16 | 16 | 8 | 0 |
| 4 | 8 | 8 | 16 | 16 | 8 | 0 |
| 16 | 16 | 4 | 16 | 32 | 4 | 0 |

At B16 this row is the dS/Delta/sink launch with dQ accumulation disabled. Final dQ uses M16/group-4, D256 feature slices, four warps, and one wave/EU. B1/B4 use the combined owner.

Local shared-KV owners retain the accepted sliding geometry:

| Batch | Kernel | Query M | Key N | Warps | waves/EU | Head unroll |
|---:|---|---:|---:|---:|---:|---:|
| 1 | dV | 32 | 32 | 8 | 1 | 1 |
| 1 | dK | 32 | 32 | 4 | 2 | 1 |
| 4 | dV | 32 | 32 | 8 | 1 | 2 |
| 4 | dK | 32 | 32 | 4 | 2 | 1 |
| 16 | dV | 32 | 32 | 8 | 1 | 2 |
| 16 | dK | 32 | 32 | 8 | 0 | 2 |

Compressed shared-KV owners require independent triangular-prefix geometry:

| Batch | Kernel | Query M | Key N | Head group | Warps | waves/EU | Head unroll |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | dV | 64 | 32 | 2 | 8 | 1 | 1 |
| 1 | dK | 16 | 32 | 2 | 4 | 0 | 1 |
| 4 | dV | 16 | 64 | 2 | 8 | 1 | 2 |
| 4 | dK | 16 | 32 | 2 | 4 | 1 | 1 |
| 16 | dV | 32 | 64 | 2 | 8 | 1 | 1 |
| 16 | dK | 16 | 64 | 2 | 8 | 0 | 2 |

Compressed key owner `c` starts at query `4*c+3`. Every head-group owner traverses only queries that may see its key and uses arithmetic visibility for different keys within a tile. One compact reducer combines the 32 deterministic partials per compressed row.

## Producer backward

Each program reconstructs the eight featurewise gate logits and pooled pre-norm vector in FP32. It reverses compressed-position RoPE and weighted RMSNorm, then for valid slot `i` computes:
- `dX_i = p_i * dZ`.
- `dGate_i = p_i * dZ * (x_i - pooled)`.

Ownership is unique: current Cb maps to the current compressed entry and Ca maps to the next entry. The last raw window's Ca half is unused and remains exactly zero. Entry zero has no previous-Ca owner. The kernel writes both projection gradients directly without atomics or reduction workspace.

## Correctness gates

Exact B1/B4/B16 results against the blockwise FP32-logit oracle:

| Tensor | B1 RMSE | B4 RMSE | B16 RMSE | Minimum cosine |
|---|---:|---:|---:|---:|
| Q gradient | 0.003076 | 0.003079 | 0.003114 | 0.999995 |
| Local shared-KV gradient | 0.004998 | 0.004985 | 0.004999 | 0.999987 |
| Compressed shared-KV gradient | 0.009915 | 0.009960 | 0.009953 | 0.999950 |
| Sink gradient | 0.001613 | 0.001917 | 0.002008 | 0.999998 |
| Producer KV gradient | 0.002034 | 0.002030 | 0.002052 | 0.999997 |
| Producer gate gradient | 0.001777 | 0.001751 | 0.001778 | 0.999998 |

Focused production gates are RMSE `0.0033/0.0055/0.0105/0.0025` for Q/local-KV/compressed-KV/sink and `0.0022/0.0021` for producer KV/gate. The compact-KV thresholds are intentionally looser than sliding because each compressed owner accumulates up to 2,045 query rows across 64 heads. FP32 score-state evidence shows the residual difference is BF16 dot/reduction ordering, not handoff quantization.

Non-reentrant checkpoint direct and recomputed outputs plus all four attention gradients are bitwise equal in the focused test. The real model uses 43 non-reentrant decoder checkpoints and completes both backward passes with finite expected adapter gradients.

## Timing, allocation, and profile

Final producer-plus-attention medians from `~/tmp/test_no_unsloth/deepseek_v4_csa_forward_phase_final.json`:

| Batch | Forward | Backward-only | Complete | Incremental peak allocation |
|---:|---:|---:|---:|---:|
| 1 | 15.464 ms | 24.729 ms | 40.193 ms | 451.563 MiB |
| 4 | 60.201 ms | 97.693 ms | 157.894 ms | 1,806.250 MiB |
| 16 | 109.168 ms | 323.732 ms | 432.900 ms | 6,199.500 MiB |

Against the original accepted direct-owner boundary, complete time improves by 18.3%, 8.7%, and 50.4% at B1/B4/B16. The final forward phase split alone improves the fresh B16 boundary by 26.1%. B1/B4 allocation rises by the 32/128 MiB compressed partial workspace. At B16, exact score packing saves 1,025 MiB and more than offsets the 512 MiB workspace, reducing net allocation by 513 MiB.

The isolated group-2 compressed dV+dK pair measures approximately `4.99/19.33/79.60 ms`, versus `15.5/40.6/144.3 ms` for direct all-head ownership. The B16 backward phase split reduced the post-partial/packed range near `785-816 ms` to `584-593 ms`. The later forward phase split lowers the final boundary to `432.9 ms`.

Final metadata is `~/tmp/test_no_unsloth/deepseek_v4_csa_forward_phase_metadata.json`. B16 QK/LSE uses 64 KiB LDS and 106 spills. D256 output uses 8 KiB and 10 spills. B16 dS formation uses 64 KiB and 838 spills, while D256 dQ uses 16 KiB and 17 spills. Local owners use 32 KiB and 0-19 spills. Compressed partial owners use at most 64 KiB and seven spills. Reducers are spill-free. All listed owners use 256 registers where pressure-bound, and no retained kernel uses global scratch.

## Optimization record

Retained work:
- removed all indexer/top-k/dense-block-bias backward work under the complete-selection proof.
- used one shared local+compressed query owner to retain FP32 `dQ`, Delta, and sink state.
- stored raw FP16 scores and overwrote them in place with `dS`.
- used separate local and compressed key owners with no atomics.
- replaced serialized all-head compressed owners with deterministic group-2 FP32 partials and one reusable compact reducer.
- split B16 dS/Delta/sink from D256 feature-sliced dQ, removing the long-lived D512 accumulator from probability reconstruction.
- retained direct dense compressed-score rows at B1/B4 and exact arithmetic-packed triangular rows at B16.
- tuned exact B1/B4/B16 query/head geometry, local/compressed N, warps, waves/EU, and head unroll.
- moved B16 query ownership from 8 warps/N16 to 4 warps/N32, reducing isolated `dQ` from about 628 to 505 ms.
- moved compressed B16 dV from about 106 to 67-72 ms and dK from about 71 to 51-53 ms.
- kept producer recomputation local and direct-owned, with no saved softmax/RMS workspace.
- removed fixed-shape tile predicates, the masked sink-reduction tail, producer store/reload debug barrier, and full producer-gradient zero fills. Register-only RoPE and explicit final-Ca writes preserve the BF16 boundary and exact zero-gradient contract.

Rejected work:
- Equal-64-row B16 query geometries with 8 warps clustered around `596-645 ms`. N32 was better but insufficient.
- Reducing query-owner state to 32 rows produced `640-653 ms` with 4 warps and about one second with 8 warps. 16-row owners exceeded one second. Extra KV loads and program overhead outweighed occupancy.
- Splitting local and compressed `dQ/dS` repeated output, `dO`, LSE, and Delta traffic and regressed complete timing to `58.01/201.93/1089.87 ms`.
- FP32 score/`dS` state doubled memory and did not materially improve gradients.
- FP8 E4M3 state caused 5-9% major-gradient RMSE and was rejected.
- Normalized FP16 probability state added a bandwidth pass and regressed complete timing by 5-8%.
- M8/group-8 was 1.1% faster than M16/group-4 in isolated B16 `dQ`, but complete B16 regressed from `873.02` to `877.65 ms` and used another 0.5 MiB of sink partials.
- The split dK/dV ownership beat fused dK/dV. One-dot dK remains accepted from sliding evidence.
- Output-derived Delta remains accepted for Delta only. Probability-based sink reconstruction is required. No additional persistent FP32 Delta state is needed.
- HCA's conditional local/compressed dQ split was screened after this phase split succeeded, but its best B16 complete result was `327.8 ms` versus the accepted `314.1-314.7 ms`. The candidate was removed.
- Stock AITER D512 backward is unusable on gfx1151 because its requested geometry exceeds the 64 KiB LDS limit, and no AITER DSV4 training backward exists.

LSE-only score recomputation remains a potential future B16 low-memory policy, not an accepted kernel branch. After exact compressed-score packing it could save the remaining 1.499 GiB score state at B16, but would add two large QK recomputations. The component now fits with 6.05 GiB incremental allocation, and project policy still requires end-to-end B16 attribution before adding segmented/recompute behavior. No dormant production selector is retained.

## Reopened experiment results

Deterministic compressed head partials are accepted. Head groups 1/2/4/8/16/32 and direct ownership were screened with 60-second per-variant compile limits. Group 2 wins at B1/B4/B16: isolated compressed dV+dK falls from about `15.5/40.6/144.3 ms` to `4.99/19.33/79.60 ms`, and the same `[B,32,512,512]` FP32 buffer is reused between dV and dK. Initial complete medians improved from `49.212/173.022/872.343 ms` to about `40.32/155.05/816.40 ms` before later state and dQ changes. Losing direct and alternate-group branches were removed.

The B16 two-phase dS/D256 dQ schedule is also accepted. D128/D256 and M8/group-8 versus M16/group-4 candidates compiled within 60 seconds. M16/group-4, D256, four warps, and one wave/EU won. It carries no query-sized partial workspace, writes two disjoint D256 slices directly, and consumes the existing FP16 dS handoff. The exact B16 Q-gradient RMSE is `0.003098`, within the `0.0033` gate. All other gradients remain within their existing gates. Complete B16 improves by more than the required 5%, from the post-partial/packing range near `785-816 ms` to `584-593 ms`.

All retained launches remain at or below 64 KiB LDS with zero global scratch. B1/B4 keep the combined dS/dQ owner because the experiment was deliberately B16-only.

## Autograd boundary note

The in-place score-to-dS handoff deliberately supports one backward per forward. A second backward through the same retained graph would no longer have raw scores. The production non-reentrant checkpoint/Trainer path performs one backward per replay. Retained-graph reuse is outside this specialized boundary.

Producer backward now applies the same signed-int32 reachable-offset guard as attention backward.

## Integration evidence

`~/tmp/test_no_unsloth/deepseek_v4_training_csa_cleanup_report.json` passes:
- `configured_csa=21` and the exact alternating CSA layer names.
- 43 non-reentrant checkpoints.
- zero missing or nonfinite expected gradients.
- all 469 LoRA-B tensors updated after the first optimizer step.
- no gradients on frozen grouped outputs.
- immutable sampled packed payload pointers, versions, and hashes.
- 90.624 GiB peak allocation, 93.232 GiB peak reservation, and zero process swap.
- warmed full-model second forward/backward of `7.467/17.178 s`, versus `9.612/22.909 s` before fused CSA.

The profiler emits normal ROCm timestamp-order warnings but the audit gate and trace complete.

## Final resource review

The current backward was reviewed against all available sources. AITER DSV4 and sparse paged prefill are forward-only. Its generic D512 MHA backward violates gfx1151 LDS limits. SGLang's fused compressor and sparse prefill have no training cotangent contract. llama.cpp confirms compressor ownership but is an inference graph. DS4's ROCm two-region attention validates shared-softmax traversal but supplies no reusable project-layout backward. vLLM integrates inference prefill only. Triton's generic top-k and attention examples do not provide compact shared-KV combined gradients or solve D512 accumulator pressure.

The complete sliding-backward experiment record was re-reviewed. Raw FP16 handoff, local Delta, split direct key/value ownership, one-dot dK, deterministic sink reduction, dynamic traversal, and complete-boundary selection transfer directly. Rejected sliding probability normalization, FP32 state, score recomputation, fused dK/dV, and output-derived sink candidates remain rejected after CSA-specific measurement where applicable.

The HCA-derived compressed head-partial schedule and the B16 two-phase dS/dQ ownership change both pass their complete-boundary gates. Remaining B16 cost is explained by 49.8 million visible pairs per batch element, repeated probability reconstruction for dV/dS, long imbalanced triangular traversal, and the still spill-heavy 64 KiB dS owner. The final D256 dQ phase itself is no longer pressure-bound.

At the end of this plan, review the current CSA forward and backward kernels together, all completed producer/attention work, the accepted and rejected sliding-forward/backward evidence, and the relevant AITER, llama.cpp, DS4, vLLM, SGLang, Transformers, and Triton resources. Any new speed or memory idea must be added to these plans before implementation and evaluated at the complete producer-plus-attention boundary. CSA backward optimization is exhausted for this fixed contract.

## Production dO layout and Stage 1 key-owner retune

### Stage 0: benchmark protocol correction

`benchmark_deepseek_v4_csa.py` used to build `dO` as `torch.randn_like(query).transpose(1, 2)` (a BHSD-backed view). The model feeds a contiguous BSHD gradient: instrumenting `_CSAAttentionFunction.backward` in the full model shows shape `[1,2048,64,512]`, stride `(67108864, 32768, 512, 1)`, `contiguous=True`. The query owner, the forward kernels, and the producer are layout-insensitive. The local and compressed dV owners are not, so the old benchmark hid a real cost. The benchmark now feeds the production layout in both `run_correctness` and `run_benchmark`.

Stage 0 production-layout baseline (median of 10, B1/B4/B16):

| Batch | Forward | Backward | Complete |
|---:|---:|---:|---:|
| 1 | 14.594 ms | 27.728 ms | 42.322 ms |
| 4 | 57.563 ms | 103.892 ms | 161.456 ms |
| 16 | 108.406 ms | 343.814 ms | 452.220 ms |

### Stage 1: launch-table retune

Every candidate was measured with paired sampling: each round profiles the base and every candidate once (alternating order) and the per-round gap is `log(t_candidate) - log(t_base)`. A candidate is adopted only when the 95% interval on the median paired gap is entirely below the 2% indifference zone (`~/evotensile/docs/noisy_measurements.md`). This cancels the 5-20% run-to-run drift seen in single runs. Every adopted candidate produced bitwise-identical gradients (min cosine 1.0, relative RMSE 0.0), because the per-output accumulation order is unchanged.

Adopted changes (kernel-level paired gaps):

| Batch | Owner | Change | Paired gap |
|---:|---|---|---:|
| 1 | local dV | `block_n` 32 -> 64, `waves_per_eu` 1 -> 2 | -22.7% [-28.3, -16.7] |
| 4 | local dV | `waves_per_eu` 1 -> 2 | -9.5% [-14.4, -4.2] |
| 1 | compressed dV | `waves_per_eu` 1 -> 2, `head_unroll` 1 -> 2 | -17.3% [-18.2, -16.4] |
| 4 | compressed dV | `block_m` 16 -> 32, `waves_per_eu` 1 -> 2 | -17.2% [-17.9, -16.4] |
| 4 | compressed dK | `head_unroll` 1 -> 2 | -9.1% [-10.2, -8.1] |
| 16 | compressed dV | `waves_per_eu` 1 -> 2 | -2.4% [-2.7, -2.1] |
| 16 | compressed dK | `head_unroll` 2 -> 1 | -9.5% [-10.8, -8.1] |

Retained without change: local dK at every batch (all candidates tied or regressed), B16 local dV (`waves_per_eu=2` measured -1.4% [-4.0, +1.3]), B1/B4 dK partial geometry.

Rejected by measurement: local dV `block_n` 128 (+207% [+180, +237]) and 256 (+669%), `block_m` 64 (+20% to +102%), `num_warps` 4 (+9% to +200% depending on batch), `head_unroll` 2 (+23% [+13, +35]) and 4 (+31%). Compressed dV `block_n` 16 (+20%) and 64 (+49%), `num_warps` 4 (+164%), `block_m` 8/32/128 (regressions). Compressed dK `block_n` 16 (+45%) and 64 (+175%), `block_m` 8 (+73%) and 32 (+57%).

Large `block_m`/`head_unroll` combinations were discarded as non-compiling: the `block_m=128, head_unroll=2` candidate failed with Triton `OutOfResources: shared memory` (required 131072 B, hardware limit 65536 B), consistent with the 64 KiB LDS contract.

### Stage 1b: stride diagnostic

A padded BSHD `dO` (head extent 520, sequence stride 33280 elements instead of 32768) recovered 9.6% [8.4, 10.9] of the local-dV gap but made `_compressed_dv_partial_kernel` 114.6% slower, so no single padded staging layout helps every owner. The BHSD-backed view recovered the full 23.8% local and 34.0% compressed gap, making it the only complete staging layout. It was then measured end-to-end and rejected in Stage 2 below because the permute copy consumes the savings.

### Final production medians after Stage 1 (median of 20)

| Batch | Forward | Backward | Complete | Backward delta vs Stage 0 |
|---:|---:|---:|---:|---:|
| 1 | 15.351 ms | 25.295 ms | 40.646 ms | -8.8% |
| 4 | 57.585 ms | 99.617 ms | 157.202 ms | -4.1% |
| 16 | 108.655 ms | 341.196 ms | 449.851 ms | -0.8% |

Forward is untouched by these configs and moved by up to 5% between runs, which measures the complete-boundary run-to-run noise. The remaining gap is the local dV owner (still about 1.7x its BHSD-backed cost at B1) and the compressed dV partial owner (about 1.5x). A backward-side transpose of `dO` was implemented and measured in Stage 2 below. It is rejected by the complete-boundary gate.

### Fail-closed guard added

`_launch_compressed_partial` now rejects `block_n` that does not divide `_COMPRESSED_LENGTH`. A sweep candidate with `block_n=32` for HCA silently corrupted gradients (cosine 0.799). The same class of misconfiguration in CSA is now rejected before launch.

## Stage 2: backward dO staging (rejected by measurement)

A whole-backward staged copy of `dO` into the BHSD-backed order was implemented and measured, then rejected and reverted. No staging code remains in production.

### Permute kernel

A Triton row-permute copies contiguous BSHD `dO` into a BHSD-backed buffer. Source rows advance in memory order (`sequence * QUERY_HEADS + head`) and each `HEAD_DIM`-wide run is written to destination row (`head * SEQUENCE_LENGTH + sequence`). The staged view was passed to the whole backward, so the copy is paid once per layer and the query owner reads the same fast layout. Every tested variant produced bitwise-identical gradients.

| Permute implementation (1 GiB of traffic per pass) | Time | Bandwidth |
|---|---:|---:|
| `transpose(1, 2).contiguous()` (torch) | 3.76 ms | 66 GB/s |
| gather (strided reads, contiguous writes) | 2.00 ms | 125 GB/s |
| scatter (contiguous reads, permuted writes) | 1.29 ms | 193 GB/s |
| scatter with `.cg` load / `.cs` or `.wt` store | 1.28 ms | 196 GB/s |
| contiguous clone (reference) | 1.25 ms | 200 GB/s |

The gfx1151 read+write copy bandwidth is the ceiling. Cache modifiers do not move it. The staged buffer adds 128 MiB / 512 MiB / 2 GiB of transient allocation per layer at B1/B4/B16.

### Complete-boundary result (paired, 8 rounds)

| Batch | Staging off | Staging on | Gap |
|---:|---:|---:|---:|
| 1 | 40.399 ms | 40.008 ms | -0.90% [-1.39, -0.41] |
| 4 | 157.157 ms | 155.702 ms | -0.95% [-1.40, -0.49] |
| 16 | 448.295 ms | 455.520 ms | +1.57% [+1.27, +1.87] |

Kernel-level paired gaps (6 profiled rounds) were `_local_dv_kernel` -24.8% / -33.3% / -36.2%, `_compressed_dv_partial_kernel` -34.6% / -38.0% / -24.4%, and `_csa_dq_kernel` +0.8% / -1.28% / +0.3% at B1/B4/B16. The dV savings are consistently large. The copy cost consumes them at B1/B4 and exceeds them at B16.

### Decision

Rejected. The ~1% B1/B4 complete gain does not justify a batch-gated production behavior, the +2 GiB B16 transient, or the +128/512 MiB transient at B1/B4 (the documented component incremental peaks are 451.6 / 1806.3 / 6200.5 MiB). The copy-free alternative is to rotate in BHSD and let the grouped output-A projection consume the BHSD tensor directly: that removes the round trip and the staging copy, but its grouped-MMQ activation reads become strided and it must be measured at the complete producer-plus-attention boundary before adoption. Fusing the staging into the query owner is not possible in the accepted order: the key owners run before the query owner because the query owner overwrites the raw scores with `dS`, and a second score/`dS` state is explicitly rejected.
