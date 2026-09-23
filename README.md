# Low-VRAM LoRA Training With GGUF Base Model

GGUF is going to replace bitsandbytes as the base model format for low-VRAM LoRA training. I've tried to train, with no CPU offload:
- Qwen3.6-35B-A3B in 16 GiB VRAM (implying it's more than enough to train Qwen3.5-122B-A10B in 64 GiB, and Qwen3.5-397B-A17B in 192 GiB)
- DeepSeek-V4-Flash (284B-A13B) in 90 GiB VRAM

Currently all kernels and parameters in this repo are tuned for Strix Halo. It should not be too hard to port to other GPUs.

Things involved in the training:
- Usual training loop with transformers 5 and PEFT
- Transformers with GGUF quantizer, see https://github.com/woct0rdho/transformers/tree/gguf . I'm tracking this in https://github.com/huggingface/transformers/issues/40070 . If I could not merge it into transformers in the end, I'll reimplement it as some monkey patches in this repo
- Tuned GEMM, see https://github.com/ROCm/rocm-libraries/pull/9385
- MMQ and grouped MMQ like llama.cpp, see https://github.com/woct0rdho/torch-ggml-ops
- Fast LoRA bwd formula like Unsloth for linear layer and MoE layer
- AITER gmm/ptgmm Triton kernels with tuned configs for non-quantized MoE LoRA
- MoE routing like OpenAI triton-kernels
- AITER FlashAttention Triton kernel with tuned configs and the bugfix https://github.com/ROCm/aiter/issues/3551
- RMSNorm from Liger Kernel
- Chunked cross entropy loss like Liger Kernel, which works with MMQ
- Autoregressive decoding cache and load balancing loss disabled to save VRAM
- Non-reentrant gradient checkpointing
- bitsandbytes AdamW 8-bit optimizer

Qwen-specific:
- Qwen3.6-35B-A3B with APEX-I-Mini quantization that only takes 13.3 GiB, see https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF/blob/main/Qwen3.6-35B-A3B-APEX-I-Mini.gguf
- GatedDeltaNet with FLA and causal-conv1d, which are automatically chosen by transformers if installed. I've added tuned configs but I've not yet fully considered how to optimize this

DeepSeek-specific:
- `IQ2_XXS` quantization, see https://huggingface.co/antirez/deepseek-v4-gguf/blob/main/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
- Sliding attention, CSA, HCA, mHC with fast Triton kernels, including bwd

Other notes:
- Unsloth gradient checkpointing provides fast async CPU-GPU copy. You need it if you actually do CPU offload. But on Strix Halo with unified memory you should just use the usual gradient checkpointing
- It does not affect loading GGUF models, but when loading safetensors models on Strix Halo, you need to patch transformers so it passes `backend="pread"` to safetensors, see https://github.com/safetensors/safetensors/pull/728
- When directly using my forked transformers with GGUF quantizer, you may add torch.compile to the GGUF dequant function to save VRAM. This repo always use torch-ggml-ops and does not depend on the GGUF dequant function in torch

## NVIDIA CUDA: dense Qwen3.8-27B on one 24 GB RTX 3090

`train_qwen3_5_27b_cuda.py` trains LoRA (r=16 on q/k/v/o/gate/up/down) on `Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.5 GB, 16.0 GiB on the GPU) with the weights kept packed. It needs torch-ggml-ops with its CUDA backend (sm_80+), which covers every LoRA-target quant type in that checkpoint (Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, IQ3_S, IQ4_NL, IQ4_XS).

Differences from the Strix Halo recipe: SDPA attention instead of AITER, no MoE paths, no gfx1151 Triton tables. `fast_lora` sends a projection to MMQ only if its quant type is in `torch_ggml_ops.DENSE_MMQ_QUANT_TYPES`; `gguf_liger_loss` has a dense `Qwen3_5ForCausalLM` forward (hidden 5120); `qwen3_5_fused_norms` also covers the dense norm classes.

```bash
python train_qwen3_5_27b_cuda.py --text-jsonl data.jsonl --seq-len 20480 --max-steps 5 \
    --offload-checkpoints --tiled-mlp-shards 4
```

`--offload-checkpoints` keeps gradient-checkpoint inputs in pinned host memory (0.625 MiB per token, 12.5 GiB at 20k); `--tiled-mlp-shards` runs each MLP through Liger's TiledMLP. Both were shown necessary by removal: without offload 8k tokens OOM, without TiledMLP 20k (and, before fused norms, 16k) OOM.

Measured on an RTX 3090 (23.61 GiB usable), torch 2.14.0+cu130, batch 1, 5 steps each, steady-state steps 2-5 (step 1 includes Triton JIT). Peak = `torch.cuda.max_memory_allocated`; device-used adds the allocator's reserve and the CUDA context.

| Tokens/step | TiledMLP shards | s/step | tokens/s | peak allocated | device used |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 18.9 | 216 | 17.97 GiB | 18.84 GiB |
| 8192 | 2 | 37.4 | 219 | 18.77 GiB | 19.65 GiB |
| 16384 | 4 | 75.8 | 216 | 21.07 GiB | 22.16 GiB |
| 20480 | 4 | 96.1 | 213 | 22.21 GiB | 23.25 GiB |

24576 tokens OOMs (the transient peak is then inside FLA's chunked gated-delta-rule backward). A 24-step run on two repeated 2048-token windows (lr 3e-4) takes the loss from 2.29 to 0.0046.
