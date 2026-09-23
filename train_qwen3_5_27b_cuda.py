#!/usr/bin/env python3
"""LoRA on dense Qwen3.8-27B GGUF (UD-Q4_K_XL) on one 24 GB NVIDIA GPU.

CUDA counterpart of ``train_qwen3_5_35b.py``. What differs, and why:

* Attention: PyTorch SDPA instead of AITER FlashAttention (ROCm-only).
* No MoE: the model is dense, so the grouped-MMQ LoRA, expert ranking and
  AITER gmm configs are not used.
* No gfx1151 Triton autotune tables (``fla_tuning``, AITER configs).
* The GGUF weights stay packed. LoRA-target projections whose quant type has
  a CUDA MMQ kernel run through ``torch_ggml_ops.mmq`` in both directions;
  the remaining projections (GatedDeltaNet in/out projections and the few
  Q3_K / IQ3_S tensors) use the transformers fork's per-call dequantization.
* The LM head (Q6_K, 248320 x 5120) is never materialized: the packed
  chunked Liger loss from ``gguf_liger_loss`` computes loss and dHidden.
* ``--offload-checkpoints`` keeps the per-layer checkpointed hidden states
  (64 x seq_len x 5120 BF16, 0.625 MiB per token) in pinned host memory via
  the transformers fork's ``gradient_checkpointing_enable(offload=True)``.
  With 16 GiB of packed weights this is what bounds the context length.
* ``--tiled-mlp-shards N`` runs every MLP through Liger's TiledMLP (DeepSpeed
  ``TiledMLP``): N sequence shards in forward, each recomputed and
  back-propagated separately in backward. This divides the [seq_len, 17408]
  intermediates that dominate the backward peak by N, for one extra MLP
  forward per layer and step.

Training data is real text packed into fixed-length sequences so that the
context length of every step is exactly ``--seq-len`` tokens.

    python train_qwen3_5_27b_cuda.py --seq-len 8192 --max-steps 4 \\
        --text-jsonl ~/scratch/priority-net/state/pairs.jsonl

Per-step wall time, tokens/s and the CUDA allocator's peak are printed and
written to ``<output-dir>/step_metrics.jsonl``.
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import Dataset
from liger_kernel.ops.tiled_mlp import apply_tiled_mlp
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

from bf16_adapter_trainer import BF16AdapterTrainer
from fast_lora import FastGgufLoraLinear, register_fast_lora
from gguf_dequant_compile import configure_compiled_gguf_dequantize
from gguf_liger_loss import apply_gguf_liger_fused_linear_cross_entropy

_DEFAULT_GGUF = (
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/*/"
    "Qwen3.8-27B-UD-Q4_K_XL.gguf"
)
_GIB = 2**30


def _packed_dataset(tokenizer, jsonl: Path, seq_len: int, samples: int) -> Dataset:
    """Concatenate prompt/reply text and cut it into ``samples`` full windows."""

    ids: list[int] = []
    needed = seq_len * samples
    with jsonl.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            text = "\n".join(str(record[key]) for key in ("prompt", "reply") if key in record)
            ids.extend(tokenizer(text, add_special_tokens=False).input_ids)
            ids.append(tokenizer.eos_token_id)
            if len(ids) >= needed:
                break
    if len(ids) < needed:
        raise RuntimeError(
            f"{jsonl} yields {len(ids)} tokens, fewer than {samples} x {seq_len}"
        )
    windows = [ids[i * seq_len : (i + 1) * seq_len] for i in range(samples)]
    return Dataset.from_dict({"input_ids": windows})


def _configure_tiled_mlp(model: torch.nn.Module, shards: int) -> int:
    """Route every decoder MLP through TiledMLP with a fixed shard count."""

    patched = 0
    for layer in model.model.layers:
        mlp = layer.mlp
        forward = type(mlp).forward

        def tiled_forward(x, mlp=mlp, forward=forward):
            return apply_tiled_mlp(forward, mlp, x, num_shards=shards)

        mlp.forward = tiled_forward
        patched += 1
    return patched


def _collate(examples):
    input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": input_ids.clone(),
    }


class _StepMetrics(TrainerCallback):
    """Wall time, tokens/s and allocator peaks per optimizer step."""

    def __init__(self, tokens_per_step: int, path: Path, snapshot: Path | None) -> None:
        self.tokens_per_step = tokens_per_step
        self.path = path
        self.snapshot = snapshot
        self.started = 0.0

    def _tracing(self, state) -> bool:
        # The second step: the first one includes Triton JIT and allocator warm-up.
        return self.snapshot is not None and state.global_step == 1

    def on_step_begin(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if self._tracing(state):
            torch.cuda.memory._record_memory_history(max_entries=2_000_000, stacks="python")
        self.started = time.perf_counter()

    def _tracing_step_end(self, state) -> bool:
        return self.snapshot is not None and state.global_step == 2

    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.synchronize()
        seconds = time.perf_counter() - self.started
        free, total = torch.cuda.mem_get_info()
        record = {
            "step": state.global_step,
            "seconds": round(seconds, 3),
            "tokens_per_second": round(self.tokens_per_step / seconds, 1),
            "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / _GIB, 3),
            "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / _GIB, 3),
            "device_used_gib_after_step": round((total - free) / _GIB, 3),
        }
        if self._tracing_step_end(state):
            torch.cuda.memory._dump_snapshot(str(self.snapshot))
            torch.cuda.memory._record_memory_history(enabled=None)
            print(f"allocator trace written to {self.snapshot}", flush=True)
        print("step_metrics", json.dumps(record), flush=True)
        with self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--gguf", default=_DEFAULT_GGUF, help="GGUF file (glob allowed)")
    parser.add_argument("--text-jsonl", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--output-dir", type=Path, default=Path("out_qwen38_27b_cuda"))
    parser.add_argument(
        "--compile-dequant",
        action="store_true",
        help="torch.compile the fork's dequantizer used by the non-MMQ projections",
    )
    parser.add_argument(
        "--offload-checkpoints",
        action="store_true",
        help="hold gradient-checkpoint inputs in pinned host memory",
    )
    parser.add_argument(
        "--tiled-mlp-shards",
        type=int,
        default=0,
        help="run each MLP in this many sequence shards (0: off)",
    )
    parser.add_argument(
        "--memory-snapshot",
        type=Path,
        help="write a CUDA allocator trace of the second step to this pickle",
    )
    parser.add_argument("--seed", type=int, default=19260817)
    args = parser.parse_args()

    matches = sorted(glob.glob(os.path.expanduser(args.gguf)))
    if not matches:
        raise FileNotFoundError(args.gguf)
    gguf_path = Path(matches[0])
    set_seed(args.seed)
    if args.compile_dequant:
        configure_compiled_gguf_dequantize()

    tokenizer = AutoTokenizer.from_pretrained(
        gguf_path.parent, gguf_file=gguf_path.name, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        gguf_path.parent,
        gguf_file=gguf_path.name,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": "cuda:0"},
    )
    model.config.use_cache = False
    # Enabled here rather than through TrainingArguments, which cannot request
    # offloading. The Trainer does not touch checkpointing when its flag is off.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False},
        offload=args.offload_checkpoints,
    )
    if args.tiled_mlp_shards > 0:
        count = _configure_tiled_mlp(model, args.tiled_mlp_shards)
        print(f"TiledMLP with {args.tiled_mlp_shards} shards on {count} MLPs", flush=True)
    print(f"loaded {type(model).__name__}: {torch.cuda.memory_allocated() / _GIB:.2f} GiB allocated", flush=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
    )
    register_fast_lora(lora_config, model)
    model = get_peft_model(model, lora_config, autocast_adapter_dtype=False)
    apply_gguf_liger_fused_linear_cross_entropy(model)
    model.print_trainable_parameters()

    wrappers = [m for m in model.modules() if isinstance(m, FastGgufLoraLinear)]
    on_mmq = sum(m.uses_packed_mmq() for m in wrappers)
    print(f"LoRA projections on packed MMQ: {on_mmq} / {len(wrappers)}", flush=True)

    samples = args.max_steps * args.grad_accum
    dataset = _packed_dataset(tokenizer, args.text_jsonl.expanduser(), args.seq_len, samples)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "step_metrics.jsonl"
    metrics_path.unlink(missing_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=False,  # enabled on the model above
        remove_unused_columns=False,
        report_to=[],
        seed=args.seed,
        dataloader_pin_memory=False,
    )
    trainer = BF16AdapterTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=_collate,
        callbacks=[
            _StepMetrics(args.seq_len * args.grad_accum, metrics_path, args.memory_snapshot)
        ],
    )
    trainer.train()


if __name__ == "__main__":
    main()
