"""Architecture categories for Qwen3.5-MoE full-step profiling."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from training_profiler import profile_warmed_training_update as _profile


def _module_category(name: str, module: torch.nn.Module) -> str | None:
    class_name = type(module).__name__
    if class_name == "Qwen3_5MoeDecoderLayer":
        return "decoder_layer"
    if class_name == "Qwen3_5MoeAttention":
        return "attention"
    if class_name == "Qwen3_5MoeGatedDeltaNet":
        return "gated_delta_net"
    if "RMSNorm" in class_name:
        return "rmsnorm"
    if class_name == "Qwen3_5MoeSparseMoeBlock":
        return "moe_block"
    if class_name == "FastGgufMoeLora":
        return "routed_experts"
    if class_name == "FastGgufLoraLinear":
        return "packed_ordinary_lora"
    if class_name == "FastLoraLinear":
        if ".shared_expert." in name or ".shared_experts." in name:
            return "shared_expert"
        return "ordinary_lora"
    if name.endswith("lm_head"):
        return "packed_lm_head"
    return None


def profile_warmed_training_update(
    model: torch.nn.Module,
    update: Callable[[str], dict[str, Any]],
    *,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return _profile(
        model,
        update,
        output_path=output_path,
        categorize=_module_category,
        metadata=metadata,
    )
