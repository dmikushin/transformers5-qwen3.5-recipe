"""Architecture categories for Qwen3.5-MoE full-step profiling."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from training_profiler import lora_module_category
from training_profiler import profile_warmed_training_update as _profile

_PACKED_LORA_CLASSES = frozenset({"FastGgufLoraLinear"})
_GENERIC_LORA_CLASSES = frozenset({"FastLoraLinear"})
_EXPERT_LORA_CLASSES = frozenset({"FastGgufMoeLora"})
_LORA_ROLE_FRAGMENTS = (
    (".shared_expert.", "shared_expert"),
    (".shared_experts.", "shared_expert"),
)


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
    return lora_module_category(
        name,
        class_name,
        packed_lora_classes=_PACKED_LORA_CLASSES,
        generic_lora_classes=_GENERIC_LORA_CLASSES,
        expert_lora_classes=_EXPERT_LORA_CLASSES,
        role_fragments=_LORA_ROLE_FRAGMENTS,
    )


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
