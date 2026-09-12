"""Architecture categories for DeepSeek V4 full-step profiling."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from training_profiler import lora_module_category, module_ranges
from training_profiler import profile_warmed_training_update as _profile

_PACKED_LORA_CLASSES = frozenset({"DeepseekV4GgufLoraLinear"})
_GENERIC_LORA_CLASSES = frozenset({"DeepseekV4LoraLinear"})
_EXPERT_LORA_CLASSES = frozenset({"DeepseekV4GgufMoeLora"})
_LORA_ROLE_FRAGMENTS = (
    (".shared_experts.", "shared_expert"),
    (".o_b_proj", "output_b"),
)


def _module_category(name: str, module: torch.nn.Module) -> str | None:
    class_name = type(module).__name__
    if class_name in {"DeepseekV4HyperConnection", "DeepseekV4HyperHead"}:
        return "mhc"
    if "RMSNorm" in class_name:
        return "rmsnorm"
    if class_name == "DeepseekV4Indexer":
        return "indexer"
    if class_name in {"DeepseekV4CompressedKV", "DeepseekV4HeavilyCompressedKV"}:
        return "compressor"
    if class_name == "DeepseekV4Attention":
        return "attention"
    if class_name == "GgufGroupedLinear":
        return "grouped_output_a"
    return lora_module_category(
        name,
        class_name,
        packed_lora_classes=_PACKED_LORA_CLASSES,
        generic_lora_classes=_GENERIC_LORA_CLASSES,
        expert_lora_classes=_EXPERT_LORA_CLASSES,
        role_fragments=_LORA_ROLE_FRAGMENTS,
    )


def deepseek_v4_module_ranges(model: torch.nn.Module):
    return module_ranges(model, _module_category)


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
