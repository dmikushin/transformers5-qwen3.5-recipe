from typing import Any, cast

import torch
from peft import LoraConfig
from transformers.integrations.gguf import (
    ALL_GGUF_EXPERTS_FUNCTIONS,
    DeepseekV4GGUFExperts,
)

from deepseek_v4_lora import DEEPSEEK_V4_TARGET_MODULES_PATTERN
from fast_moe_lora import (
    FastGGUFMoeLora,
    _ExpertLoraWeights,
    qwen3_5_moe_gguf_mmq_aiter_lora_forward,
)

EXPERTS_IMPLEMENTATION = "deepseek_v4_gguf_mmq_aiter_lora"
_LORA_WEIGHTS_KWARG = "_deepseek_v4_gguf_lora_weights"


def _bind_deepseek_expert_priors(
    model: torch.nn.Module,
    fallback_prior: str,
) -> dict[str, int]:
    """Bind the route-law prior to each DeepSeek routed-expert module."""

    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    counts = {"deepseek-learned": 0, "deepseek-hash": 0}

    for name, module in base.named_modules():
        if not isinstance(module, DeepseekV4GGUFExperts):
            continue
        prior = module.__dict__.get("_aiter_expert_prior", fallback_prior)
        block_name, separator, suffix = name.rpartition(".experts")
        if separator and not suffix:
            block = base.get_submodule(block_name)
            is_hash = getattr(block, "is_hash", None)
            if isinstance(is_hash, bool):
                prior = "deepseek-hash" if is_hash else "deepseek-learned"
            else:
                gate = getattr(block, "gate", None)
                gate_name = type(gate).__name__
                if gate_name == "DeepseekV4HashRouter":
                    prior = "deepseek-hash"
                elif gate_name == "DeepseekV4TopKRouter":
                    prior = "deepseek-learned"
        if prior not in counts:
            raise ValueError(
                f"DeepSeek expert module {name!r} has unsupported prior {prior!r}."
            )
        module.__dict__["_aiter_expert_prior"] = prior
        counts[prior] += 1

    model_type = getattr(getattr(base, "config", None), "model_type", None)
    if model_type == "deepseek_v4" and counts != {
        "deepseek-learned": 40,
        "deepseek-hash": 3,
    }:
        raise RuntimeError(
            "expected 40 learned and 3 hash DeepSeek expert modules, found "
            f"{counts['deepseek-learned']} and {counts['deepseek-hash']}"
        )
    return counts


class DeepseekV4GGUFMoeLora(FastGGUFMoeLora):
    """PEFT wrapper for all gate, up, and down transforms of one MoE layer."""

    def forward(
        self, hidden_states: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        adapter_names = kwargs.pop("adapter_names", None)
        lora_weights = self._active_lora_weights(adapter_names)
        experts = self.get_base_layer()
        if not isinstance(experts, DeepseekV4GGUFExperts):
            raise TypeError(
                "DeepSeek V4 expert LoRA requires DeepseekV4GGUFExperts, got "
                f"{type(experts).__name__}."
            )
        if experts.config._experts_implementation != EXPERTS_IMPLEMENTATION:
            raise RuntimeError(
                f"DeepSeek V4 expert LoRA requires experts_implementation={EXPERTS_IMPLEMENTATION!r}, "
                f"got {experts.config._experts_implementation!r}."
            )
        kwargs[_LORA_WEIGHTS_KWARG] = lora_weights
        return self.base_layer(hidden_states, *args, **kwargs)


def deepseek_v4_gguf_mmq_aiter_lora_forward(
    self: Any,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
    _deepseek_v4_gguf_lora_weights: _ExpertLoraWeights | None = None,
) -> torch.Tensor:
    """Run packed GGTensile base MMQ and AITER LoRA grouped MM."""

    if not isinstance(self, DeepseekV4GGUFExperts):
        raise TypeError(
            f"{EXPERTS_IMPLEMENTATION} requires DeepseekV4GGUFExperts, got {type(self).__name__}."
        )
    return qwen3_5_moe_gguf_mmq_aiter_lora_forward(
        self,
        hidden_states,
        top_k_index,
        top_k_weights,
        _qwen3_5_moe_gguf_lora_weights=_deepseek_v4_gguf_lora_weights,
    )


def register_deepseek_v4_moe_lora(
    lora_config: LoraConfig,
    model: torch.nn.Module,
    *,
    expert_prior: str,
) -> LoraConfig:
    """Register the DeepSeek backend and bind a prior per routed layer."""

    if expert_prior not in {"deepseek-learned", "deepseek-hash"}:
        raise ValueError(
            "DeepSeek expert registration requires prior='deepseek-learned' or "
            "prior='deepseek-hash'."
        )
    register = getattr(lora_config, "_register_custom_module", None)
    if register is None:
        raise RuntimeError(
            "This PEFT version has no LoraConfig._register_custom_module API."
        )
    if lora_config.target_parameters:
        raise ValueError(
            "DeepSeek GGUF experts target complete modules, not parameters."
        )
    if isinstance(lora_config.target_modules, str):
        if lora_config.target_modules != DEEPSEEK_V4_TARGET_MODULES_PATTERN:
            raise ValueError(
                "DeepSeek V4 LoRA uses one exact indexer-excluding target pattern."
            )
    else:
        target_modules = set(lora_config.target_modules or ())
        target_modules.add("experts")
        lora_config.__dict__["target_modules"] = target_modules
    ALL_GGUF_EXPERTS_FUNCTIONS[EXPERTS_IMPLEMENTATION] = (
        deepseek_v4_gguf_mmq_aiter_lora_forward
    )
    cast(Any, model).set_experts_implementation(EXPERTS_IMPLEMENTATION)
    lora_config.__dict__["_aiter_expert_prior_counts"] = _bind_deepseek_expert_priors(
        model, expert_prior
    )
    register({DeepseekV4GGUFExperts: DeepseekV4GGUFMoeLora})
    return lora_config
