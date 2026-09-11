"""DeepSeek V4 Liger losses, including the packed Q8_0 LM-head path.

The reference helper intentionally remains available for correctness audits. The
model-instance patch uses the packed helper by default: it quantizes bounded
BF16 hidden-state chunks, runs Liger cross-entropy in place, and computes the
frozen Q8_0 head's input gradient directly from the packed payload. No logical
vocabulary matrix or full-sequence logits tensor is created during training.
"""

from types import MethodType
from typing import Any, cast

import torch
import triton
from liger_kernel.ops.cross_entropy import liger_cross_entropy_kernel
from liger_kernel.ops.fused_linear_cross_entropy import MAX_FUSED_SIZE
from liger_kernel.ops.utils import (
    amp_custom_bwd,
    amp_custom_fwd,
    element_mul_kernel,
    is_hip,
)
from liger_kernel.transformers.model.loss_utils import LigerForCausalLMLoss
from torch import nn
from torch_ggml_ops import mmq_grad_input_inplace, mmq_inplace
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear
from transformers.modeling_outputs import MoeCausalLMOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4ForCausalLM,
    load_balancing_loss_func,
)

_PACKED_LM_HEAD_CHUNK_SIZE = 512


def _scale_input_gradient_in_place(
    grad_input: torch.Tensor, grad_output: torch.Tensor
) -> None:
    """Multiply a retained hidden-state gradient by the scalar loss gradient.

    Liger's ``fused_linear_cross_entropy_backward`` opens with
    ``torch.equal(grad_output, torch.tensor(1.0, ...))``, which synchronizes the
    device on every step. That test only decides whether this multiply can be
    skipped, so running it unconditionally is cheaper than the stall.
    """

    rows, hidden = grad_input.shape
    element_mul_kernel[(rows,)](
        grad_input,
        grad_input.stride(-2),
        grad_output,
        hidden,
        BLOCK_SIZE=min(MAX_FUSED_SIZE, triton.next_power_of_2(hidden)),
        num_warps=32 if not is_hip() else 16,
    )


def deepseek_v4_liger_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    *,
    hidden_size: int,
    loss_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Reference loss that materializes the frozen head for comparison tests."""

    if lm_head.weight.requires_grad:
        raise RuntimeError("DeepSeek V4 reference loss requires a frozen LM head.")
    loss_kwargs = {} if loss_kwargs is None else loss_kwargs
    logical_weight = lm_head.materialize_logical_weight(
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    loss = LigerForCausalLMLoss(
        hidden_states=hidden_states,
        lm_head_weight=logical_weight,
        labels=labels,
        hidden_size=hidden_size,
        **loss_kwargs,
    )
    if not isinstance(loss, torch.Tensor):
        raise TypeError(
            f"Liger causal-LM loss returned {type(loss).__name__}, expected Tensor."
        )
    return loss


def _packed_q8_linear_cross_entropy_forward(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    target: torch.Tensor,
    quant_type: int,
    out_features: int,
    chunk_size: int,
    ignore_index: int,
    lse_square_scale: float,
    label_smoothing: float,
    softcap: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run chunked Q8_1 MMQ, in-place Liger CE, and packed dHidden.

    The returned loss is the un-normalized sum over rows. The caller applies the
    mean reduction on device, so the non-ignored count never reaches the host.
    """

    if input.ndim != 2:
        raise ValueError(
            f"Packed LM-head loss expects two-dimensional hidden states, got {input.shape}."
        )
    if input.dtype != torch.bfloat16:
        raise TypeError(
            f"Packed LM-head loss requires BF16 hidden states, got {input.dtype}."
        )
    if target.ndim != 1 or target.shape[0] != input.shape[0]:
        raise ValueError(
            "Packed LM-head targets must be one-dimensional and match the hidden-state row count."
        )
    if target.dtype != torch.int64:
        raise TypeError(
            f"Packed LM-head targets require torch.int64, got {target.dtype}."
        )
    if chunk_size <= 0:
        raise ValueError(
            f"Packed LM-head chunk_size must be positive, got {chunk_size}."
        )

    rows = input.shape[0]
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(out_features))

    grad_input = torch.empty_like(input)
    hidden_size = input.shape[1]
    buffer_rows = min(rows, chunk_size)
    logits_buffer = torch.empty(
        (buffer_rows, out_features), dtype=input.dtype, device=input.device
    )
    workspace_buffer = torch.empty(
        buffer_rows * hidden_size // 128 * 144,
        dtype=torch.uint8,
        device=input.device,
    )
    loss_1d = torch.zeros(rows, dtype=torch.float32, device=input.device)
    for start in range(0, rows, chunk_size):
        end = min(start + chunk_size, rows)
        input_chunk = input[start:end]
        chunk_rows = end - start
        logits_chunk = logits_buffer[:chunk_rows]
        workspace_chunk = workspace_buffer[: chunk_rows * hidden_size // 128 * 144]
        mmq_inplace(
            input_chunk,
            packed_weight,
            quant_type,
            out_features,
            logits_chunk,
            workspace_chunk,
        )
        target_chunk = target[start:end].contiguous()
        loss_1d_slice = loss_1d[start:end]

        liger_cross_entropy_kernel[(end - start,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=None,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=None,
            loss_stride=loss_1d_slice.stride(-1),
            token_accuracy_ptr=None,
            token_accuracy_stride=0,
            predicted_tokens_ptr=None,
            predicted_tokens_stride=0,
            n_cols=out_features,
            n_non_ignore=1,
            sum_non_ignore_weight=1,
            weight_sum=0.0,
            ignore_index=ignore_index,
            lse_square_scale=lse_square_scale,
            label_smoothing=label_smoothing,
            reduction="sum",
            softcap=softcap,
            RETURN_Z_LOSS=False,
            RETURN_TOKEN_ACCURACY=False,
            RETURN_PREDICTED_TOKENS=False,
            HAS_WEIGHT=False,
            HAS_SOFTCAPPING=softcap is not None,
            HAS_GRADIENTS=True,
            BLOCK_SIZE=block_size,
            num_warps=16 if is_hip() else 32,
        )
        mmq_grad_input_inplace(
            logits_chunk,
            packed_weight,
            quant_type,
            hidden_size,
            grad_input[start:end],
        )

    return torch.sum(loss_1d), grad_input


class _PackedQ8LigerLinearCrossEntropyFunction(torch.autograd.Function):
    """Liger-style fused loss over a frozen packed Q8_0 GGUF LM head."""

    @staticmethod
    @amp_custom_fwd
    def forward(
        ctx,
        input: torch.Tensor,
        packed_weight: torch.Tensor,
        target: torch.Tensor,
        quant_type: int,
        out_features: int,
        chunk_size: int,
        ignore_index: int,
        lse_square_scale: float,
        label_smoothing: float,
        softcap: float | None,
    ):
        loss, grad_input = _packed_q8_linear_cross_entropy_forward(
            input,
            packed_weight,
            target,
            quant_type,
            out_features,
            chunk_size,
            ignore_index,
            lse_square_scale,
            label_smoothing,
            softcap,
        )
        ctx.save_for_backward(grad_input.detach())
        return loss

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output):
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Packed Q8_0 GGUF LM-head loss does not support higher-order gradients"
            )
        (grad_input,) = ctx.saved_tensors
        _scale_input_gradient_in_place(grad_input, grad_output)
        return (grad_input,) + (None,) * 9


def _packed_q8_liger_for_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    hidden_size: int,
    num_items_in_batch: int | torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    final_logit_softcapping: float | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the optimized DeepSeek Q8_0 causal-LM loss."""

    unsupported = (
        ("return_z_loss", kwargs.get("return_z_loss", False)),
        ("use_token_scaling", kwargs.get("use_token_scaling", False)),
        ("ce_weight", kwargs.get("ce_weight") is not None),
        ("bias", kwargs.get("bias") is not None),
        ("label_smoothing", float(kwargs.get("label_smoothing", 0.0)) != 0.0),
        ("lse_square_scale", float(kwargs.get("lse_square_scale", 0.0)) != 0.0),
        ("final_logit_softcapping", final_logit_softcapping is not None),
    )
    for name, enabled in unsupported:
        if enabled:
            raise RuntimeError(
                f"Packed DeepSeek Q8_0 LM-head loss does not support {name}."
            )
    accum_dtype = kwargs.get("accum_dtype")
    if accum_dtype not in {None, torch.float32}:
        raise RuntimeError(
            "Packed DeepSeek Q8_0 LM-head loss supports only FP32 internal accumulation."
        )
    if not isinstance(lm_head.weight, GgufQuantizedParameter):
        raise TypeError(
            "Packed DeepSeek LM-head loss requires a GgufQuantizedParameter."
        )
    if lm_head.input_permutation is not None or lm_head.output_permutation is not None:
        raise RuntimeError(
            "Packed DeepSeek Q8_0 LM-head loss does not support layout permutations."
        )
    if lm_head.compute_dtype != torch.bfloat16:
        raise RuntimeError(
            "Packed DeepSeek Q8_0 LM-head loss requires BF16 compute_dtype."
        )
    if lm_head.in_features != hidden_size or hidden_states.shape[-1] != hidden_size:
        raise RuntimeError(
            "Packed DeepSeek Q8_0 LM-head loss hidden size does not match the LM head."
        )
    quant_type = int(cast(Any, lm_head.weight.quant_type))
    if quant_type != 8:
        raise RuntimeError(
            "Packed DeepSeek LM-head loss is validated only for Q8_0 weights."
        )
    if lm_head.weight.requires_grad:
        raise RuntimeError("Packed DeepSeek LM-head weights must remain frozen.")

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    hidden_states = hidden_states.reshape(-1, hidden_size)
    shift_labels = shift_labels.reshape(-1).to(hidden_states.device)
    payload = lm_head.weight.as_subclass(torch.Tensor)
    loss = _PackedQ8LigerLinearCrossEntropyFunction.apply(
        hidden_states,
        payload,
        shift_labels,
        quant_type,
        lm_head.out_features,
        _PACKED_LM_HEAD_CHUNK_SIZE,
        ignore_index,
        0.0,
        0.0,
        None,
    )
    # The kernel produces an un-normalized sum, and the reduction is applied here
    # on device so the non-ignored count never round-trips through the host. As a
    # plain tensor division it also scales the backward through autograd.
    denominator = (
        num_items_in_batch
        if num_items_in_batch is not None
        else (shift_labels != ignore_index).sum()
    )
    return loss / denominator


def deepseek_v4_packed_liger_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    *,
    hidden_size: int,
    loss_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Public scoped packed loss used by the optimized model forward."""

    return _packed_q8_liger_for_causal_lm_loss(
        hidden_states,
        lm_head,
        labels,
        hidden_size=hidden_size,
        **({} if loss_kwargs is None else loss_kwargs),
    )


def _deepseek_v4_liger_forward(
    self: DeepseekV4ForCausalLM,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Any | None = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Any,
) -> MoeCausalLMOutputWithPast:
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state
    loss = None
    logits = None
    if labels is None:
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
    else:
        loss = deepseek_v4_packed_liger_causal_lm_loss(
            hidden_states,
            cast(GgufLinear, self.lm_head),
            labels,
            hidden_size=self.config.hidden_size,
            loss_kwargs=kwargs,
        )

    aux_loss = None
    if output_router_logits:
        aux_loss = cast(
            torch.Tensor,
            load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            ),
        )
        if labels is not None:
            if loss is None:
                raise RuntimeError("router auxiliary loss requires a primary loss")
            loss = loss + self.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithPast(
        loss=cast(Any, loss),
        aux_loss=cast(Any, aux_loss),
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def apply_deepseek_v4_liger_loss(
    model: torch.nn.Module,
) -> DeepseekV4ForCausalLM:
    """Patch one model instance with the packed Q8_0 scoped loss."""

    get_base_model = getattr(model, "get_base_model", None)
    base = get_base_model() if callable(get_base_model) else model
    if not isinstance(base, DeepseekV4ForCausalLM):
        raise TypeError(
            "DeepSeek V4 Liger loss requires DeepseekV4ForCausalLM, got "
            f"{type(base).__name__}."
        )
    if not isinstance(base.lm_head, GgufLinear):
        raise TypeError(
            "DeepSeek V4 GGUF loss requires GgufLinear lm_head, got "
            f"{type(base.lm_head).__name__}."
        )
    if getattr(base, "_deepseek_v4_liger_loss_enabled", False):
        return base
    base.forward = MethodType(_deepseek_v4_liger_forward, base)
    base.__dict__["_deepseek_v4_liger_loss_enabled"] = True
    return base
