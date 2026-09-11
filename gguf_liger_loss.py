"""Packed GGUF-aware Liger-style cross-entropy for Qwen3.5-MoE.

The frozen LM head remains in its authoritative GGUF representation. Each
hidden-state chunk is quantized to Q8_1 by native MMQ, Liger's cross-entropy
kernel converts its BF16 logits to cotangents in place, and the packed logical
input Jacobian is computed before model backward. No logical LM-head matrix or
full-sequence logits tensor is materialized.
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
from liger_kernel.transformers.model.loss_utils import unpack_cross_entropy_result
from liger_kernel.transformers.model.output_classes import (
    LigerMoeCausalLMOutputWithPast,
)
from torch import nn
from torch_ggml_ops import mmq_grad_input_inplace, mmq_inplace
from transformers.integrations.gguf.gguf_quantized_parameter import (
    GgufQuantizedParameter,
)
from transformers.integrations.gguf.modules import GgufLinear
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeForCausalLM,
    load_balancing_loss_func,
)

_PACKED_LM_HEAD_CHUNK_SIZE = 256


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
    return_token_accuracy: bool,
    return_predicted_tokens: bool,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor,
]:
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

    rows, hidden_size = input.shape
    # Kept on device: dividing the accuracy by a tensor stays off the host, and
    # the kernel's reduction no longer reads this count.
    total_n_non_ignore = (target != ignore_index).sum()
    block_size = min(MAX_FUSED_SIZE, triton.next_power_of_2(out_features))

    grad_input = torch.empty_like(input)
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
    token_accuracy_1d = (
        torch.zeros(rows, dtype=torch.float32, device=input.device)
        if return_token_accuracy
        else None
    )
    predicted_tokens_1d = (
        torch.full((rows,), -1, dtype=torch.int64, device=input.device)
        if return_predicted_tokens
        else None
    )

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
        if return_token_accuracy:
            if token_accuracy_1d is None:
                raise RuntimeError("token accuracy storage was not allocated")
            token_accuracy_1d_slice = token_accuracy_1d[start:end]
        else:
            token_accuracy_1d_slice = None
        if return_predicted_tokens:
            if predicted_tokens_1d is None:
                raise RuntimeError("predicted-token storage was not allocated")
            predicted_tokens_1d_slice = predicted_tokens_1d[start:end]
        else:
            predicted_tokens_1d_slice = None
        token_accuracy_stride = (
            token_accuracy_1d_slice.stride(-1)
            if token_accuracy_1d_slice is not None
            else 0
        )
        predicted_tokens_stride = (
            predicted_tokens_1d_slice.stride(-1)
            if predicted_tokens_1d_slice is not None
            else 0
        )

        liger_cross_entropy_kernel[(end - start,)](
            X_ptr=logits_chunk,
            X_stride=logits_chunk.stride(-2),
            Y_ptr=target_chunk,
            Y_stride=target_chunk.stride(-1),
            weight_ptr=None,
            loss_ptr=loss_1d_slice,
            z_loss_ptr=None,
            loss_stride=loss_1d_slice.stride(-1),
            token_accuracy_ptr=token_accuracy_1d_slice,
            token_accuracy_stride=token_accuracy_stride,
            predicted_tokens_ptr=predicted_tokens_1d_slice,
            predicted_tokens_stride=predicted_tokens_stride,
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
            RETURN_TOKEN_ACCURACY=return_token_accuracy,
            RETURN_PREDICTED_TOKENS=return_predicted_tokens,
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

    loss = torch.sum(loss_1d)
    if return_token_accuracy:
        if token_accuracy_1d is None:
            raise RuntimeError("token accuracy storage was not allocated")
        token_accuracy = torch.sum(token_accuracy_1d) / total_n_non_ignore
    else:
        token_accuracy = None
    return loss, token_accuracy, predicted_tokens_1d, grad_input


class _PackedQ8LigerLinearCrossEntropyFunction(torch.autograd.Function):
    """Liger-style fused loss over a frozen packed GGUF LM head."""

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
        return_token_accuracy: bool,
        return_predicted_tokens: bool,
    ):
        loss, token_accuracy, predicted_tokens, grad_input = (
            _packed_q8_linear_cross_entropy_forward(
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
                return_token_accuracy,
                return_predicted_tokens,
            )
        )
        ctx.save_for_backward(grad_input.detach())
        non_differentiable = tuple(
            output
            for output in (token_accuracy, predicted_tokens)
            if output is not None
        )
        if non_differentiable:
            ctx.mark_non_differentiable(*non_differentiable)
        return loss, token_accuracy, predicted_tokens

    @staticmethod
    @amp_custom_bwd
    def backward(ctx, grad_output, grad_token_accuracy, grad_predicted_tokens):
        del grad_token_accuracy, grad_predicted_tokens
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Packed Q8_1 GGUF LM-head loss does not support higher-order gradients"
            )
        (grad_input,) = ctx.saved_tensors
        _scale_input_gradient_in_place(grad_input, grad_output)
        return (
            grad_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def _packed_q8_liger_for_causal_lm_loss(
    hidden_states: torch.Tensor,
    lm_head: GgufLinear,
    labels: torch.Tensor,
    hidden_size: int,
    num_items_in_batch: int | torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    final_logit_softcapping: float | None = None,
    return_token_accuracy: bool = False,
    return_predicted_tokens: bool = False,
    **kwargs: Any,
):
    if kwargs.get("return_z_loss", False):
        raise RuntimeError("Packed GGUF LM-head loss does not support z-loss output.")
    if kwargs.get("use_token_scaling", False):
        raise RuntimeError("Packed GGUF LM-head loss does not support token scaling.")
    if kwargs.get("ce_weight") is not None:
        raise RuntimeError("Packed GGUF LM-head loss does not support class weights.")
    if kwargs.get("bias") is not None:
        raise RuntimeError("Packed GGUF LM-head loss does not support a fused bias.")
    if float(kwargs.get("label_smoothing", 0.0)) != 0.0:
        raise RuntimeError("Packed GGUF LM-head loss does not support label smoothing.")
    if float(kwargs.get("lse_square_scale", 0.0)) != 0.0:
        raise RuntimeError(
            "Packed GGUF LM-head loss does not support LSE square scaling."
        )
    if final_logit_softcapping is not None:
        raise RuntimeError(
            "Packed GGUF LM-head loss does not support logit softcapping."
        )
    accum_dtype = kwargs.get("accum_dtype")
    if accum_dtype not in {None, torch.float32}:
        raise RuntimeError(
            "Packed GGUF LM-head loss supports only FP32 internal accumulation."
        )
    if not isinstance(lm_head.weight, GgufQuantizedParameter):
        raise TypeError(
            "Packed GGUF LM-head loss requires a GgufQuantizedParameter weight."
        )
    if lm_head.input_permutation is not None or lm_head.output_permutation is not None:
        raise RuntimeError(
            "Packed GGUF LM-head loss does not support layout permutations."
        )
    if lm_head.compute_dtype != torch.bfloat16:
        raise RuntimeError("Packed GGUF LM-head loss requires BF16 compute_dtype.")
    if lm_head.in_features != hidden_size or hidden_size != 2048:
        raise RuntimeError(
            "Packed GGUF LM-head loss is validated only for hidden size 2048."
        )
    quant_type = int(cast(Any, lm_head.weight.quant_type))
    if quant_type != 14:
        raise RuntimeError(
            "Packed GGUF LM-head loss is validated only for Q6_K weights."
        )
    if lm_head.weight.requires_grad:
        raise RuntimeError("Packed GGUF LM-head weights must remain frozen.")

    if shift_labels is None:
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    hidden_states = hidden_states.view(-1, hidden_size)
    shift_labels = shift_labels.view(-1).to(hidden_states.device)
    payload = lm_head.weight.as_subclass(torch.Tensor)
    loss, token_accuracy, predicted_tokens = (
        _PackedQ8LigerLinearCrossEntropyFunction.apply(
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
            return_token_accuracy,
            return_predicted_tokens,
        )
    )
    # The kernel produces an un-normalized sum, and the reduction is applied here
    # on device so the non-ignored count never round-trips through the host. As a
    # plain tensor division it also scales the backward through autograd.
    denominator = (
        num_items_in_batch
        if num_items_in_batch is not None
        else (shift_labels != ignore_index).sum()
    )
    loss = loss / denominator
    if return_token_accuracy or return_predicted_tokens:
        return loss, None, token_accuracy, predicted_tokens
    return loss


def gguf_liger_lce_forward(
    self: Qwen3_5MoeForCausalLM,
    input_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    past_key_values: Any = None,
    inputs_embeds: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    use_cache: bool | None = None,
    output_attentions: bool | None = None,
    output_hidden_states: bool | None = None,
    output_router_logits: bool | None = None,
    mm_token_type_ids: torch.Tensor | None = None,
    cache_position: torch.Tensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    skip_logits: bool | None = None,
    return_dict: bool | None = None,
    **kwargs: Any,
) -> LigerMoeCausalLMOutputWithPast:
    """Qwen3.5-MoE forward using a chunked packed GGUF LM-head loss."""

    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_router_logits = (
        output_router_logits
        if output_router_logits is not None
        else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    return_dict = (
        return_dict if return_dict is not None else self.config.use_return_dict
    )

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        mm_token_type_ids=mm_token_type_ids,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    if hidden_states is None:
        raise RuntimeError("Qwen model did not return hidden states")
    slice_indices = (
        slice(-logits_to_keep, None)
        if isinstance(logits_to_keep, int)
        else logits_to_keep
    )
    kept_hidden_states = hidden_states[:, slice_indices, :]

    shift_labels = kwargs.pop("shift_labels", None)
    logits = None
    loss = None
    token_accuracy = None
    predicted_tokens = None

    if skip_logits is None:
        skip_logits = self.training and (labels is not None or shift_labels is not None)
    if skip_logits and not (isinstance(logits_to_keep, int) and logits_to_keep == 0):
        raise RuntimeError(
            "Packed GGUF LM-head loss requires logits_to_keep=0 during training."
        )
    if skip_logits and labels is None and shift_labels is None:
        raise RuntimeError("Packed GGUF LM-head loss requires labels or shift_labels.")
    if (
        self.training
        and not skip_logits
        and (labels is not None or shift_labels is not None)
    ):
        raise RuntimeError(
            "Packed GGUF training with labels requires the no-full-logits fused loss."
        )

    if skip_logits:
        if not isinstance(self.lm_head, GgufLinear):
            raise TypeError(
                f"GGUF-aware Liger loss requires a GgufLinear LM head, got {type(self.lm_head).__name__}."
            )
        if self.lm_head.bias is not None:
            raise RuntimeError(
                "GGUF-aware Liger loss currently requires a bias-free LM head."
            )
        result = _packed_q8_liger_for_causal_lm_loss(
            hidden_states=kept_hidden_states,
            lm_head=self.lm_head,
            labels=labels,
            shift_labels=shift_labels,
            hidden_size=self.config.hidden_size,
            **kwargs,
        )
        loss, _, token_accuracy, predicted_tokens = unpack_cross_entropy_result(result)
    else:
        logits = self.lm_head(kept_hidden_states)
        if labels is not None or shift_labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                shift_labels=shift_labels,
                vocab_size=self.vocab_size,
                **kwargs,
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

    if not return_dict:
        output = (logits,) + outputs[1:]
        output = ((aux_loss,) + output) if aux_loss is not None else output
        output = ((loss,) + output) if loss is not None else output
        output = output + (token_accuracy,) if token_accuracy is not None else output
        output = (
            output + (predicted_tokens,) if predicted_tokens is not None else output
        )
        return output

    return LigerMoeCausalLMOutputWithPast(
        loss=cast(Any, loss),
        aux_loss=cast(Any, aux_loss),
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        token_accuracy=token_accuracy,
        predicted_tokens=predicted_tokens,
    )


def apply_gguf_liger_fused_linear_cross_entropy(
    model: torch.nn.Module,
) -> Qwen3_5MoeForCausalLM:
    """Patch one loaded text model with the GGUF-aware Liger loss forward."""

    get_base_model = getattr(model, "get_base_model", None)
    target = get_base_model() if callable(get_base_model) else model
    if not isinstance(target, Qwen3_5MoeForCausalLM):
        raise TypeError(
            "GGUF-aware Liger loss requires Qwen3_5MoeForCausalLM after unwrapping PEFT, "
            f"got {type(target).__name__}."
        )
    if not isinstance(target.lm_head, GgufLinear):
        raise TypeError(
            f"GGUF-aware Liger loss requires GgufLinear, got {type(target.lm_head).__name__}."
        )
    target.forward = MethodType(gguf_liger_lce_forward, target)
    return target
