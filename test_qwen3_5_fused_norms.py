import types
from typing import Any

import pytest
import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeRMSNorm,
    Qwen3_5MoeRMSNormGated,
)

from qwen3_5_fused_norms import (
    EXPECTED_GATED_RMSNORMS,
    EXPECTED_RMSNORMS,
    _fla_gated_rmsnorm_forward,
    _liger_rmsnorm_forward,
    configure_qwen35_fused_norms,
    require_complete_qwen35_fused_norms,
)

_MIN_COSINE = 0.999
_MAX_RELATIVE_RMSE = 0.005


def _require_grad(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.grad is None:
        raise AssertionError("expected a tensor gradient")
    return tensor.grad


def _assert_close_mixed_precision(
    candidate: torch.Tensor, reference: torch.Tensor
) -> None:
    reference_flat = reference.detach().float().flatten()
    candidate_flat = candidate.detach().float().flatten()
    assert torch.isfinite(reference_flat).all()
    assert torch.isfinite(candidate_flat).all()
    cosine = torch.nn.functional.cosine_similarity(
        reference_flat, candidate_flat, dim=0
    )
    relative_rmse = (candidate_flat - reference_flat).square().mean().sqrt() / (
        reference_flat.square().mean().sqrt() + 1e-12
    )
    assert float(cosine) >= _MIN_COSINE
    assert float(relative_rmse) <= _MAX_RELATIVE_RMSE


def _randomize(weight: torch.Tensor, generator: torch.Generator) -> None:
    with torch.no_grad():
        weight.normal_(mean=0.0, std=0.2, generator=generator)


def _patched_forward(module: torch.nn.Module) -> Any:
    forward = module.__dict__.get("forward")
    if not isinstance(forward, types.MethodType):
        raise TypeError("expected an instance-local patched forward")
    return forward.__func__


class _PlainNormToy(torch.nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = Qwen3_5MoeRMSNorm(width, eps=1e-6)
        self.other = torch.nn.LayerNorm(width)


class _GatedNormHolder(torch.nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.norm = Qwen3_5MoeRMSNormGated(head_dim, eps=1e-6)


class _GatedNormToy(torch.nn.Module):
    def __init__(self, head_dim: int) -> None:
        super().__init__()
        self.linear_attn = _GatedNormHolder(head_dim)


class _InventoryToy(torch.nn.Module):
    def __init__(self, rmsnorms: int, gated: int, width: int = 8) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Qwen3_5MoeRMSNorm(width, eps=1e-6) for _ in range(rmsnorms)]
        )
        self.gated = torch.nn.ModuleList(
            [Qwen3_5MoeRMSNormGated(width, eps=1e-6) for _ in range(gated)]
        )
        self.untouched = torch.nn.LayerNorm(width)


@pytest.mark.parametrize("width", [128, 512, 2048])
def test_liger_rmsnorm_matches_eager_reference(width: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(31415)
    reference = _PlainNormToy(width).cuda().to(torch.bfloat16)
    candidate = _PlainNormToy(width).cuda().to(torch.bfloat16)
    _randomize(reference.norm.weight, generator)
    candidate.load_state_dict(reference.state_dict())
    reference.requires_grad_(False)
    candidate.requires_grad_(False)

    report = configure_qwen35_fused_norms(candidate)
    assert report["rmsnorms"] == 1
    assert report["gated_rmsnorms"] == 0
    assert _patched_forward(candidate.norm) is _liger_rmsnorm_forward

    reference_input = torch.randn(
        2, 16, width, generator=generator, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    grad_output = torch.randn(
        2, 16, width, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    reference_output = reference.norm(reference_input)
    candidate_output = candidate.norm(candidate_input)
    reference_output.backward(grad_output)
    candidate_output.backward(grad_output)

    _assert_close_mixed_precision(candidate_output, reference_output)
    _assert_close_mixed_precision(
        _require_grad(candidate_input), _require_grad(reference_input)
    )
    assert candidate.norm.weight.grad is None
    assert reference.norm.weight.grad is None


def test_fla_gated_rmsnorm_matches_eager_reference() -> None:
    head_dim = 128
    generator = torch.Generator(device="cuda").manual_seed(2718)
    reference = _GatedNormToy(head_dim).cuda().to(torch.bfloat16)
    candidate = _GatedNormToy(head_dim).cuda().to(torch.bfloat16)
    _randomize(reference.linear_attn.norm.weight, generator)
    candidate.load_state_dict(reference.state_dict())
    reference.requires_grad_(False)
    candidate.requires_grad_(False)

    report = configure_qwen35_fused_norms(candidate)
    assert report["gated_rmsnorms"] == 1
    assert report["rmsnorms"] == 0
    norm = candidate.linear_attn.norm
    assert _patched_forward(norm) is _fla_gated_rmsnorm_forward

    reference_input = torch.randn(
        512, head_dim, generator=generator, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    candidate_input = reference_input.detach().clone().requires_grad_(True)
    reference_gate = torch.randn(
        512, head_dim, generator=generator, device="cuda", dtype=torch.bfloat16
    ).requires_grad_(True)
    candidate_gate = reference_gate.detach().clone().requires_grad_(True)
    grad_output = torch.randn(
        512, head_dim, generator=generator, device="cuda", dtype=torch.bfloat16
    )

    reference_output = reference.linear_attn.norm(reference_input, reference_gate)
    candidate_output = candidate.linear_attn.norm(candidate_input, candidate_gate)
    reference_output.backward(grad_output)
    candidate_output.backward(grad_output)

    _assert_close_mixed_precision(candidate_output, reference_output)
    _assert_close_mixed_precision(
        _require_grad(candidate_input), _require_grad(reference_input)
    )
    _assert_close_mixed_precision(
        _require_grad(candidate_gate), _require_grad(reference_gate)
    )
    assert norm.weight.grad is None
    assert reference.linear_attn.norm.weight.grad is None


def test_patched_full_inventory_is_required_and_idempotent() -> None:
    model = _InventoryToy(EXPECTED_RMSNORMS, EXPECTED_GATED_RMSNORMS).cuda()
    report = configure_qwen35_fused_norms(model)
    require_complete_qwen35_fused_norms(report)
    assert report["patched"] == EXPECTED_RMSNORMS + EXPECTED_GATED_RMSNORMS
    assert report["already_patched"] == 0
    assert not getattr(model.untouched, "_fused_norm", False)

    again = configure_qwen35_fused_norms(model)
    require_complete_qwen35_fused_norms(again)
    assert again["patched"] == 0
    assert again["already_patched"] == EXPECTED_RMSNORMS + EXPECTED_GATED_RMSNORMS


def test_incomplete_inventory_is_rejected() -> None:
    model = _InventoryToy(EXPECTED_RMSNORMS - 1, EXPECTED_GATED_RMSNORMS).cuda()
    report = configure_qwen35_fused_norms(model)
    with pytest.raises(RuntimeError, match="incomplete Qwen3.5 fused norm"):
        require_complete_qwen35_fused_norms(report)


def test_unsupported_activation_is_rejected() -> None:
    model = _GatedNormToy(128).cuda().to(torch.bfloat16)
    model.linear_attn.norm.activation = "gelu"
    with pytest.raises(RuntimeError, match="unsupported activation"):
        configure_qwen35_fused_norms(model)


def test_cpu_norm_is_rejected() -> None:
    model = _PlainNormToy(64)
    with pytest.raises(RuntimeError, match="requires a CUDA/ROCm weight"):
        configure_qwen35_fused_norms(model)


def test_gated_norm_requires_its_gate() -> None:
    model = _GatedNormToy(128).cuda().to(torch.bfloat16)
    configure_qwen35_fused_norms(model)
    hidden = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="requires its gate tensor"):
        model.linear_attn.norm(hidden)
