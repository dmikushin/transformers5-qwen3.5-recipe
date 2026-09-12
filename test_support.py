"""Shared assertion helpers for the training tests.

The numeric gates follow one shape: a fused or BF16 candidate is compared against its eager or
FP32 reference, and the caller supplies the bounds that were measured for that site. The bounds
stay at the call sites so a tolerance can never drift through this module.
"""

import torch
import torch.nn.functional as F


def require_grad(tensor: torch.Tensor) -> torch.Tensor:
    """Return a tensor's gradient, or fail when autograd kept nothing."""

    if tensor.grad is None:
        raise AssertionError("expected a tensor gradient")
    return tensor.grad


def mixed_precision_metrics(
    candidate: torch.Tensor, reference: torch.Tensor
) -> tuple[float, float]:
    """Flat FP32 cosine similarity and relative RMSE of a candidate against a reference."""

    candidate_flat = candidate.detach().float().flatten()
    reference_flat = reference.detach().float().flatten()
    delta = candidate_flat - reference_flat
    cosine = float(F.cosine_similarity(candidate_flat, reference_flat, dim=0))
    relative_rmse = float(
        delta.square().mean().sqrt() / (reference_flat.square().mean().sqrt() + 1e-12)
    )
    return cosine, relative_rmse


def assert_close_mixed_precision(
    candidate: torch.Tensor,
    reference: torch.Tensor,
    *,
    minimum_cosine: float,
    maximum_relative_rmse: float,
) -> None:
    """Gate a floating-point candidate against a reference with caller-supplied bounds."""

    assert torch.isfinite(candidate).all()
    assert torch.isfinite(reference).all()
    cosine, relative_rmse = mixed_precision_metrics(candidate, reference)
    assert cosine >= minimum_cosine, (cosine, minimum_cosine)
    assert relative_rmse <= maximum_relative_rmse, (
        relative_rmse,
        maximum_relative_rmse,
    )


def assert_relative_rmse(
    actual: torch.Tensor,
    reference: torch.Tensor,
    maximum: float,
) -> None:
    """Gate an FP64 relative RMSE against a caller-supplied bound."""

    actual_float = actual.detach().double().reshape(-1)
    reference_float = reference.detach().double().reshape(-1)
    delta_rmse = (actual_float - reference_float).square().mean().sqrt()
    reference_rms = reference_float.square().mean().sqrt().clamp_min(1e-12)
    relative_rmse = float(delta_rmse / reference_rms)
    assert relative_rmse <= maximum, (relative_rmse, maximum)
