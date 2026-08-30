import numpy as np
import pytest

from expert_distribution_prior import (
    EXPERTS,
    ExpertPrior,
    expert_prior_metadata,
    route_bank_for_routed_rows,
    sample_expert_profile,
    sample_expert_profiles,
)


@pytest.mark.parametrize(
    ("prior", "tokens"),
    [
        ("qwen-learned", 2048),
        ("deepseek-learned", 8192),
        ("deepseek-hash", 32768),
    ],
)
def test_profile_is_deterministic_and_preserves_route_mass(
    prior: str, tokens: int
) -> None:
    first = sample_expert_profile(prior, tokens, seed=7001)
    second = sample_expert_profile(prior, tokens, seed=7001)
    assert first == second
    assert len(first.rows_per_expert) == EXPERTS
    assert first.aggregate_rows == first.top_k * tokens
    assert all(0 <= rows <= tokens for rows in first.rows_per_expert)
    if prior == "deepseek-hash":
        assert all(rows > 0 for rows in first.rows_per_expert)


def test_prior_laws_are_explicit_and_not_mixed() -> None:
    learned = sample_expert_profile("deepseek-learned", 2048, seed=91)
    hashed = sample_expert_profile("deepseek-hash", 2048, seed=91)
    assert learned.prior is ExpertPrior.DeepSeekLearned
    assert hashed.prior is ExpertPrior.DeepSeekHash
    assert learned.rows_per_expert != hashed.rows_per_expert
    assert expert_prior_metadata("deepseek-learned")["router_kind"] == "learned"
    assert expert_prior_metadata("deepseek-hash")["router_kind"] == "hash"


def test_route_vector_metadata_uses_full_physical_group_sizes() -> None:
    routes = route_bank_for_routed_rows(
        "deepseek-learned", aggregate_rows=12288, count=3, seed=901
    )
    assert len(routes) == 3
    for route in routes:
        assert len(route.group_sizes) == EXPERTS
        assert route.expert_indices == tuple(
            index for index, rows in enumerate(route.group_sizes) if rows
        )
        compact_sizes = tuple(
            route.group_sizes[index] for index in route.expert_indices
        )
        assert route.expert_offsets == tuple(np.cumsum(compact_sizes).tolist())
        assert route.expert_offsets[-1] == sum(route.group_sizes)
        assert route.profile.aggregate_rows == sum(route.group_sizes)


def test_profile_bank_advances_seed_without_changing_the_selected_law() -> None:
    profiles = sample_expert_profiles("qwen-learned", 2048, 3, seed=100)
    assert [profile.seed for profile in profiles] == [100, 101, 102]
    assert {profile.prior for profile in profiles} == {ExpertPrior.QwenLearned}


def test_invalid_route_mass_is_rejected() -> None:
    with pytest.raises(ValueError, match="match the selected fitted law"):
        route_bank_for_routed_rows("qwen-learned", aggregate_rows=12287, count=1)
