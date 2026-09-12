from typing import Any, cast

import pytest
import torch

from module_patching import (
    PATCH_MARKER,
    ModulePatchSpec,
    patch_module_forwards,
    require_complete_inventory,
)


def _identity_forward(self: torch.nn.Module, hidden_states: torch.Tensor):
    del self
    return hidden_states


def _toy() -> torch.nn.Module:
    model = torch.nn.Module()
    model.first = torch.nn.LayerNorm(4)
    model.second = torch.nn.LayerNorm(4)
    return model


def test_patch_module_forwards_counts_handled_skipped_and_declared_keys() -> None:
    model = _toy()
    spec = ModulePatchSpec(
        module_type=torch.nn.LayerNorm,
        forward=_identity_forward,
        handled_key="handled",
        accept=lambda name, module: name == "first",
        skip_key="skipped",
    )

    report = patch_module_forwards(model, (spec,), declared_keys=("reserved",))
    assert report["handled"] == 1
    assert report["skipped"] == 1
    assert report["reserved"] == 0
    assert report["patched"] == 1
    assert report["already_patched"] == 0
    assert report["patched_names"] == ["first"]
    assert report["handled_by_key"] == {
        "handled": ["first"],
        "skipped": [],
        "reserved": [],
    }
    assert getattr(model.first, PATCH_MARKER)
    assert not getattr(model.second, PATCH_MARKER, False)

    again = patch_module_forwards(model, (spec,), declared_keys=("reserved",))
    assert again["handled"] == 1
    assert again["skipped"] == 1
    assert again["patched"] == 0
    assert again["already_patched"] == 1
    assert again["patched_names"] == []
    assert again["handled_by_key"]["handled"] == ["first"]


def test_patch_module_forwards_rejects_a_skip_without_a_key() -> None:
    model = _toy()
    spec = ModulePatchSpec(
        module_type=torch.nn.LayerNorm,
        forward=_identity_forward,
        handled_key="handled",
        accept=lambda name, module: False,
    )
    with pytest.raises(RuntimeError, match="without a skip key"):
        patch_module_forwards(model, (spec,))


def test_patch_module_forwards_prepares_every_match_including_patched() -> None:
    model = _toy()
    prepared: list[str] = []
    spec = ModulePatchSpec(
        module_type=torch.nn.LayerNorm,
        forward=_identity_forward,
        handled_key="handled",
        prepare=lambda name, module: prepared.append(name),
    )

    patch_module_forwards(model, (spec,))
    assert prepared == ["first", "second"]
    assert cast(Any, model.first).weight.requires_grad is False

    patch_module_forwards(model, (spec,))
    assert prepared == ["first", "second", "first", "second"]


def _family_spec(index: int) -> ModulePatchSpec[torch.nn.LayerNorm]:
    def _matches(name: str, module: torch.nn.LayerNorm) -> bool:
        del name
        return cast(Any, module).family == f"family_{index}"

    return ModulePatchSpec(
        module_type=torch.nn.LayerNorm,
        forward=_identity_forward,
        handled_key=f"family_{index}",
        matches=_matches,
        freeze_weight=False,
    )


def test_patch_module_forwards_matches_routes_same_type_specs() -> None:
    model = _toy()
    first = cast(Any, model.first)
    second = cast(Any, model.second)
    first.family = "family_0"
    second.family = "family_1"

    report = patch_module_forwards(model, (_family_spec(0), _family_spec(1)))
    assert report["family_0"] == 1
    assert report["family_1"] == 1
    assert report["patched"] == 2
    assert report["handled_by_key"]["family_0"] == ["first"]
    assert report["handled_by_key"]["family_1"] == ["second"]
    assert getattr(model.first, PATCH_MARKER)
    assert getattr(model.second, PATCH_MARKER)


def test_patch_module_forwards_mark_only_spec_keeps_the_forward() -> None:
    model = _toy()
    original = torch.nn.LayerNorm.forward
    spec = ModulePatchSpec(
        module_type=torch.nn.LayerNorm,
        forward=None,
        handled_key="marked",
        freeze_weight=False,
    )

    report = patch_module_forwards(model, (spec,))
    assert report["marked"] == 2
    assert report["patched"] == 2
    assert getattr(model.first, PATCH_MARKER)
    assert getattr(model.second, PATCH_MARKER)
    assert cast(Any, model.first).forward.__func__ is original

    again = patch_module_forwards(model, (spec,))
    assert again["marked"] == 2
    assert again["patched"] == 0
    assert again["already_patched"] == 2


def test_require_complete_inventory_reports_every_mismatch() -> None:
    with pytest.raises(RuntimeError, match="incomplete Toy configuration"):
        require_complete_inventory({"a": 1}, {"a": 2, "b": 0}, subject="Toy")
    require_complete_inventory({"a": 2, "b": 0}, {"a": 2, "b": 0}, subject="Toy")
    with pytest.raises(RuntimeError, match="incomplete Toy configuration"):
        require_complete_inventory(
            {"implementation": "slow"},
            {"implementation": "fast"},
            subject="Toy",
        )
    require_complete_inventory(
        {"implementation": "fast"}, {"implementation": "fast"}, subject="Toy"
    )
