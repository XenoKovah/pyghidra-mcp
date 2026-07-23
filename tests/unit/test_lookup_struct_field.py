"""Unit tests for GhidraTools._lookup_struct_field."""

from unittest.mock import Mock

import pytest

from pyghidra_mcp.tools import GhidraTools


def _make_component(name: str | None, offset: int):
    c = Mock()
    c.getFieldName.return_value = name
    c.getOffset.return_value = offset
    return c


def _make_struct(components: list, component_at_map: dict | None = None):
    """Build a minimal Mock Structure.

    ``component_at_map`` lets tests stub ``getComponentAt(offset)`` lookups.
    """
    component_at_map = component_at_map or {}
    struct = Mock()
    struct.getDefinedComponents.return_value = components
    struct.getComponentAt.side_effect = lambda off: component_at_map.get(off)
    return struct


def test_lookup_by_name_returns_matching_component():
    a = _make_component("a", 0)
    b = _make_component("b", 4)
    struct = _make_struct([a, b])

    result = GhidraTools._lookup_struct_field(struct, field_name="b")

    assert result is b


def test_lookup_by_name_skips_unnamed_components():
    unnamed = _make_component(None, 0)
    named = _make_component("real", 4)
    struct = _make_struct([unnamed, named])

    assert GhidraTools._lookup_struct_field(struct, field_name="real") is named


def test_lookup_by_name_raises_when_missing():
    a = _make_component("a", 0)
    struct = _make_struct([a])

    with pytest.raises(ValueError, match="not found"):
        GhidraTools._lookup_struct_field(struct, field_name="missing")


def test_lookup_by_offset_returns_component_at_offset():
    target = _make_component("y", 4)
    struct = _make_struct([], component_at_map={4: target})

    assert GhidraTools._lookup_struct_field(struct, field_offset=4) is target


def test_lookup_by_offset_raises_when_no_component_there():
    struct = _make_struct([], component_at_map={})  # everything returns None

    with pytest.raises(ValueError, match="No field at offset"):
        GhidraTools._lookup_struct_field(struct, field_offset=12)


def test_neither_param_provided_raises():
    struct = _make_struct([])

    with pytest.raises(ValueError, match="exactly one"):
        GhidraTools._lookup_struct_field(struct)


def test_both_params_provided_raises():
    struct = _make_struct([])

    with pytest.raises(ValueError, match="exactly one"):
        GhidraTools._lookup_struct_field(struct, field_name="x", field_offset=0)
