"""Unit tests for tools._resolve_data_type_by_name_or_path."""

from unittest.mock import Mock

from pyghidra_mcp.tools import _resolve_data_type_by_name_or_path


def _make_dtm(by_path: dict[str, object] | None = None, all_types: list | None = None):
    """Build a Mock DataTypeManager whose lookups return pre-canned values."""
    by_path = by_path or {}
    all_types = all_types or []

    def get_data_type(path: str):
        return by_path.get(path)

    dtm = Mock()
    dtm.getDataType.side_effect = get_data_type
    dtm.getAllDataTypes.return_value = all_types
    return dtm


def _make_dt(name: str):
    """Return a Mock DataType whose getName() returns the given string."""
    dt = Mock()
    dt.getName.return_value = name
    return dt


def test_resolves_full_dtm_path_directly():
    target = _make_dt("Point")
    dtm = _make_dtm(by_path={"/MyLib/Point": target})
    assert _resolve_data_type_by_name_or_path(dtm, "/MyLib/Point") is target


def test_resolves_root_path_directly():
    target = _make_dt("Foo")
    dtm = _make_dtm(by_path={"/Foo": target})
    assert _resolve_data_type_by_name_or_path(dtm, "/Foo") is target


def test_bare_name_tries_root_path_first():
    target = _make_dt("Foo")
    dtm = _make_dtm(by_path={"/Foo": target})
    # Bare "Foo" should hit "/Foo" via the leading-slash fallback before scanning.
    assert _resolve_data_type_by_name_or_path(dtm, "Foo") is target


def test_no_leading_slash_path():
    target = _make_dt("Point")
    dtm = _make_dtm(by_path={"/MyLib/Point": target})
    # "MyLib/Point" should hit "/MyLib/Point" via the leading-slash fallback.
    assert _resolve_data_type_by_name_or_path(dtm, "MyLib/Point") is target


def test_bare_name_falls_back_to_full_scan():
    # Type lives at /SomeDeep/Path/Foo — neither "Foo" nor "/Foo" is a direct
    # path lookup hit, so we have to scan to find it.
    target = _make_dt("Foo")
    dtm = _make_dtm(all_types=[_make_dt("OtherType"), target])
    assert _resolve_data_type_by_name_or_path(dtm, "Foo") is target


def test_returns_none_when_missing():
    dtm = _make_dtm(all_types=[_make_dt("Other")])
    assert _resolve_data_type_by_name_or_path(dtm, "Missing") is None
