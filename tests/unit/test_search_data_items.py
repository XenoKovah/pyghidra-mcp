"""Unit tests for GhidraTools.search_data_items filter logic."""

from unittest.mock import Mock

from pyghidra_mcp.tools import GhidraTools


def _make_data(label: str | None, address: str, type_name: str, length: int):
    """Build a Mock Ghidra Data object."""
    data = Mock()
    data.getLabel.return_value = label
    addr = Mock()
    addr.__str__ = lambda self, _addr=address: _addr  # type: ignore[assignment]
    data.getAddress.return_value = addr
    dt = Mock()
    dt.getName.return_value = type_name
    data.getDataType.return_value = dt
    data.getLength.return_value = length
    return data, addr


def _make_tools(items: list[tuple]) -> GhidraTools:
    """Build a minimally-mocked GhidraTools whose iteration yields the given (data, refcount).

    ``items`` is a list of ``(data, refcount)`` tuples. The mocked ReferenceManager
    returns the matching refcount keyed by the data's address object.
    """
    listings: list = []
    refcount_by_addr: dict = {}
    for data_obj, refcount in items:
        # _make_data returned (data, addr); accept either form
        if isinstance(data_obj, tuple):
            data, addr = data_obj
        else:
            data = data_obj
            addr = data.getAddress.return_value
        listings.append(data)
        refcount_by_addr[id(addr)] = refcount

    listing = Mock()
    listing.getDefinedData.return_value = listings

    rm = Mock()
    rm.getReferenceCountTo.side_effect = lambda addr: refcount_by_addr.get(id(addr), 0)

    program_info = Mock()
    program_info.program.getListing.return_value = listing
    program_info.program.getReferenceManager.return_value = rm

    tools = GhidraTools.__new__(GhidraTools)
    tools.program_info = program_info
    tools.program = program_info.program
    tools.decompiler_pool = Mock()
    return tools


def test_no_filters_returns_all_in_iteration_order():
    items = [
        (_make_data("login_msg", "0x1000", "char[20]", 20), 3),
        (_make_data("g_count", "0x2000", "dword", 4), 7),
    ]
    tools = _make_tools(items)
    results = tools.search_data_items()
    assert [d.name for d in results] == ["login_msg", "g_count"]
    assert [d.refcount for d in results] == [3, 7]


def test_unlabeled_data_gets_dat_prefix():
    items = [
        (_make_data(None, "0x4000", "dword", 4), 0),
    ]
    tools = _make_tools(items)
    results = tools.search_data_items()
    assert results[0].name == "DAT_0x4000"


def test_query_filters_by_regex_on_label():
    items = [
        (_make_data("g_config", "0x1000", "Config", 16), 5),
        (_make_data("login_msg", "0x2000", "char[20]", 20), 1),
        (_make_data("g_state", "0x3000", "dword", 4), 12),
    ]
    tools = _make_tools(items)
    results = tools.search_data_items(query="^g_")
    assert {d.name for d in results} == {"g_config", "g_state"}


def test_min_refcount_filters_low_xref_data():
    items = [
        (_make_data("dead_const", "0x1000", "dword", 4), 0),
        (_make_data("popular_table", "0x2000", "byte[256]", 256), 42),
    ]
    tools = _make_tools(items)
    results = tools.search_data_items(min_refcount=10)
    assert [d.name for d in results] == ["popular_table"]


def test_max_refcount_filters_heavy_xref_data():
    items = [
        (_make_data("rare", "0x1000", "dword", 4), 1),
        (_make_data("popular", "0x2000", "byte[256]", 256), 42),
    ]
    tools = _make_tools(items)
    results = tools.search_data_items(max_refcount=10)
    assert [d.name for d in results] == ["rare"]


def test_offset_and_limit_paginate():
    items = [(_make_data(f"data_{i}", f"0x{i:04x}", "dword", 4), i) for i in range(10)]
    tools = _make_tools(items)
    page = tools.search_data_items(offset=3, limit=2)
    assert [d.name for d in page] == ["data_3", "data_4"]


def test_combined_filters_compose():
    items = [
        (_make_data("g_a", "0x1000", "dword", 4), 1),
        (_make_data("g_b", "0x2000", "dword", 4), 15),
        (_make_data("login_msg", "0x3000", "char[20]", 20), 30),
    ]
    tools = _make_tools(items)
    results = tools.search_data_items(query="^g_", min_refcount=10)
    assert [d.name for d in results] == ["g_b"]


def test_returns_data_type_and_length_per_item():
    items = [
        (_make_data("buf", "0x1000", "char[16]", 16), 2),
    ]
    tools = _make_tools(items)
    results = tools.search_data_items()
    assert results[0].type == "char[16]"
    assert results[0].length == 16
    assert results[0].address == "0x1000"
