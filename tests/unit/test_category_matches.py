"""Unit tests for GhidraTools._category_matches."""

import pytest

from pyghidra_mcp.tools import GhidraTools


@pytest.mark.parametrize(
    "category_path, filter_cat, expected",
    [
        # Root selects everything
        ("/", "/", True),
        ("/MyStruct", "/", True),
        ("/MyLib/Sub", "/", True),
        # Exact category match
        ("/MyLib", "/MyLib", True),
        ("/MyLib/Sub", "/MyLib", True),
        ("/MyLib/Sub/Deep", "/MyLib", True),
        # Sibling that shares a prefix is NOT a child
        ("/MyLibrary", "/MyLib", False),
        ("/MyLibrary/Inner", "/MyLib", False),
        # Different roots
        ("/OtherLib/Point", "/MyLib", False),
        # Trailing-slash tolerance on the filter input
        ("/MyLib", "/MyLib/", True),
        ("/MyLib/Sub", "/MyLib/", True),
        # Case sensitivity
        ("/MyLib", "/mylib", False),
    ],
)
def test_category_matches(category_path: str, filter_cat: str, expected: bool) -> None:
    assert GhidraTools._category_matches(category_path, filter_cat) is expected
