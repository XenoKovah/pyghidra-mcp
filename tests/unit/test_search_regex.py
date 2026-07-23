"""Unit tests for regex search in search_symbols (with and without kinds filter)."""

from unittest.mock import Mock

from pyghidra_mcp.models import SymbolKind
from pyghidra_mcp.tools import GhidraTools


def _make_mock_symbol(name, address="0x1000", qualified_name=None):
    """Create a mock Ghidra Symbol."""
    sym = Mock()
    sym.name = name
    qualified = qualified_name or name
    sym.getName.side_effect = lambda include_namespace=False: (
        qualified if include_namespace else name
    )
    sym.getAddress.return_value = address
    sym.getSymbolType.return_value = "Function"
    sym.getParentNamespace.return_value = "Global"
    sym.getSource.return_value = "USER_DEFINED"
    sym.isExternal.return_value = False
    return sym


def _make_mock_function(name, address="0x1000", qualified_name=None):
    """Create a mock Ghidra Function with a Symbol."""
    sym = _make_mock_symbol(name, address, qualified_name=qualified_name)
    func = Mock()
    func.getSymbol.return_value = sym
    func.getEntryPoint.return_value = address
    func.isExternal.return_value = False
    func.thunk = False
    return func


def _make_tools(functions=None, symbols=None):
    """Create a GhidraTools instance with mocked internals."""
    program_info = Mock()
    rm = Mock()
    rm.getReferencesTo.return_value = []
    program_info.program.getReferenceManager.return_value = rm

    tools = GhidraTools.__new__(GhidraTools)
    tools.program_info = program_info
    tools.program = program_info.program
    tools.decompiler_pool = Mock()

    if functions is not None:
        tools.get_all_functions = Mock(return_value=functions)
        tools.find_functions = Mock(return_value=functions)

    if symbols is not None:
        tools.get_all_symbols = Mock(return_value=symbols)
        tools.find_symbols = Mock(return_value=symbols)

    return tools


class TestSearchFunctionsOnly:
    """Tests for search_symbols filtered to kinds=[SymbolKind.FUNCTIONS]."""

    def _funcs(self):
        return [
            _make_mock_function("main", "0x1000"),
            _make_mock_function("_main_init", "0x1100"),
            _make_mock_function("function_one", "0x2000"),
            _make_mock_function("function_two", "0x3000"),
            _make_mock_function("helper_func", "0x4000"),
        ]

    def test_plain_substring_still_works(self):
        """Plain substring query works as before (it's valid regex too)."""
        tools = _make_tools(functions=self._funcs())
        results = tools.search_symbols("function", kinds=[SymbolKind.FUNCTIONS])
        names = [s.name for s in results]
        assert "function_one" in names
        assert "function_two" in names

    def test_regex_exact_match(self):
        """^main$ matches only 'main', not '_main_init'."""
        tools = _make_tools(functions=self._funcs())
        results = tools.search_symbols("^main$", kinds=[SymbolKind.FUNCTIONS])
        names = [s.name for s in results]
        assert names == ["main"]

    def test_regex_dotstar_matches_all(self):
        """.* matches every function."""
        funcs = self._funcs()
        tools = _make_tools(functions=funcs)
        results = tools.search_symbols(".*", kinds=[SymbolKind.FUNCTIONS])
        assert len(results) == len(funcs)

    def test_regex_pattern_with_groups(self):
        """func.*(one|two) matches function_one and function_two."""
        tools = _make_tools(functions=self._funcs())
        results = tools.search_symbols("func.*(one|two)", kinds=[SymbolKind.FUNCTIONS])
        names = [s.name for s in results]
        assert "function_one" in names
        assert "function_two" in names
        assert "helper_func" not in names

    def test_regex_case_insensitive(self):
        """Search is case-insensitive."""
        tools = _make_tools(functions=self._funcs())
        results = tools.search_symbols("^MAIN$", kinds=[SymbolKind.FUNCTIONS])
        names = [s.name for s in results]
        assert names == ["main"]

    def test_regex_exact_match_uses_simple_name_when_symbol_is_namespaced(self):
        """Anchored matches should work against display names, not just qualified names."""
        funcs = [_make_mock_function("entry", "0x1000", qualified_name="Global::entry")]
        tools = _make_tools(functions=funcs)
        results = tools.search_symbols("^entry$", kinds=[SymbolKind.FUNCTIONS])
        names = [s.name for s in results]
        assert names == ["entry"]

    def test_invalid_regex_falls_back_to_substring(self):
        """Invalid regex like bare '*' falls back to substring match."""
        tools = _make_tools(functions=self._funcs())
        # '*' is invalid regex but also not a substring of any name -> empty
        results = tools.search_symbols("*", kinds=[SymbolKind.FUNCTIONS])
        assert results == []

    def test_invalid_regex_substring_match(self):
        """Invalid regex that is a valid substring still matches."""
        funcs = [_make_mock_function("test[func", "0x5000")]
        tools = _make_tools(functions=funcs)
        # '[func' is invalid regex, falls back to substring
        results = tools.search_symbols("[func", kinds=[SymbolKind.FUNCTIONS])
        names = [s.name for s in results]
        assert names == ["test[func"]

    def test_regex_metachar_uses_get_all_functions(self):
        """When query has regex metacharacters, uses get_all_functions."""
        tools = _make_tools(functions=self._funcs())
        tools.search_symbols("^main$", kinds=[SymbolKind.FUNCTIONS])
        tools.get_all_functions.assert_called_once_with(True)
        tools.find_functions.assert_not_called()

    def test_plain_query_uses_find_functions(self):
        """When query is plain text, uses find_functions (pre-filter)."""
        tools = _make_tools(functions=self._funcs())
        tools.search_symbols("main", kinds=[SymbolKind.FUNCTIONS])
        tools.find_functions.assert_called_once_with("main")
        tools.get_all_functions.assert_not_called()

    def test_empty_query_defaults_to_dot_star(self):
        """Empty query is coerced to ``.*`` and returns all functions."""
        funcs = self._funcs()
        tools = _make_tools(functions=funcs)
        results = tools.search_symbols("", kinds=[SymbolKind.FUNCTIONS])
        assert len(results) == len(funcs)

    def test_offset_and_limit(self):
        """Offset and limit pagination works with regex."""
        tools = _make_tools(functions=self._funcs())
        all_results = tools.search_symbols(".*", kinds=[SymbolKind.FUNCTIONS])
        page = tools.search_symbols(".*", kinds=[SymbolKind.FUNCTIONS], offset=1, limit=2)
        assert len(page) == 2
        assert page == all_results[1:3]


class TestSearchAllSymbols:
    """Tests for search_symbols with kinds=None (default)."""

    def _syms(self):
        return [
            _make_mock_symbol("main", "0x1000"),
            _make_mock_symbol("_main_init", "0x1100"),
            _make_mock_symbol("printf", "0x2000"),
            _make_mock_symbol("__libc_start_main", "0x3000"),
        ]

    def test_regex_exact_match(self):
        """^printf$ matches only 'printf'."""
        tools = _make_tools(symbols=self._syms())
        results = tools.search_symbols("^printf$")
        names = [s.name for s in results]
        assert names == ["printf"]

    def test_regex_dotstar_matches_all(self):
        """.* matches every symbol."""
        syms = self._syms()
        tools = _make_tools(symbols=syms)
        results = tools.search_symbols(".*")
        assert len(results) == len(syms)

    def test_regex_metachar_uses_get_all_symbols(self):
        """Regex query uses get_all_symbols."""
        tools = _make_tools(symbols=self._syms())
        tools.search_symbols("^main$")
        tools.get_all_symbols.assert_called_once_with(True)
        tools.find_symbols.assert_not_called()

    def test_plain_query_uses_find_symbols(self):
        """Plain query uses find_symbols."""
        tools = _make_tools(symbols=self._syms())
        tools.search_symbols("main")
        tools.find_symbols.assert_called_once_with("main")
        tools.get_all_symbols.assert_not_called()

    def test_invalid_regex_falls_back(self):
        """Invalid regex falls back to substring."""
        tools = _make_tools(symbols=self._syms())
        results = tools.search_symbols("*")
        assert results == []

    def test_default_is_all_symbols(self):
        """Default (no kinds filter) searches all symbols."""
        tools = _make_tools(symbols=self._syms())
        tools.search_symbols("main")
        tools.find_symbols.assert_called_once()
