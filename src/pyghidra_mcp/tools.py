"""
Comprehensive tool implementations for pyghidra-mcp.
"""

import functools
import logging
import re
import typing
from contextlib import contextmanager

from ghidrecomp.callgraph import gen_callgraph
from jpype import JByte

from pyghidra_mcp.models import (
    CallGraphDirection,
    CallGraphDisplayType,
    CodeSearchResult,
    CrossReferenceInfo,
    DataItemInfo,
    DataTypeInfo,
    DecompiledFunction,
    ExportInfo,
    FunctionInfo,
    FunctionRef,
    GenCallgraphResponse,
    ImportInfo,
    ReadBytesResponse,
    SearchCodeResponse,
    SearchMode,
    StringInfo,
    StringSearchResult,
    SymbolInfo,
)

_REGEX_META = re.compile(r"[\\^$.|?*+(){}\[\]]")

if typing.TYPE_CHECKING:
    from ghidra.app.decompiler import DecompileResults
    from ghidra.program.model.listing import Function
    from ghidra.program.model.symbol import Symbol

    from .context import ProgramInfo

logger = logging.getLogger(__name__)


@contextmanager
def ghidra_transaction(program, description: str):
    tx_id = program.startTransaction(description)
    committed = False
    try:
        yield
        committed = True
    finally:
        program.endTransaction(tx_id, committed)


def _compute_struct_init_size(resolved: list[tuple[dict, typing.Any]]) -> int:
    """Largest `offset + dt.getLength()` across fields that carry an offset.

    Used to size a StructureDataType big enough to fit all explicitly-placed
    fields; fields without an offset are ignored here (they'll be appended
    later by `_layout_struct_fields`).
    """
    init_size = 0
    for f, dt in resolved:
        off = f.get("offset")
        if off is not None:
            end = off + dt.getLength()
            if end > init_size:
                init_size = end
    return init_size


def _layout_struct_fields(
    struct, resolved: list[tuple[dict, typing.Any]], has_offsets: bool
) -> None:
    """Add each resolved field to a StructureDataType.

    When any field specifies an offset (``has_offsets=True``), fields with an
    offset are placed explicitly via replaceAtOffset; others are appended.
    When no field has an offset, all fields are appended in order.
    """
    for f, dt in resolved:
        off = f.get("offset")
        if has_offsets and off is not None:
            struct.replaceAtOffset(off, dt, dt.getLength(), f["name"], "")
        else:
            struct.add(dt, dt.getLength(), f["name"], "")


def _resolve_data_type_by_name_or_path(dtm, name_or_path: str):
    """Look up a Ghidra DataType by full DTM path, falling back to bare-name search.

    Resolution order:
      1. ``dtm.getDataType(name_or_path)`` — fast path lookup; works for
         ``"/Point"`` and ``"/MyLib/Point"``.
      2. If ``name_or_path`` doesn't start with ``/``, retry with one prepended
         (handles bare names at root and ``"MyLib/Point"`` style paths).
      3. Linear scan of every category for a name match (last-resort).

    Returns ``None`` if nothing matches.
    """
    dt = dtm.getDataType(name_or_path)
    if dt is not None:
        return dt
    if not name_or_path.startswith("/"):
        dt = dtm.getDataType("/" + name_or_path)
        if dt is not None:
            return dt
    for candidate in dtm.getAllDataTypes():
        if str(candidate.getName()) == name_or_path:
            return candidate
    return None


def _resolve_script_path(script: str):
    """Locate a script file through the standard search chain.

    Search order:
      1. ``script`` as-given (absolute path or CWD-relative).
      2. ``~/ghidra_scripts/<script>`` and ``~/ghidra_scripts/<basename>``.
      3. ``./ghidra_scripts/<script>`` and ``./ghidra_scripts/<basename>``.
      4. If ``script`` has no extension, each of the above is also tried with
         ``.py`` appended (``.py`` first, then bare).

    Returns the first existing file as a resolved ``Path``, or ``None``.
    """
    from pathlib import Path

    home_scripts = Path.home() / "ghidra_scripts"
    cwd_scripts = Path.cwd() / "ghidra_scripts"
    basename = Path(script).name

    bases = [
        Path(script),
        home_scripts / script,
        home_scripts / basename,
        cwd_scripts / script,
        cwd_scripts / basename,
    ]

    has_extension = "." in basename
    suffixes: list[str] = [""] if has_extension else [".py", ""]

    for base in bases:
        for suffix in suffixes:
            candidate = Path(str(base) + suffix) if suffix else base
            if candidate.is_file():
                return candidate.resolve()

    return None


def handle_exceptions(func):
    """Decorator to handle exceptions in tool methods"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e!s}")
            raise

    return wrapper


class GhidraTools:
    """Comprehensive tool handler for Ghidra MCP tools"""

    def __init__(self, program_info: "ProgramInfo"):
        """Initialize with a Ghidra ProgramInfo object"""
        self.program_info = program_info
        self.program = program_info.program
        self.decompiler_pool = program_info.decompiler_pool

    def _get_filename(self, func: "Function"):
        max_path_len = 50
        return f"{func.getSymbol().getName(True)[:max_path_len]}-{func.entryPoint}"

    def _resolve_function_variable(
        self,
        function_name_or_address: str,
        variable_name: str,
    ) -> tuple["Function", str, typing.Any]:
        func = self.find_function(function_name_or_address)
        function_name = str(func.getName())

        matches: list[tuple[str, typing.Any]] = []
        for param in func.getParameters():
            if str(param.getName()) == variable_name:
                matches.append(("parameter", param))
        for local in func.getLocalVariables():
            if str(local.getName()) == variable_name:
                matches.append(("local", local))

        if not matches:
            raise ValueError(f"Variable '{variable_name}' not found in function '{function_name}'.")
        if len(matches) > 1:
            kinds = ", ".join(kind for kind, _ in matches)
            raise ValueError(
                f"Ambiguous variable '{variable_name}' in function '{function_name}' ({kinds})."
            )

        variable_kind, variable = matches[0]
        return func, variable_kind, variable

    def _parse_data_type(self, type_name: str):
        from ghidra.util.data import DataTypeParser  # type: ignore
        from ghidra.util.data.DataTypeParser import AllowedDataTypes  # type: ignore

        dtm = self.program.getDataTypeManager()
        parser = DataTypeParser(dtm, dtm, typing.cast(typing.Any, None), AllowedDataTypes.DYNAMIC)
        return parser.parse(type_name)

    def _lookup_functions(
        self,
        name_or_address: str,
        *,
        exact: bool = True,
        partial: bool = False,
        include_externals: bool = True,
    ) -> list["Function"]:
        """
        Resolve functions by name or address.
        Returns a flat list of unique Function objects.
        Search modes (exact, partial) are optional and only applied if enabled.
        """
        af = self.program.getAddressFactory()
        fm = self.program.getFunctionManager()

        # Try interpreting as an address first
        try:
            addr = af.getAddress(name_or_address)
            if addr:
                func = fm.getFunctionAt(addr)
                if func:
                    return [func]
        except Exception:
            pass  # Not an address, fall back to name search

        name_lc = name_or_address.lower()
        functions = self.get_all_functions(include_externals=include_externals)
        seen: set = set()
        matches: list[Function] = []

        if exact:
            for f in functions:
                key = f.getEntryPoint()
                if key not in seen and name_lc == f.getSymbol().getName(True).lower():
                    seen.add(key)
                    matches.append(f)

        if partial:
            for f in functions:
                key = f.getEntryPoint()
                if key not in seen and name_lc in f.getSymbol().getName(True).lower():
                    seen.add(key)
                    matches.append(f)

        return matches

    @handle_exceptions
    def find_function(
        self,
        name_or_address: str,
        include_externals: bool = True,
    ) -> "Function":
        """
        Resolve a single function by name or address (exact match only).
        Raises if ambiguous or not found.
        """
        matches = self._lookup_functions(
            name_or_address, exact=True, partial=False, include_externals=include_externals
        )

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            suggestions = [
                f"{f.getSymbol().getName(True)}({f.getSignature()}) @ {f.getEntryPoint()}"
                for f in matches
            ]
            raise ValueError(
                f"Ambiguous match for '{name_or_address}'. Did you mean one of these: "
                + ", ".join(suggestions)
            )
        else:
            raise ValueError(f"Function or symbol '{name_or_address}' not found.")

    @handle_exceptions
    def find_functions(
        self,
        name_or_address: str,
        include_externals: bool = True,
    ) -> list["Function"]:
        """
        Return all functions that match name_or_address (exact or partial).
        Never raises; returns empty list if none.
        """
        return self._lookup_functions(
            name_or_address, exact=True, partial=True, include_externals=include_externals
        )

    def _lookup_symbols(
        self,
        name_or_address: str,
        *,
        exact: bool = True,
        partial: bool = False,
        dynamic: bool = False,
    ) -> list["Symbol"]:
        """
        Resolve symbols by name or address.
        Returns a single flat list of unique Symbol objects.
        Search modes (exact, partial, dynamic) are optional and only applied if enabled.
        """
        st = self.program.getSymbolTable()
        af = self.program.getAddressFactory()

        # Try interpreting as an address first
        try:
            addr = af.getAddress(name_or_address)
            if addr:
                addr_symbols = st.getSymbols(addr)
                if addr_symbols:
                    return list(addr_symbols)
        except Exception:
            pass  # Not an address, fall back to name search

        name_lc = name_or_address.lower()
        matches: set[Symbol] = set()

        # Base symbol set (externals only once)
        base_symbols = self.get_all_symbols(include_externals=True)

        # Exact match
        if exact:
            matches.update(s for s in base_symbols if name_lc == s.getName(True).lower())

        # Partial match
        if partial:
            matches.update(s for s in base_symbols if name_lc in s.getName(True).lower())

        # Dynamic match (requires second scan)
        if dynamic:
            dyn_symbols = self.get_all_symbols(include_externals=True, include_dynamic=True)
            matches.update(s for s in dyn_symbols if name_lc in s.getName(True).lower())

        return list(matches)

    @handle_exceptions
    def find_symbols(self, name_or_address: str) -> list["Symbol"]:
        """
        Return all symbols that match name_or_address (exact or partial).
        Never raises; returns empty list if none.
        """
        return self._lookup_symbols(name_or_address, exact=True, partial=True)

    @handle_exceptions
    def find_symbol(self, name_or_address: str) -> "Symbol":
        """
        Resolve a single symbol by name or address (exact match only).
        Raises if ambiguous or not found.
        """
        matches = self._lookup_symbols(name_or_address, exact=True, partial=False)

        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            suggestions = [f"{s.getName(True)} @ {s.getAddress()}" for s in matches]
            raise ValueError(
                f"Ambiguous match for '{name_or_address}'. Did you mean one of these: "
                + ", ".join(suggestions)
            )
        else:
            raise ValueError(f"Symbol '{name_or_address}' not found.")

    @handle_exceptions
    def decompile_function_by_name_or_addr(
        self, name_or_address: str, timeout: int = 0
    ) -> DecompiledFunction:
        """Finds and decompiles a function in a specified binary and returns its pseudo-C code."""

        func = self.find_function(name_or_address)
        return self.decompile_function(func, timeout=timeout)

    def decompile_function(self, func: "Function", timeout: int = 0) -> DecompiledFunction:
        """Decompiles a function in a specified binary and returns its pseudo-C code."""
        import os

        from ghidra.util.task import ConsoleTaskMonitor

        # Per-function decompile bound (seconds). When no explicit timeout is given
        # (timeout <= 0), fall back to PYGHIDRA_MCP_DECOMP_TIMEOUT (default 120);
        # set it to 0 for stock/infinite behavior. This keeps a pathological /
        # CFG-corrupt function — e.g. an ARC image whose decompiler chases
        # references into unmapped memory and never returns — from wedging the
        # decompiler pool forever (background indexing calls this with no timeout).
        if timeout <= 0:
            timeout = int(os.environ.get("PYGHIDRA_MCP_DECOMP_TIMEOUT", "120") or "120")

        monitor = ConsoleTaskMonitor()
        with self.decompiler_pool.acquire() as decompiler:
            result: DecompileResults = decompiler.decompileFunction(func, timeout, monitor)
        if "" == result.getErrorMessage():
            code = result.decompiledFunction.getC()
            sig = result.decompiledFunction.getSignature()
        else:
            code = result.getErrorMessage()
            sig = None
        return DecompiledFunction(name=self._get_filename(func), code=code, signature=sig)

    @handle_exceptions
    def get_all_functions(self, include_externals=False) -> list["Function"]:
        """
        Gets all functions within a binary.
        Returns a python list that doesn't need to be re-intialized
        """

        funcs = set()
        fm = self.program.getFunctionManager()
        functions = fm.getFunctions(True)
        for func in functions:
            func: Function
            if not include_externals and func.isExternal():
                continue
            if not include_externals and func.thunk:
                continue
            funcs.add(func)
        return list(funcs)

    @handle_exceptions
    def get_all_symbols(
        self, include_externals: bool = False, include_dynamic=False
    ) -> list["Symbol"]:
        """
        Gets all symbols within a binary.
        Returns a python list that doesn't need to be re-initialized.
        """

        symbols = set()
        from ghidra.program.model.symbol import SymbolTable

        st: SymbolTable = self.program.getSymbolTable()
        all_symbols = st.getAllSymbols(include_dynamic)

        for sym in all_symbols:
            sym: Symbol
            if not include_externals and sym.isExternal():
                continue
            symbols.add(sym)

        return list(symbols)

    @handle_exceptions
    def get_all_strings(self) -> list[StringInfo]:
        """Gets all defined strings for a binary"""
        try:
            from ghidra.program.util import DefinedStringIterator  # type: ignore

            data_iterator = DefinedStringIterator.forProgram(self.program)
        except ImportError:
            # Support Ghidra 11.3.2
            from ghidra.program.util import DefinedDataIterator

            data_iterator = DefinedDataIterator.definedStrings(self.program)

        strings = []
        for data in data_iterator:
            try:
                string_value = data.getValue()
                strings.append(StringInfo(value=str(string_value), address=str(data.getAddress())))
            except Exception as e:
                logger.debug(f"Could not get string value from data at {data.getAddress()}: {e}")

        return strings

    @staticmethod
    def _matches_query(query: str, symbol_name: str) -> bool:
        """Check if a symbol name matches a query (regex with substring fallback)."""
        try:
            return bool(re.search(query, symbol_name, re.IGNORECASE))
        except re.error:
            return query.lower() in symbol_name.lower()

    @classmethod
    def _symbol_matches_query(cls, query: str, symbol) -> bool:
        """Match against both simple and namespace-qualified symbol names."""
        names = {str(symbol.getName())}
        try:
            names.add(str(symbol.getName(True)))
        except TypeError:
            pass
        return any(cls._matches_query(query, name) for name in names)

    def _symbol_to_info(self, symbol, rm) -> SymbolInfo:
        """Convert a Ghidra Symbol to a SymbolInfo model."""
        ref_count = len(list(rm.getReferencesTo(symbol.getAddress())))
        return SymbolInfo(
            name=symbol.getName(),
            address=str(symbol.getAddress()),
            type=str(symbol.getSymbolType()),
            namespace=str(symbol.getParentNamespace()),
            source=str(symbol.getSource()),
            refcount=ref_count,
            external=symbol.isExternal(),
        )

    def _fetch_symbols_for_kinds(self, kind_set, query, is_regex):
        """Return the initial iterable of symbols based on the kind filter.

        Single-kind ``{"functions"}`` takes the FunctionManager fast path;
        everything else scans the symbol table.
        """
        if kind_set == {"functions"}:
            sources = self.get_all_functions(True) if is_regex else self.find_functions(query)
            return [func.getSymbol() for func in sources]
        return self.get_all_symbols(True) if is_regex else self.find_symbols(query)

    @staticmethod
    def _symbol_matches_kind_set(sym, kind_set, global_ns) -> bool:
        """Return True when sym satisfies any of the SymbolKind values in kind_set."""
        from ghidra.program.model.symbol import SymbolType

        st = sym.getSymbolType()
        if "functions" in kind_set and st == SymbolType.FUNCTION:
            return True
        if "labels" in kind_set and st == SymbolType.LABEL:
            return True
        if "globals" in kind_set and st != SymbolType.FUNCTION:
            if sym.getParentNamespace() == global_ns:
                return True
        return False

    @handle_exceptions
    def search_symbols(
        self,
        query: str = ".*",
        kinds: list | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[SymbolInfo]:
        """Search symbols within a binary by name (regex supported).

        ``kinds`` is an optional list of SymbolKind values (or their string
        equivalents). ``None`` or omitted means include every symbol.
        Supported values: ``"functions"``, ``"globals"``, ``"labels"``.
        """
        if not query:
            query = ".*"

        kind_set = None
        if kinds:
            kind_set = {k.value if hasattr(k, "value") else str(k) for k in kinds}

        rm = self.program.getReferenceManager()
        is_regex = bool(_REGEX_META.search(query))
        symbols: typing.Iterable = self._fetch_symbols_for_kinds(kind_set, query, is_regex)

        # Only need per-symbol kind classification when the filter spans more
        # than the fast path already pre-filtered for us.
        if kind_set is not None and kind_set != {"functions"}:
            global_ns = self.program.getGlobalNamespace()
            symbols = [s for s in symbols if self._symbol_matches_kind_set(s, kind_set, global_ns)]

        results = [
            self._symbol_to_info(sym, rm)
            for sym in symbols
            if self._symbol_matches_query(query, sym)
        ]
        return results[offset : limit + offset]

    @staticmethod
    def _function_to_info(func) -> FunctionInfo:
        """Project a Ghidra Function into the FunctionInfo response model."""
        from ghidra.program.model.symbol import SourceType

        symbol = func.getSymbol()
        return FunctionInfo(
            name=str(func.getName()),
            address=str(func.getEntryPoint()),
            refcount=int(symbol.getReferenceCount()),
            is_thunk=bool(func.isThunk()),
            is_external=bool(func.isExternal()),
            is_user_defined=symbol.getSource() != SourceType.DEFAULT,
        )

    @handle_exceptions
    def search_functions(
        self,
        query: str = ".*",
        min_refcount: int | None = None,
        max_refcount: int | None = None,
        user_defined_only: bool = False,
        include_thunks: bool = True,
        include_externals: bool = True,
        offset: int = 0,
        limit: int = 100,
    ) -> list[FunctionInfo]:
        """Search functions with optional filters.

        ``query`` is a case-insensitive regex matched against the function
        name (defaults to ``".*"`` so the call lists every function unless
        narrowed). ``min_refcount`` / ``max_refcount`` bound xref counts
        (inclusive). ``user_defined_only`` excludes Ghidra-autogenerated
        names like ``FUN_*``. ``include_thunks`` and ``include_externals``
        gate those function categories. Results are returned in address
        order.
        """
        if not query:
            query = ".*"

        fm = self.program.getFunctionManager()
        results: list[FunctionInfo] = []
        for func in fm.getFunctions(True):
            if not include_thunks and func.isThunk():
                continue
            if not include_externals and func.isExternal():
                continue

            info = self._function_to_info(func)

            if user_defined_only and not info.is_user_defined:
                continue
            if min_refcount is not None and info.refcount < min_refcount:
                continue
            if max_refcount is not None and info.refcount > max_refcount:
                continue
            if not self._matches_query(query, info.name):
                continue

            results.append(info)

        return results[offset : limit + offset]

    @handle_exceptions
    def list_exports(
        self, query: str | None = None, offset: int = 0, limit: int = 25
    ) -> list[ExportInfo]:
        """Lists all exported functions and symbols from a specified binary."""
        exports = []
        symbols = self.program.getSymbolTable().getAllSymbols(True)
        for symbol in symbols:
            if symbol.isExternalEntryPoint():
                if query and not re.search(query, symbol.getName(), re.IGNORECASE):
                    continue
                exports.append(ExportInfo(name=symbol.getName(), address=str(symbol.getAddress())))
        return exports[offset : limit + offset]

    @handle_exceptions
    def list_imports(
        self, query: str | None = None, offset: int = 0, limit: int = 25
    ) -> list[ImportInfo]:
        """Lists all imported functions and symbols for a specified binary."""
        imports = []
        symbols = self.program.getSymbolTable().getExternalSymbols()
        for symbol in symbols:
            if query and not re.search(query, symbol.getName(), re.IGNORECASE):
                continue
            imports.append(
                ImportInfo(name=symbol.getName(), library=str(symbol.getParentNamespace()))
            )
        return imports[offset : limit + offset]

    @handle_exceptions
    def list_xrefs(self, name_or_address: str) -> list[CrossReferenceInfo]:
        """Finds and lists all cross-references (x-refs) to a given function, symbol,
        or address within a binary.
        """
        # Use the unified resolver
        sym: Symbol = self.find_symbol(name_or_address)
        addr = sym.getAddress()

        cross_references: list[CrossReferenceInfo] = []
        rm = self.program.getReferenceManager()
        references = rm.getReferencesTo(addr)

        for ref in references:
            from_func = self.program.getFunctionManager().getFunctionContaining(
                ref.getFromAddress()
            )
            cross_references.append(
                CrossReferenceInfo(
                    function_name=from_func.getName() if from_func else None,
                    from_address=str(ref.getFromAddress()),
                    to_address=str(ref.getToAddress()),
                    type=str(ref.getReferenceType()),
                )
            )
        return cross_references

    @handle_exceptions
    def disassemble_function(self, name_or_address: str) -> dict:
        """Return the assembly listing of the named or addressed function.

        Produces a flat text listing (`"addr: mnemonic operands ; comment"`
        per instruction) plus name/entry/signature/instruction_count.
        """
        func = self.find_function(name_or_address)
        listing = self.program.getListing()
        entry = func.getEntryPoint()
        end = func.getBody().getMaxAddress()

        try:
            from ghidra.program.model.listing import CommentType

            eol_comment = CommentType.EOL
        except ImportError:
            from ghidra.program.model.listing import CodeUnit

            eol_comment = CodeUnit.EOL_COMMENT

        lines: list[str] = []
        instruction_count = 0
        for instr in listing.getInstructions(entry, True):
            if instr.getAddress().compareTo(end) > 0:
                break

            addr_str = str(instr.getAddress())
            instr_text = str(instr)
            try:
                comment = listing.getComment(eol_comment, instr.getAddress())
            except Exception:
                comment = None

            if comment:
                lines.append(f"{addr_str}: {instr_text} ; {comment}")
            else:
                lines.append(f"{addr_str}: {instr_text}")
            instruction_count += 1

        signature = func.getSignature()
        return {
            "name": str(func.getName()),
            "entry": str(entry),
            "signature": str(signature) if signature is not None else None,
            "listing": "\n".join(lines),
            "instruction_count": instruction_count,
        }

    def _get_related_functions(self, name_or_address: str, direction: str) -> list[FunctionRef]:
        """Return called-from or calling-into functions, sorted by address.

        ``direction`` is ``"callees"`` (functions called by this one) or
        ``"callers"`` (functions that call this one).
        """
        from ghidra.util.task import ConsoleTaskMonitor

        func = self.find_function(name_or_address)
        monitor = ConsoleTaskMonitor()
        if direction == "callees":
            related = func.getCalledFunctions(monitor)
        elif direction == "callers":
            related = func.getCallingFunctions(monitor)
        else:
            raise ValueError(f"Unknown direction {direction!r}")

        refs = [FunctionRef(name=str(f.getName()), address=str(f.getEntryPoint())) for f in related]
        refs.sort(key=lambda r: r.address)
        return refs

    @handle_exceptions
    def list_callees(
        self, name_or_address: str, offset: int = 0, limit: int = 100
    ) -> list[FunctionRef]:
        """List functions called by the given function, sorted by address."""
        refs = self._get_related_functions(name_or_address, "callees")
        return refs[offset : offset + limit]

    @handle_exceptions
    def list_callers(
        self, name_or_address: str, offset: int = 0, limit: int = 100
    ) -> list[FunctionRef]:
        """List functions that call the given function, sorted by address."""
        refs = self._get_related_functions(name_or_address, "callers")
        return refs[offset : offset + limit]

    @handle_exceptions
    def get_referenced_strings(self, name_or_address: str) -> list[str]:
        """Get string literals referenced within the given function's body."""
        from ghidra.program.model.data import AbstractStringDataType as StringDataType

        func = self.find_function(name_or_address)
        listing = self.program.getListing()
        strings: list[str] = []
        body = func.getBody()

        for insn in listing.getInstructions(body, True):
            for ref in insn.getReferencesFrom():
                data = listing.getDefinedDataAt(ref.getToAddress())
                if data is not None and isinstance(data.getDataType(), StringDataType):
                    val = data.getValue()
                    if val is not None:
                        strings.append(str(val))

        return strings

    def _search_code_literal(
        self,
        literal_results: typing.Any,
        limit: int,
        offset: int,
        include_full_code: bool,
        preview_length: int,
    ) -> list[CodeSearchResult]:
        search_results: list[CodeSearchResult] = []
        if literal_results and literal_results.get("documents"):
            # Apply offset and limit
            docs = literal_results["documents"] or []
            metadatas = literal_results["metadatas"] or []

            # Paginate
            start_idx = offset
            end_idx = offset + limit
            paginated_docs = docs[start_idx:end_idx]
            paginated_meta = metadatas[start_idx:end_idx] if metadatas else []

            for i, doc in enumerate(paginated_docs):
                metadata = paginated_meta[i] if i < len(paginated_meta) else {}
                code = doc
                preview = None

                if not include_full_code:
                    preview = code[:preview_length] + "..." if len(code) > preview_length else code
                    code = preview

                search_results.append(
                    CodeSearchResult(
                        function_name=str(
                            metadata.get("function_name", "unknown")
                            if isinstance(metadata, dict)
                            else "unknown"
                        ),
                        code=code,
                        similarity=1.0,  # Exact match
                        search_mode=SearchMode.LITERAL,
                        preview=preview,
                    )
                )
        return search_results

    def _search_code_semantic(
        self,
        query: str,
        limit: int,
        offset: int,
        similarity_threshold: float,
        include_full_code: bool,
        preview_length: int,
        total_functions: int,  # Added total_functions to correctly calculate semantic_total
    ) -> tuple[list[CodeSearchResult], int]:  # Changed return type to int for semantic_total
        assert self.program_info.code_collection is not None
        search_results: list[CodeSearchResult] = []
        # Semantic search
        results = self.program_info.code_collection.query(
            query_texts=[query],
            n_results=limit + offset,
        )

        docs_list = results.get("documents") if results else None
        semantic_total = total_functions  # Initialize semantic_total here

        if results and docs_list and len(docs_list) > 0 and docs_list[0]:
            # Apply offset
            docs = docs_list[0][offset:]
            metadatas_list = results.get("metadatas")
            distances_list = results.get("distances")
            metadatas = (
                metadatas_list[0][offset:] if metadatas_list and len(metadatas_list) > 0 else []
            )
            distances = (
                distances_list[0][offset:] if distances_list and len(distances_list) > 0 else []
            )

            for i, doc in enumerate(docs):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0
                # ChromaDB uses L2 distance by default (0 = identical, can be > 1)
                # Normalize to 0-1 range where 1 = identical
                similarity = 1 / (1 + distance)

                # Skip results below similarity threshold
                if similarity < similarity_threshold:
                    continue

                code = doc
                preview = None

                if not include_full_code:
                    preview = code[:preview_length] + "..." if len(code) > preview_length else code
                    code = preview

                search_results.append(
                    CodeSearchResult(
                        function_name=str(
                            metadata.get("function_name", "unknown")
                            if isinstance(metadata, dict)
                            else "unknown"
                        ),
                        code=code,
                        similarity=similarity,
                        search_mode=SearchMode.SEMANTIC,
                        preview=preview,
                    )
                )

            # Refine semantic_total
            # If we got fewer results than requested limit (after filtering),
            # providing we fetched enough (n_results was limit+offset)
            # and we processed strictly what we asked for.
            # Actually, if the RAW result count was less than n_results, we know we exhausted
            # the DB.
            # If valid_results_count < limit, we *might* have exhausted matches above threshold
            # in this batch.
            # A better heuristic: if result count < limit, we found everything.
            if len(search_results) < limit:
                # This is only accurate if we assume we found "the end".
                # However, since we queried limit + offset, if we got less than limit (and we
                # started at offset),
                # it implies we are at the tail.
                semantic_total = offset + len(search_results)

        return search_results, semantic_total

    @handle_exceptions
    def search_code(
        self,
        query: str,
        limit: int = 10,
        offset: int = 0,
        search_mode: SearchMode = SearchMode.SEMANTIC,
        include_full_code: bool = True,
        preview_length: int = 500,
        similarity_threshold: float = 0.0,
    ) -> SearchCodeResponse:
        """
        Searches the code in the binary for a given query.

        Supports semantic (vector similarity) and literal (exact match) modes.
        Always returns dual-mode counts to help LLM decide on mode switching.

        Args:
            similarity_threshold: Minimum similarity score (0.0-1.0) for semantic results.
                                  Results below this threshold are filtered out.
        """
        if not self.program_info.code_collection:
            raise ValueError(
                "Code indexing is not complete for this binary. Please try again later."
            )

        # ALWAYS get literal count (reuse for literal mode search)
        literal_results = self.program_info.code_collection.get(where_document={"$contains": query})
        literal_total = (
            len(literal_results["ids"]) if literal_results and literal_results.get("ids") else 0
        )

        # Total functions in collection (absolute total)
        total_functions = self.program_info.code_collection.count()

        # Default semantic total to "available" (filtered by limit)
        # If we filter and get FEWER than requested, we effectively found "all" above threshold
        # in this range.
        # But we don't know beyond the limit.
        # So we default to total_functions as "estimated matches" if we hit the limit.
        semantic_total = total_functions

        search_results: list[CodeSearchResult] = []

        if search_mode == SearchMode.LITERAL:
            search_results = self._search_code_literal(
                literal_results, limit, offset, include_full_code, preview_length
            )
        else:
            search_results, estimated_total = self._search_code_semantic(
                query,
                limit,
                offset,
                similarity_threshold,
                include_full_code,
                preview_length,
                total_functions,
            )
            if estimated_total is not None:
                semantic_total = estimated_total

        return SearchCodeResponse(
            results=search_results,
            query=query,
            search_mode=search_mode,
            returned_count=len(search_results),
            offset=offset,
            limit=limit,
            literal_total=literal_total,
            semantic_total=semantic_total,
            total_functions=total_functions,
        )

    @handle_exceptions
    def search_strings(self, query: str, limit: int = 100) -> list[StringSearchResult]:
        """Searches for strings within a binary using substring matching."""

        if self.program_info.strings is None:
            raise ValueError(
                "String indexing is not complete for this binary. Please try again later."
            )

        query_lower = query.lower()
        return [
            StringSearchResult(value=s.value, address=s.address, similarity=1.0)
            for s in self.program_info.strings
            if query_lower in s.value.lower()
        ][:limit]

    @handle_exceptions
    def read_bytes(self, address: str, size: int = 32) -> ReadBytesResponse:
        """Reads raw bytes from memory at a specified address."""
        # Maximum size limit to prevent excessive memory reads
        max_read_size = 8192

        if size <= 0:
            raise ValueError("size must be > 0")

        if size > max_read_size:
            raise ValueError(f"Size {size} exceeds maximum {max_read_size}")

        # Get address factory and parse address
        af = self.program.getAddressFactory()

        try:
            # Handle common hex address formats
            addr_str = address
            if address.lower().startswith("0x"):
                addr_str = address[2:]

            addr = af.getAddress(addr_str)
            if addr is None:
                raise ValueError(f"Invalid address: {address}")
        except Exception as e:
            raise ValueError(f"Invalid address format '{address}': {e}") from e

        # Check if address is in valid memory
        mem = self.program.getMemory()
        if not mem.contains(addr):
            raise ValueError(f"Address {address} is not in mapped memory")

        # Use JPype to handle byte arrays properly for PyGhidra
        # Create Java byte array - JPype's runtime magic confuses static type checkers
        buf = JByte[size]  # type: ignore[reportInvalidTypeArguments]
        n = mem.getBytes(addr, buf)

        # Convert Java signed bytes (-128 to 127) to Python unsigned (0 to 255)
        if n > 0:
            data = bytes([b & 0xFF for b in buf[:n]])  # type: ignore[reportGeneralTypeIssues]
        else:
            data = b""

        return ReadBytesResponse(
            address=str(addr),
            size=len(data),
            data=data.hex(),
        )

    @handle_exceptions
    def gen_callgraph(
        self,
        function_name_or_address: str,
        cg_direction: CallGraphDirection = CallGraphDirection.CALLING,
        cg_display_type: CallGraphDisplayType = CallGraphDisplayType.FLOW,
        include_refs: bool = True,
        max_depth: int | None = None,
        max_run_time: int = 60,
        condense_threshold: int = 50,
        top_layers: int = 5,
        bottom_layers: int = 5,
    ) -> GenCallgraphResponse:
        """Generates a call graph for a specified function."""

        cg_func = self.find_function(function_name_or_address)
        mermaid_url: str = ""

        # Call the ghidrecomp function
        name, direction, _, graphs_data = gen_callgraph(
            func=cg_func,
            max_display_depth=max_depth,
            direction=cg_direction.value,
            max_run_time=max_run_time,
            name=cg_func.getSymbol().getName(True),
            include_refs=include_refs,
            condense_threshold=condense_threshold,
            top_layers=top_layers,
            bottom_layers=bottom_layers,
            wrap_mermaid=False,
        )

        selected_graph_content = ""
        for graph_type, graph_content in graphs_data:
            if CallGraphDisplayType(graph_type) == cg_display_type:
                selected_graph_content = graph_content
                break

        if not selected_graph_content:
            raise ValueError(
                f"Cg display type {cg_display_type.value} not found for function {cg_func}."
            )

        for graph_type, graph_content in graphs_data:
            if graph_type == "mermaid_url":
                mermaid_url = graph_content.split("\n")[0]
                break

        return GenCallgraphResponse(
            function_name=name,
            direction=CallGraphDirection(direction),
            display_type=cg_display_type,
            graph=selected_graph_content,
            mermaid_url=mermaid_url,
        )

    @handle_exceptions
    def rename_function(self, name_or_address: str, new_name: str) -> dict:
        from ghidra.program.model.symbol import SourceType

        func = self.find_function(name_or_address)
        old_name = str(func.getName())
        address = str(func.getEntryPoint())

        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: rename {old_name} -> {new_name}",
        ):
            func.setName(new_name, SourceType.USER_DEFINED)

        self.invalidate_decompiler_cache()
        return {
            "address": address,
            "old_name": old_name,
            "new_name": new_name,
        }

    @handle_exceptions
    def save_program(self) -> dict:
        from ghidra.util.task import ConsoleTaskMonitor

        program = self.program
        # Ghidra/pyghidra leaves a stray sub-transaction with an empty
        # description open during program load. ``save`` rejects the lock as
        # long as any sub-transaction is open ("Unable to lock due to active
        # transaction"), so close every still-open entry first by reaching
        # into ``DomainObjectDBTransaction`` via reflection.
        if not bool(program.canLock()):
            tx_info = program.getCurrentTransactionInfo()
            if tx_info is not None:
                self._force_end_open_subtransactions(program, tx_info)

        program.save("pyghidra-mcp: save_program", ConsoleTaskMonitor())
        return {"saved": True}

    @staticmethod
    def _force_end_open_subtransactions(program, tx_info) -> None:
        """End every still-open ``TransactionEntry`` of ``tx_info``.

        ``DomainObjectDBTransaction`` assigns each entry an id of
        ``baseId + list_index``; passing that to ``program.endTransaction``
        closes the entry. Entries with status != NOT_DONE are already closed
        and are skipped to avoid IllegalStateException.
        """
        cls = tx_info.getClass()
        try:
            list_field = cls.getDeclaredField("list")
            base_field = cls.getDeclaredField("baseId")
        except Exception as e:
            logger.debug("save_program: cannot access transaction internals: %s", e)
            return
        list_field.setAccessible(True)
        base_field.setAccessible(True)
        entries = list_field.get(tx_info)
        base_id = int(base_field.get(tx_info))
        if entries is None:
            return
        for index in range(entries.size()):
            entry = entries.get(index)
            try:
                ec = entry.getClass()
                status_f = ec.getDeclaredField("status")
                status_f.setAccessible(True)
                status = status_f.get(entry)
                if str(status) != "NOT_DONE":
                    continue
                desc_f = ec.getDeclaredField("description")
                desc_f.setAccessible(True)
                desc_str = desc_f.get(entry)
                entry_id = base_id + index
                program.endTransaction(entry_id, True)
                logger.debug(
                    "save_program: closed leftover sub-transaction id=%s desc=%r",
                    entry_id, desc_str,
                )
            except Exception as e:
                logger.debug("save_program: failed to close entry %d: %s", index, e)

    @handle_exceptions
    def create_function(
        self,
        address: str,
        name: str | None = None,
        disassemble_first: bool = True,
    ) -> dict:
        """Create a function at ``address``.

        When ``disassemble_first`` is True (default) and there are no
        instructions at the target, runs Ghidra's disassembler first so
        ``CreateFunctionCmd`` has something to work with. Optionally renames
        the new function to ``name``. Errors if a function already exists at
        the address.
        """
        from ghidra.app.cmd.disassemble import DisassembleCommand
        from ghidra.app.cmd.function import CreateFunctionCmd
        from ghidra.program.model.address import AddressSet
        from ghidra.program.model.symbol import SourceType
        from ghidra.util.task import TaskMonitor

        addr = self._parse_address(address)
        fm = self.program.getFunctionManager()
        if fm.getFunctionAt(addr) is not None:
            existing = fm.getFunctionAt(addr)
            raise ValueError(f"Function already exists at {address}: {existing.getName()}")

        listing = self.program.getListing()
        with ghidra_transaction(self.program, f"pyghidra-mcp: create_function @ {addr}"):
            if disassemble_first and listing.getInstructionAt(addr) is None:
                addr_set = AddressSet(addr, addr)
                disasm_cmd = DisassembleCommand(addr_set, None, True)
                if not disasm_cmd.applyTo(self.program, TaskMonitor.DUMMY):
                    raise ValueError(
                        f"Failed to disassemble at {address}: {disasm_cmd.getStatusMsg()}"
                    )

            create_cmd = CreateFunctionCmd(addr)
            if not create_cmd.applyTo(self.program, TaskMonitor.DUMMY):
                raise ValueError(
                    f"Failed to create function at {address}: {create_cmd.getStatusMsg()}"
                )

            func = fm.getFunctionAt(addr)
            if func is None:
                raise ValueError(
                    f"Function creation reported success but function not found at {address}"
                )

            if name:
                func.setName(name, SourceType.USER_DEFINED)

        self.invalidate_decompiler_cache()
        return {
            "name": str(func.getName()),
            "entry_point": str(func.getEntryPoint()),
            "body_size": int(func.getBody().getNumAddresses()),
        }

    @handle_exceptions
    def delete_function(self, name_or_address: str) -> dict:
        """Delete the function identified by ``name_or_address``.

        Accepts a name or entry-point address (same lookup as
        ``rename_function``). Returns the deleted function's name and entry
        point for confirmation. Errors if no function matches.
        """
        func = self.find_function(name_or_address)
        name = str(func.getName())
        entry_point = str(func.getEntryPoint())

        with ghidra_transaction(
            self.program, f"pyghidra-mcp: delete_function {name} @ {entry_point}"
        ):
            removed = self.program.getFunctionManager().removeFunction(func.getEntryPoint())
            if not removed:
                raise ValueError(f"Failed to remove function {name} @ {entry_point}")

        self.invalidate_decompiler_cache()
        return {
            "name": name,
            "entry_point": entry_point,
        }

    def _resolve_calling_convention(self, calling_convention: str) -> str:
        """Match ``calling_convention`` to one of the program's compiler-spec
        names (with or without leading ``__``). Raises ``ValueError`` if the
        name is unknown, listing the available conventions for diagnostics.
        """
        target = calling_convention.lower()
        target_stripped = target.replace("__", "")
        available = self.program.getCompilerSpec().getCallingConventions()
        for model in available:
            model_name = str(model.getName()).lower()
            if (
                model_name == target
                or model_name == "__" + target
                or model_name.replace("__", "") == target_stripped
            ):
                return str(model.getName())
        names = ", ".join(str(m.getName()) for m in available)
        raise ValueError(f"Unknown calling convention '{calling_convention}'. Available: {names}")

    @handle_exceptions
    def set_function_prototype(
        self,
        name_or_address: str,
        prototype: str,
        calling_convention: str | None = None,
    ) -> dict:
        """Apply a full C-style prototype to a function.

        Parses ``prototype`` with Ghidra's ``FunctionSignatureParser`` and
        applies it via ``ApplyFunctionSignatureCmd``. Optionally sets
        ``calling_convention`` (e.g. ``"__cdecl"``, ``"__stdcall"``) — the
        target name must match one of the program's compiler-spec
        conventions, with or without the leading underscores.

        Preserves the function's existing plate comment across the change
        (``ApplyFunctionSignatureCmd`` would otherwise wipe it).

        Raises ``ValueError`` (-> INVALID_PARAMS) on parse failure, on an
        unknown calling convention, or if the command itself rejects the
        signature.
        """
        from ghidra.app.cmd.function import ApplyFunctionSignatureCmd
        from ghidra.app.util.parser import FunctionSignatureParser
        from ghidra.program.model.symbol import SourceType
        from ghidra.util.task import ConsoleTaskMonitor

        if not prototype or not prototype.strip():
            raise ValueError("prototype must be a non-empty string")

        func = self.find_function(name_or_address)
        entry_point = str(func.getEntryPoint())
        addr = func.getEntryPoint()

        dtm = self.program.getDataTypeManager()
        parser = FunctionSignatureParser(dtm, None)
        try:
            sig = parser.parse(None, prototype)
        except Exception as e:
            raise ValueError(f"Failed to parse prototype {prototype!r}: {e}") from e
        if sig is None:
            raise ValueError(f"Failed to parse prototype {prototype!r}")

        # Resolve calling convention (if any) before opening the transaction so
        # we surface a clean INVALID_PARAMS without a half-applied state.
        resolved_cc_name = (
            self._resolve_calling_convention(calling_convention) if calling_convention else None
        )

        saved_plate_comment = func.getComment()

        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: set prototype {func.getName()} @ {entry_point}",
        ):
            cmd = ApplyFunctionSignatureCmd(addr, sig, SourceType.USER_DEFINED)
            if not cmd.applyTo(self.program, ConsoleTaskMonitor()):
                raise ValueError(f"ApplyFunctionSignatureCmd failed: {cmd.getStatusMsg()}")

            if resolved_cc_name is not None:
                func.setCallingConvention(resolved_cc_name)

            # ApplyFunctionSignatureCmd can clobber the plate comment; restore it.
            if saved_plate_comment:
                current = func.getComment()
                if not current or current.startswith("Setting prototype:"):
                    func.setComment(saved_plate_comment)

        self.invalidate_decompiler_cache()
        return {
            "name": str(func.getName()),
            "entry_point": entry_point,
            "signature": str(func.getSignature()),
        }

    @handle_exceptions
    def rename_variable(
        self,
        function_name_or_address: str,
        variable_name: str,
        new_name: str,
    ) -> dict:
        from ghidra.program.model.symbol import SourceType

        func, variable_kind, variable = self._resolve_function_variable(
            function_name_or_address, variable_name
        )
        old_name = str(variable_name)
        function_name = str(func.getName())
        function_address = str(func.getEntryPoint())
        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: rename {variable_kind} {old_name} -> {new_name}",
        ):
            variable.setName(new_name, SourceType.USER_DEFINED)

        self.invalidate_decompiler_cache()
        return {
            "function_name": function_name,
            "function_address": function_address,
            "variable_kind": variable_kind,
            "old_name": old_name,
            "new_name": new_name,
        }

    @handle_exceptions
    def set_variable_type(
        self,
        function_name_or_address: str,
        variable_name: str,
        type_name: str,
    ) -> dict:
        from ghidra.program.model.symbol import SourceType

        func, variable_kind, variable = self._resolve_function_variable(
            function_name_or_address, variable_name
        )
        function_name = str(func.getName())
        function_address = str(func.getEntryPoint())
        old_type = str(variable.getDataType().getDisplayName())
        data_type = self._parse_data_type(type_name)

        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: set {variable_kind} type {variable_name} -> {type_name}",
        ):
            variable.setDataType(data_type, SourceType.USER_DEFINED)

        self.invalidate_decompiler_cache()
        return {
            "function_name": function_name,
            "function_address": function_address,
            "variable_kind": variable_kind,
            "variable_name": str(variable.getName()),
            "old_type": old_type,
            "new_type": str(variable.getDataType().getDisplayName()),
        }

    @handle_exceptions
    def set_comment(self, target: str, comment: str, comment_type: str) -> dict:
        try:
            from ghidra.program.model.listing import CommentType

            listing_comment_types = {
                "plate": CommentType.PLATE,
                "pre": CommentType.PRE,
                "eol": CommentType.EOL,
                "post": CommentType.POST,
                "repeatable": CommentType.REPEATABLE,
            }
        except ImportError:
            from ghidra.program.model.listing import CodeUnit

            listing_comment_types = {
                "plate": CodeUnit.PLATE_COMMENT,
                "pre": CodeUnit.PRE_COMMENT,
                "eol": CodeUnit.EOL_COMMENT,
                "post": CodeUnit.POST_COMMENT,
                "repeatable": CodeUnit.REPEATABLE_COMMENT,
            }

        normalized_type = comment_type.lower()
        if normalized_type == "decompiler":
            func = self.find_function(target)
            addr = func.getEntryPoint()

            with ghidra_transaction(
                self.program,
                f"pyghidra-mcp: set function comment @ {addr}",
            ):
                func.setComment(comment)

            self.invalidate_decompiler_cache()
            return {
                "address": str(addr),
                "comment": comment,
                "comment_type": "decompiler",
            }

        ghidra_comment_type = listing_comment_types.get(normalized_type)
        if ghidra_comment_type is None:
            allowed = ["decompiler", *listing_comment_types.keys()]
            raise ValueError(f"Invalid comment_type '{comment_type}'. Expected one of: {allowed}")

        addr = self._parse_address(target)
        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: set {normalized_type} comment @ {addr}",
        ):
            self.program.getListing().setComment(addr, ghidra_comment_type, comment)

        self.invalidate_decompiler_cache()
        return {
            "address": str(addr),
            "comment": comment,
            "comment_type": normalized_type,
        }

    def run_inline_script(self, code: str, args: list[str] | None = None) -> dict:
        """Execute inline Python code against the program.

        The code runs inside a Ghidra transaction that commits on normal
        completion and rolls back on exception. Available names in the script
        namespace: `program` (alias `currentProgram`), `monitor`, and `args`.
        Assign to `result` to return a value (its repr is captured).
        """
        import contextlib
        import io

        from ghidra.util.task import TaskMonitor

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        namespace: dict = {
            "program": self.program,
            "currentProgram": self.program,
            "monitor": TaskMonitor.DUMMY,
            "args": list(args or []),
        }

        error: str | None = None
        tx_id = self.program.startTransaction("pyghidra-mcp: run_inline_script")
        committed = False
        try:
            with (
                contextlib.redirect_stdout(stdout_buf),
                contextlib.redirect_stderr(stderr_buf),
            ):
                try:
                    exec(code, namespace)
                    committed = True
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
        finally:
            self.program.endTransaction(tx_id, committed)

        if committed:
            self.invalidate_decompiler_cache()

        has_result = "result" in namespace
        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue(),
            "result_repr": repr(namespace["result"]) if has_result else None,
            "committed": committed,
            "error": error,
        }

    def run_script(self, script: str, args: list[str] | None = None) -> dict:
        """Load a Python script from disk and execute it against the program.

        See `_resolve_script_path` for the search chain. Once located, the
        file's contents are executed through the same path as
        `run_inline_script` — inside a Ghidra transaction with `program`,
        `currentProgram`, `monitor`, and `args` in the namespace.
        """
        resolved = _resolve_script_path(script)
        if resolved is None:
            raise ValueError(f"Script not found: {script}")

        code = resolved.read_text()
        result = self.run_inline_script(code, args=args)
        result["script"] = script
        result["resolved_path"] = str(resolved)
        return result

    @handle_exceptions
    def create_struct_type(self, name: str, fields: list) -> dict:
        """Create a new structure data type.

        `fields` is a list of StructField (or dicts with name/type/offset). If
        any field has an explicit offset, all fields are placed explicitly
        (initial size is computed to fit); otherwise fields are appended in
        order.
        """
        from ghidra.program.model.data import StructureDataType

        if not name:
            raise ValueError("name is required")
        if not fields:
            raise ValueError("at least one field is required")

        dtm = self.program.getDataTypeManager()
        if dtm.getDataType(f"/{name}") is not None:
            raise ValueError(f"Data type '{name}' already exists")

        # Normalize + resolve field types up front so we fail before touching
        # the program if anything is bad.
        resolved = self._resolve_composite_fields(fields)
        has_offsets = any(f.get("offset") is not None for f, _ in resolved)
        init_size = _compute_struct_init_size(resolved) if has_offsets else 0

        with ghidra_transaction(self.program, f"pyghidra-mcp: create_struct_type {name}"):
            struct = StructureDataType(name, init_size)
            _layout_struct_fields(struct, resolved, has_offsets)
            created = dtm.addDataType(struct, None)

        self.invalidate_decompiler_cache()
        return {
            "name": str(created.getName()),
            "size": int(created.getLength()),
        }

    def _resolve_composite_fields(self, fields: list) -> list[tuple[dict, typing.Any]]:
        """Normalize field dicts and resolve each declared type via Ghidra.

        Shared by create_struct_type / create_union_type. Raises ValueError
        if a field is missing name/type or references an unknown type.
        """
        resolved: list[tuple[dict, typing.Any]] = []
        for f in fields:
            field = f.model_dump() if hasattr(f, "model_dump") else dict(f)
            fname = field.get("name")
            ftype = field.get("type")
            if not fname or not ftype:
                raise ValueError(f"field is missing name/type: {field}")
            dt = self._parse_data_type(ftype)
            if dt is None:
                raise ValueError(f"Unknown field type: {ftype}")
            resolved.append((field, dt))
        return resolved

    @handle_exceptions
    def create_enum_type(self, name: str, values: dict, size: int = 4) -> dict:
        """Create a new enumeration data type.

        `values` is a {member_name: int_value} mapping. `size` must be 1, 2,
        4, or 8 bytes.
        """
        from ghidra.program.model.data import EnumDataType

        if not name:
            raise ValueError("name is required")
        if size not in (1, 2, 4, 8):
            raise ValueError(f"Invalid size {size}; must be 1, 2, 4, or 8")
        if not values:
            raise ValueError("at least one enum value is required")

        dtm = self.program.getDataTypeManager()
        if dtm.getDataType(f"/{name}") is not None:
            raise ValueError(f"Data type '{name}' already exists")

        with ghidra_transaction(self.program, f"pyghidra-mcp: create_enum_type {name}"):
            enum_dt = EnumDataType(name, size)
            for member_name, member_value in values.items():
                enum_dt.add(str(member_name), int(member_value))
            created = dtm.addDataType(enum_dt, None)

        self.invalidate_decompiler_cache()
        return {
            "name": str(created.getName()),
            "size": int(created.getLength()),
        }

    @handle_exceptions
    def create_union_type(self, name: str, fields: list) -> dict:
        """Create a new union data type.

        `fields` is a list of StructField (or dicts with name/type). The
        `offset` attribute on each field is ignored for unions.
        """
        from ghidra.program.model.data import UnionDataType

        if not name:
            raise ValueError("name is required")
        if not fields:
            raise ValueError("at least one field is required")

        dtm = self.program.getDataTypeManager()
        if dtm.getDataType(f"/{name}") is not None:
            raise ValueError(f"Data type '{name}' already exists")

        # Resolve upfront so we fail fast. Unions ignore the `offset` attr.
        resolved = self._resolve_composite_fields(fields)

        with ghidra_transaction(self.program, f"pyghidra-mcp: create_union_type {name}"):
            union = UnionDataType(name)
            for field, dt in resolved:
                union.add(dt, field["name"], None)
            created = dtm.addDataType(union, None)

        self.invalidate_decompiler_cache()
        return {
            "name": str(created.getName()),
            "size": int(created.getLength()),
        }

    @handle_exceptions
    def create_array_type(self, name: str, base_type: str, length: int) -> dict:
        """Create a named array data type (e.g. ``int[10]`` aliased to ``name``).

        `length` is the element count (must be > 0). Errors if a type with
        the given name already exists.
        """
        from ghidra.program.model.data import ArrayDataType

        if not name:
            raise ValueError("name is required")
        if not base_type:
            raise ValueError("base_type is required")
        if length <= 0:
            raise ValueError(f"length must be positive (got {length})")

        dtm = self.program.getDataTypeManager()
        if dtm.getDataType(f"/{name}") is not None:
            raise ValueError(f"Data type '{name}' already exists")

        base_dt = self._parse_data_type(base_type)
        if base_dt is None:
            raise ValueError(f"Unknown base type: {base_type}")

        with ghidra_transaction(self.program, f"pyghidra-mcp: create_array_type {name}"):
            array = ArrayDataType(base_dt, length, base_dt.getLength())
            array.setName(name)
            created = dtm.addDataType(array, None)

        self.invalidate_decompiler_cache()
        return {
            "name": str(created.getName()),
            "size": int(created.getLength()),
        }

    @handle_exceptions
    def create_pointer_type(self, name: str, base_type: str) -> dict:
        """Create a named pointer data type (``base_type *`` aliased to ``name``).

        Special-cases ``base_type == "void"`` to use Ghidra's built-in
        ``VoidDataType``. Errors if a type with the given name already exists.
        """
        from ghidra.program.model.data import PointerDataType, VoidDataType

        if not name:
            raise ValueError("name is required")
        if not base_type:
            raise ValueError("base_type is required")

        dtm = self.program.getDataTypeManager()
        if dtm.getDataType(f"/{name}") is not None:
            raise ValueError(f"Data type '{name}' already exists")

        if base_type == "void":
            base_dt = VoidDataType.dataType
        else:
            base_dt = self._parse_data_type(base_type)
        if base_dt is None:
            raise ValueError(f"Unknown base type: {base_type}")

        with ghidra_transaction(self.program, f"pyghidra-mcp: create_pointer_type {name}"):
            pointer = PointerDataType(base_dt)
            pointer.setName(name)
            created = dtm.addDataType(pointer, None)

        self.invalidate_decompiler_cache()
        return {
            "name": str(created.getName()),
            "size": int(created.getLength()),
        }

    @handle_exceptions
    def create_typedef(self, name: str, base_type: str) -> dict:
        """Create a typedef alias for an existing data type.

        `base_type` is passed to Ghidra's DataTypeParser, so pointer/array
        syntax (``Foo *``, ``int[10]``) is supported natively. Errors if a
        type with the given name already exists.
        """
        from ghidra.program.model.data import TypedefDataType

        if not name:
            raise ValueError("name is required")
        if not base_type:
            raise ValueError("base_type is required")

        dtm = self.program.getDataTypeManager()
        if dtm.getDataType(f"/{name}") is not None:
            raise ValueError(f"Data type '{name}' already exists")

        base_dt = self._parse_data_type(base_type)
        if base_dt is None:
            raise ValueError(f"Unknown base type: {base_type}")

        with ghidra_transaction(self.program, f"pyghidra-mcp: create_typedef {name}"):
            typedef = TypedefDataType(name, base_dt)
            created = dtm.addDataType(typedef, None)

        self.invalidate_decompiler_cache()
        return {
            "name": str(created.getName()),
            "size": int(created.getLength()),
        }

    @handle_exceptions
    def apply_data_type(
        self,
        address: str,
        type_name: str,
        clear_existing: bool = True,
    ) -> dict:
        """Apply a data type at the given memory address.

        When ``clear_existing`` is True (default), any code/data units in the
        target range are cleared before creating the new data. When False,
        ``createData`` may raise if the range is occupied.
        """
        if not address:
            raise ValueError("address is required")
        if not type_name:
            raise ValueError("type_name is required")

        dt = self._parse_data_type(type_name)
        if dt is None:
            raise ValueError(f"Unknown data type: {type_name}")

        addr = self._parse_address(address)
        if not self.program.getMemory().contains(addr):
            raise ValueError(f"Address is not in program memory: {address}")

        listing = self.program.getListing()
        expected_size = int(dt.getLength())

        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: apply_data_type {type_name} @ {addr}",
        ):
            if clear_existing and expected_size > 0:
                end_addr = addr.add(expected_size - 1)
                listing.clearCodeUnits(addr, end_addr, False)
            data = listing.createData(addr, dt)

        self.invalidate_decompiler_cache()
        applied_size = int(data.getLength()) if data is not None else 0
        return {
            "address": str(addr),
            "applied_size": applied_size,
        }

    @handle_exceptions
    def delete_data_type(self, name_or_path: str) -> dict:
        """Delete a data type from the program's data type manager.

        Resolves ``name_or_path`` via the standard
        ``_resolve_data_type_by_name_or_path`` lookup (full DTM path or bare
        name). Captures name/path/kind before deletion so the response can
        confirm exactly what was removed. Errors with INVALID_PARAMS if the
        type doesn't exist or if Ghidra refuses to remove it (typically
        because it's referenced elsewhere).
        """
        if not name_or_path:
            raise ValueError("name_or_path is required")

        dtm = self.program.getDataTypeManager()
        dt = _resolve_data_type_by_name_or_path(dtm, name_or_path)
        if dt is None:
            raise ValueError(f"Data type not found: {name_or_path}")

        # Capture identity before deletion — these calls are unsafe afterwards.
        name = str(dt.getName())
        path = str(dt.getPathName())
        kind = self._classify_data_type(dt)

        with ghidra_transaction(self.program, f"pyghidra-mcp: delete_data_type {path}"):
            removed = dtm.remove(dt, None)
            if not removed:
                raise ValueError(
                    f"Failed to delete data type '{path}'. "
                    "It may still be referenced; clear references first."
                )

        self.invalidate_decompiler_cache()
        return {
            "name": name,
            "path": path,
            "kind": kind,
        }

    def _build_imported_data_type_info(self, dt) -> dict:
        """Render a DataType as an ``DataTypeInfo``-shaped dict for response use."""
        length = int(dt.getLength())
        return {
            "name": str(dt.getName()),
            "kind": self._classify_data_type(dt),
            "size": length if length > 0 else None,
            "path": str(dt.getPathName()),
        }

    def _move_to_category(self, dt, target_category, handler) -> str:
        """Move ``dt`` into ``target_category`` and return the new pathname.

        Returns the original pathname on failure so the caller can still log
        a sensible response (the type itself remains in the DTM either way).
        """
        try:
            target_category.moveDataType(dt, handler)
        except Exception:
            logger.debug(
                "Failed to move %s to %s", dt.getPathName(), target_category, exc_info=True
            )
        return str(dt.getPathName())

    @handle_exceptions
    def import_c_types(
        self,
        source: str,
        category_path: str | None = None,
        replace: bool = False,
    ) -> dict:
        """Parse a C-declarations string and add the resulting types to the DTM.

        Uses Ghidra's ``CParser``. ``source`` is a free-form string of C
        declarations: ``struct``, ``union``, ``enum``, ``typedef``, function
        prototypes. ``#include`` directives are NOT resolved — paste flat
        declarations.

        ``category_path`` (e.g. ``"/MyLib"``) places newly-added types in
        the given DTM category, creating it if needed; ``None`` keeps them
        at the root.

        ``replace=False`` (default) preserves any pre-existing type with
        the same name and reports it under ``skipped``. ``replace=True``
        overwrites the existing type with the parsed one.

        Returns lists of imported types (with kind/size/path), skipped
        type names, and any per-type errors emitted by the parser.
        """
        from ghidra.app.util.cparser.C import CParser
        from ghidra.program.model.data import CategoryPath, DataTypeConflictHandler

        if not source or not source.strip():
            raise ValueError("source must be a non-empty C declarations string")

        dtm = self.program.getDataTypeManager()

        # Resolve / create the target category up front.
        target_category = None
        if category_path:
            cp_str = category_path if category_path.startswith("/") else "/" + category_path
            cp = CategoryPath(cp_str)
            target_category = dtm.getCategory(cp) or dtm.createCategory(cp)

        # Snapshot existing pathnames so we can classify each parsed type.
        pre_existing_paths: set[str] = {str(dt.getPathName()) for dt in dtm.getAllDataTypes()}

        handler = (
            DataTypeConflictHandler.REPLACE_HANDLER
            if replace
            else DataTypeConflictHandler.DEFAULT_HANDLER
        )

        imported: list[dict] = []
        skipped: list[str] = []
        errors: list[str] = []

        with ghidra_transaction(self.program, "pyghidra-mcp: import_c_types"):
            parser = CParser(dtm, True, None)
            try:
                parser.parse(source)
            except Exception as e:
                raise ValueError(f"Failed to parse C source: {e}") from e

            # Ghidra's CParser (12.x) has no getDefinedTypes(): it exposes the
            # parsed types through per-kind maps. Union composites/enums/typedefs/
            # function-defs, de-duplicating by path name, and iterate those.
            defined_types: list = []
            _seen_paths: set[str] = set()
            for _getter in (
                parser.getComposites,
                parser.getEnums,
                parser.getTypes,
                parser.getFunctions,
            ):
                for dt in _getter().values():
                    _p = str(dt.getPathName())
                    if _p not in _seen_paths:
                        _seen_paths.add(_p)
                        defined_types.append(dt)

            for dt in defined_types:
                path = str(dt.getPathName())

                # CParser produces a `.conflict` variant when a name collides
                # under DEFAULT_HANDLER. Drop it and record the original as skipped.
                if ".conflict" in path:
                    base_path = path.split(".conflict", 1)[0]
                    if not replace:
                        try:
                            dtm.remove(dt, None)
                        except Exception as e:
                            errors.append(f"Failed to clean up {path}: {e}")
                        skipped.append(base_path)
                    else:
                        # With REPLACE_HANDLER this branch shouldn't fire; surface it if it does.
                        errors.append(f"Unexpected conflict variant under replace=True: {path}")
                    continue

                # Pre-existing path means the parser reused (or replaced) an existing type.
                if path in pre_existing_paths and not replace:
                    skipped.append(path)
                    continue

                if target_category is not None:
                    path = self._move_to_category(dt, target_category, handler)

                imported.append(self._build_imported_data_type_info(dt))

        self.invalidate_decompiler_cache()
        return {
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
        }

    @staticmethod
    def _classify_data_type(dt) -> str:
        """Map a Ghidra DataType to one of our DataTypeKind string values."""
        from ghidra.program.model.data import (
            Array,
            BuiltInDataType,
            Enum as GhidraEnum,
            FunctionDefinition,
            Pointer,
            Structure,
            TypeDef,
            Union as GhidraUnion,
        )

        if isinstance(dt, Structure):
            return "struct"
        if isinstance(dt, GhidraUnion):
            return "union"
        if isinstance(dt, GhidraEnum):
            return "enum"
        if isinstance(dt, Array):
            return "array"
        if isinstance(dt, Pointer):
            return "pointer"
        if isinstance(dt, TypeDef):
            return "typedef"
        if isinstance(dt, FunctionDefinition):
            return "function"
        if isinstance(dt, BuiltInDataType):
            return "primitive"
        return "other"

    @staticmethod
    def _category_matches(category_path: str, filter_cat: str) -> bool:
        """True if ``category_path`` is ``filter_cat`` or a descendant of it.

        Treats the category path as a directory tree:
          - ``filter_cat="/"`` matches every type (root prefix).
          - ``filter_cat="/MyLib"`` matches ``/MyLib`` exactly and any
            ``/MyLib/Sub/...``, but NOT ``/MyLibrary/...``.
          - The match is case-sensitive (Ghidra category names are identifiers).
        """
        if filter_cat == "/":
            return True
        exact = filter_cat.rstrip("/")
        return category_path == exact or category_path.startswith(exact + "/")

    @handle_exceptions
    def search_data_types(
        self,
        query: str = ".*",
        kinds: list | None = None,
        category: str | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DataTypeInfo]:
        """Search the program's data type manager.

        ``query`` is a case-insensitive regex matched against the type name.
        ``kinds`` is an optional list of DataTypeKind values (or their string
        equivalents). ``None`` or omitted means include every kind (including
        primitives).
        ``category`` is an optional DTM folder path (e.g. ``"/MyLib"``).
        When set, results are restricted to that folder and its descendants;
        ``"/"`` matches everything. The leading ``/`` is added if missing.
        """
        if not query:
            query = ".*"

        # Normalize kinds to a set of kind-string values; None means all.
        if kinds:
            kind_set = {k.value if hasattr(k, "value") else str(k) for k in kinds}
        else:
            kind_set = None

        # Normalize category: strip empty, ensure leading slash.
        if category:
            if not category.startswith("/"):
                category = "/" + category
        else:
            category = None

        dtm = self.program.getDataTypeManager()
        results: list[DataTypeInfo] = []

        for dt in dtm.getAllDataTypes():
            dt_kind = self._classify_data_type(dt)

            if kind_set is not None and dt_kind not in kind_set:
                continue

            if category is not None:
                cat_path = str(dt.getCategoryPath().getPath())
                if not self._category_matches(cat_path, category):
                    continue

            name = str(dt.getName())
            if not self._matches_query(query, name):
                continue

            length = int(dt.getLength())
            results.append(
                DataTypeInfo(
                    name=name,
                    kind=dt_kind,
                    size=length if length > 0 else None,
                    path=str(dt.getPathName()),
                )
            )

        return results[offset : limit + offset]

    @handle_exceptions
    def search_data_items(
        self,
        query: str = ".*",
        min_refcount: int | None = None,
        max_refcount: int | None = None,
        offset: int = 0,
        limit: int = 100,
    ) -> list[DataItemInfo]:
        """Search defined data items in the program.

        Iterates ``Listing.getDefinedData(True)`` over every memory block.
        Each match reports the label (or synthesized ``DAT_<addr>``), the
        data type name, the byte length, and the xref refcount.
        ``query`` is a case-insensitive regex matched against the label.
        """
        if not query:
            query = ".*"

        listing = self.program.getListing()
        rm = self.program.getReferenceManager()
        results: list[DataItemInfo] = []

        for data in listing.getDefinedData(True):
            addr = data.getAddress()
            addr_str = str(addr)
            refcount = int(rm.getReferenceCountTo(addr))

            if min_refcount is not None and refcount < min_refcount:
                continue
            if max_refcount is not None and refcount > max_refcount:
                continue

            label = data.getLabel()
            name = str(label) if label is not None else f"DAT_{addr_str}"
            if not self._matches_query(query, name):
                continue

            dt = data.getDataType()
            type_name = str(dt.getName()) if dt is not None else "undefined"

            results.append(
                DataItemInfo(
                    name=name,
                    address=addr_str,
                    type=type_name,
                    length=int(data.getLength()),
                    refcount=refcount,
                )
            )

        return results[offset : limit + offset]

    def _resolve_data_type_or_raise(self, name_or_path: str, expected_kind: str):
        """Resolve a DataType via name-or-path lookup; raise on miss / wrong kind."""
        dtm = self.program.getDataTypeManager()
        dt = _resolve_data_type_by_name_or_path(dtm, name_or_path)
        if dt is None:
            raise ValueError(f"Data type not found: {name_or_path}")
        actual_kind = self._classify_data_type(dt)
        if actual_kind != expected_kind:
            raise ValueError(
                f"Data type '{name_or_path}' is a {actual_kind}, not a {expected_kind}"
            )
        return dt

    @handle_exceptions
    def get_struct_layout(self, name_or_path: str) -> dict:
        """Return the field layout of a struct identified by name or DTM path."""
        dt = self._resolve_data_type_or_raise(name_or_path, "struct")
        fields = []
        for component in dt.getDefinedComponents():
            field_name = component.getFieldName()
            fields.append(
                {
                    "offset": int(component.getOffset()),
                    "size": int(component.getLength()),
                    "type": str(component.getDataType().getName()),
                    "name": str(field_name) if field_name else None,
                }
            )
        return {
            "name": str(dt.getName()),
            "path": str(dt.getPathName()),
            "size": int(dt.getLength()),
            "fields": fields,
        }

    @handle_exceptions
    def get_union_layout(self, name_or_path: str) -> dict:
        """Return the field layout of a union identified by name or DTM path."""
        dt = self._resolve_data_type_or_raise(name_or_path, "union")
        fields = []
        for component in dt.getDefinedComponents():
            field_name = component.getFieldName()
            fields.append(
                {
                    "size": int(component.getLength()),
                    "type": str(component.getDataType().getName()),
                    "name": str(field_name) if field_name else None,
                }
            )
        return {
            "name": str(dt.getName()),
            "path": str(dt.getPathName()),
            "size": int(dt.getLength()),
            "fields": fields,
        }

    @handle_exceptions
    def get_enum_values(self, name_or_path: str) -> dict:
        """Return the {member_name: value} map of an enum identified by name or DTM path."""
        dt = self._resolve_data_type_or_raise(name_or_path, "enum")
        values: dict[str, int] = {}
        for value_name in dt.getNames():
            values[str(value_name)] = int(dt.getValue(value_name))
        return {
            "name": str(dt.getName()),
            "path": str(dt.getPathName()),
            "size": int(dt.getLength()),
            "values": values,
        }

    @staticmethod
    def _lookup_struct_field(
        struct, field_name: str | None = None, field_offset: int | None = None
    ):
        """Find a struct component by name or offset; raise on miss / bad input.

        Exactly one of ``field_name`` or ``field_offset`` must be provided.
        Offset lookup uses ``Structure.getComponentAt`` (returns the component
        containing that byte, even mid-field — Ghidra's natural semantic).
        """
        if (field_name is None) == (field_offset is None):
            raise ValueError("provide exactly one of field_name or field_offset")

        if field_name is not None:
            for component in struct.getDefinedComponents():
                cname = component.getFieldName()
                if cname is not None and str(cname) == field_name:
                    return component
            raise ValueError(f"Field '{field_name}' not found in struct")

        component = struct.getComponentAt(int(field_offset))
        if component is None:
            raise ValueError(f"No field at offset {field_offset} in struct")
        return component

    @handle_exceptions
    def add_struct_field(
        self,
        struct_name_or_path: str,
        field_name: str,
        field_type: str,
        offset: int | None = None,
    ) -> dict:
        """Add a field to an existing struct.

        ``offset=None`` (default) appends at the end. A specific offset
        overlays into existing padding (or grows the struct to fit). The
        field type is parsed via ``DataTypeParser`` so pointer/array syntax
        is supported (``"int*"``, ``"char[16]"``).
        """
        if not field_name:
            raise ValueError("field_name is required")
        if not field_type:
            raise ValueError("field_type is required")

        struct = self._resolve_data_type_or_raise(struct_name_or_path, "struct")
        new_dt = self._parse_data_type(field_type)
        if new_dt is None:
            raise ValueError(f"Unknown field type: {field_type}")

        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: add_struct_field {struct.getName()}.{field_name}",
        ):
            if offset is None:
                struct.add(new_dt, new_dt.getLength(), field_name, None)
            else:
                if offset < 0:
                    raise ValueError(f"offset must be >= 0 (got {offset})")
                needed = offset + new_dt.getLength() - struct.getLength()
                if needed > 0:
                    struct.growStructure(needed)
                struct.replaceAtOffset(offset, new_dt, new_dt.getLength(), field_name, None)

        self.invalidate_decompiler_cache()

        # Resolve the actual placed offset (when appending we need to look it up)
        placed = self._lookup_struct_field(struct, field_name=field_name)
        return {
            "struct_name": str(struct.getName()),
            "offset": int(placed.getOffset()),
            "struct_size": int(struct.getLength()),
        }

    @handle_exceptions
    def set_struct_field(
        self,
        struct_name_or_path: str,
        field_name: str | None = None,
        field_offset: int | None = None,
        new_name: str | None = None,
        new_type: str | None = None,
    ) -> dict:
        """Set the name and/or type of an existing struct field.

        Identify the target via ``field_name`` OR ``field_offset`` (exactly
        one). At least one of ``new_name`` / ``new_type`` must be provided.
        """
        if new_name is None and new_type is None:
            raise ValueError("nothing to modify; pass new_name or new_type")

        struct = self._resolve_data_type_or_raise(struct_name_or_path, "struct")
        component = self._lookup_struct_field(
            struct, field_name=field_name, field_offset=field_offset
        )
        ordinal = int(component.getOrdinal())

        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: set_struct_field {struct.getName()}[{ordinal}]",
        ):
            if new_type is not None:
                new_dt = self._parse_data_type(new_type)
                if new_dt is None:
                    raise ValueError(f"Unknown field type: {new_type}")
                struct.replace(ordinal, new_dt, new_dt.getLength())
                # struct.replace() may return a fresh component handle
                component = struct.getComponent(ordinal)

            if new_name is not None:
                component.setFieldName(new_name)

        self.invalidate_decompiler_cache()
        component = struct.getComponent(ordinal)
        return {
            "struct_name": str(struct.getName()),
            "offset": int(component.getOffset()),
            "struct_size": int(struct.getLength()),
        }

    @handle_exceptions
    def delete_struct_field(
        self,
        struct_name_or_path: str,
        field_name: str | None = None,
        field_offset: int | None = None,
    ) -> dict:
        """Delete a field from a struct.

        Identify the target via ``field_name`` OR ``field_offset`` (exactly
        one). The struct's total size shrinks accordingly.
        """
        struct = self._resolve_data_type_or_raise(struct_name_or_path, "struct")
        component = self._lookup_struct_field(
            struct, field_name=field_name, field_offset=field_offset
        )
        ordinal = int(component.getOrdinal())
        offset = int(component.getOffset())

        with ghidra_transaction(
            self.program,
            f"pyghidra-mcp: delete_struct_field {struct.getName()}[{ordinal}]",
        ):
            struct.delete(ordinal)

        self.invalidate_decompiler_cache()
        return {
            "struct_name": str(struct.getName()),
            "offset": offset,
            "struct_size": int(struct.getLength()),
        }

    def invalidate_decompiler_cache(self) -> None:
        try:
            self.decompiler_pool.invalidate_all()
        except Exception:
            logger.debug("Failed to invalidate decompiler cache", exc_info=True)

    def _parse_address(self, address: str):
        addr_str = address[2:] if address.lower().startswith("0x") else address
        addr = self.program.getAddressFactory().getAddress(addr_str)
        if addr is None:
            raise ValueError(f"Invalid address: {address}")
        return addr



