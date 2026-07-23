"""
MCP Tool handlers for pyghidra-mcp.

This module contains all MCP tool implementations with centralized error handling.
"""

import asyncio
import functools
import logging
from typing import Literal, cast

from mcp.server.fastmcp import Context
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ErrorData

from pyghidra_mcp.context_protocol import MCPContext
from pyghidra_mcp.models import (
    ApplyDataTypeResponse,
    CallGraphDirection,
    CallGraphDisplayType,
    CreateDataTypeResponse,
    CreateFunctionResponse,
    CrossReferenceInfos,
    DataTypeInfo,
    DataTypeKind,
    DecompiledFunction,
    DisassembledFunction,
    GenCallgraphResponse,
    GetEnumValuesResponse,
    GetStructLayoutResponse,
    GetUnionLayoutResponse,
    ImportCTypesResponse,
    ListFunctionCallsResponse,
    ListProgramsResponse,
    ReadBytesResponse,
    RenameFunctionResponse,
    RenameVariableResponse,
    RunInlineScriptResponse,
    RunScriptResponse,
    SaveProgramResponse,
    ScriptJobHandle,
    ScriptJobStatus,
    SearchCodeResponse,
    SearchDataItemsResponse,
    SearchDataTypesResponse,
    SearchFunctionsResponse,
    SearchMode,
    SearchStringsResponse,
    SearchSymbolsResponse,
    SetCommentResponse,
    SetFunctionPrototypeResponse,
    SetVariableTypeResponse,
    StructField,
    SymbolKind,
)
from pyghidra_mcp.tools import GhidraTools

logger = logging.getLogger(__name__)


def _run_for_context(pyghidra_context: MCPContext, fn):
    from pyghidra_mcp.gui_context import GuiPyGhidraContext

    if isinstance(pyghidra_context, GuiPyGhidraContext):
        return pyghidra_context.run_on_swing(fn)
    return fn()


def _get_action_name(func_name: str) -> str:
    """Derives a gerund action name from a function name."""
    action = func_name.replace("_", " ")
    words = action.split()
    if words and not words[0].endswith("ing"):
        first = words[0]
        if first.endswith("e"):
            words[0] = first[:-1] + "ing"
        else:
            words[0] = first + "ing"
    return " ".join(words)


def mcp_error_handler(func):
    """
    Decorator that provides centralized error handling for MCP tools.
    """
    action = _get_action_name(func.__name__)

    def handle_error(e):
        if isinstance(e, ValueError):
            return McpError(ErrorData(code=INVALID_PARAMS, message=str(e)))
        if isinstance(e, McpError):
            return e
        return McpError(ErrorData(code=INTERNAL_ERROR, message=f"Error {action}: {e!s}"))

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            raise handle_error(e) from e

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise handle_error(e) from e

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# MCP Tool Implementations
# ---------------------------------------------------------------------------------


@mcp_error_handler
async def decompile_function(
    program_name: str,
    name_or_address: str | list[str],
    ctx: Context,
    include_callees: bool = False,
    include_strings: bool = False,
    include_xrefs: bool = False,
    timeout_sec: int = 30,
) -> list[DecompiledFunction]:
    """Decompile function(s) to pseudo-C by name or address.

    Accepts a single target or a list for batch decompilation.
    Rich response flags attach callees, strings, and/or xrefs to each result.
    `timeout_sec` applies per target.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    targets = [name_or_address] if isinstance(name_or_address, str) else name_or_address
    results: list[DecompiledFunction] = []

    def _decompile_target(target: str) -> DecompiledFunction:
        result = tools.decompile_function_by_name_or_addr(target, timeout=timeout_sec)
        if include_callees:
            result.callees = ListFunctionCallsResponse(functions=tools.list_callees(target))
        if include_strings:
            result.referenced_strings = tools.get_referenced_strings(target)
        if include_xrefs:
            result.xrefs = tools.list_xrefs(target)
        return result

    for target in targets:
        try:
            result = await asyncio.to_thread(_decompile_target, target)
            results.append(result)
        except Exception as e:
            results.append(DecompiledFunction(name=target, code="", error=str(e)))
    return results


@mcp_error_handler
async def disassemble_function(
    program_name: str,
    name_or_address: str | list[str],
    ctx: Context,
) -> list[DisassembledFunction]:
    """Disassemble function(s) to assembly text by name or address.

    Accepts a single target or a list for batch disassembly. Each result's
    ``listing`` field carries the flat assembly dump (one ``"addr: mnemonic
    operands ; EOL comment"`` line per instruction).
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    targets = [name_or_address] if isinstance(name_or_address, str) else name_or_address
    results: list[DisassembledFunction] = []

    def _disassemble_target(target: str) -> DisassembledFunction:
        return DisassembledFunction(**tools.disassemble_function(target))

    for target in targets:
        try:
            result = await asyncio.to_thread(_disassemble_target, target)
            results.append(result)
        except Exception as e:
            results.append(
                DisassembledFunction(
                    name=target,
                    entry="",
                    listing="",
                    instruction_count=0,
                    error=str(e),
                )
            )
    return results


@mcp_error_handler
def search_symbols(
    program_name: str,
    ctx: Context,
    query: str = ".*",
    kinds: list[SymbolKind] | None = None,
    offset: int = 0,
    limit: int = 100,
) -> SearchSymbolsResponse:
    """Search symbols by regex pattern (case-insensitive), optionally filtered
    to one or more kinds.

    Supports full regex (e.g. ``^main$``, ``func.*init``). Plain substrings
    still work since they are valid regex. ``query`` defaults to ``".*"`` so
    you can list every symbol with no pattern.

    ``kinds`` is an optional list restricting the result. Omit or pass
    ``null`` to include every symbol. Allowed values:

      - ``"functions"`` — only function symbols
      - ``"globals"`` — non-function symbols in the global namespace
      - ``"labels"`` — only LABEL-type symbols (code + data labels)

    Examples:
      - ``kinds=["functions"]`` — just functions
      - ``kinds=["globals","labels"]`` — any global non-function or any label
      - omitted — every symbol (labels, functions, classes, namespaces, ...)
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    symbols = tools.search_symbols(query, kinds=kinds, offset=offset, limit=limit)
    return SearchSymbolsResponse(symbols=symbols)


@mcp_error_handler
def search_functions(
    program_name: str,
    ctx: Context,
    query: str = ".*",
    min_refcount: int | None = None,
    max_refcount: int | None = None,
    user_defined_only: bool = False,
    include_thunks: bool = True,
    include_externals: bool = True,
    offset: int = 0,
    limit: int = 100,
) -> SearchFunctionsResponse:
    """Search functions with optional filters specific to function analysis.

    ``query`` is a case-insensitive regex matched against the function name
    (defaults to ``".*"`` so the call lists every function unless narrowed).

    Optional filters:
      - ``min_refcount`` / ``max_refcount`` — bound xref counts (inclusive).
        Useful for triage: ``min_refcount=10`` finds heavily-called functions.
      - ``user_defined_only`` — exclude Ghidra autogenerated names like
        ``FUN_*``, ``LAB_*``, etc. ("show me what humans / debug info named").
      - ``include_thunks`` — set False to skip thunk wrappers.
      - ``include_externals`` — set False to skip imported/external symbols.

    Each result carries ``{name, address, refcount, is_thunk, is_external,
    is_user_defined}``. Results are returned in address order. Pagination
    via ``offset`` / ``limit`` (default 100). For a less function-specific
    search, use ``search_symbols(kinds=["functions"])``.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    functions = tools.search_functions(
        query=query,
        min_refcount=min_refcount,
        max_refcount=max_refcount,
        user_defined_only=user_defined_only,
        include_thunks=include_thunks,
        include_externals=include_externals,
        offset=offset,
        limit=limit,
    )
    return SearchFunctionsResponse(functions=functions)


@mcp_error_handler
def search_code(
    program_name: str,
    query: str,
    ctx: Context,
    limit: int = 5,
    offset: int = 0,
    search_mode: Literal["semantic", "literal"] = "semantic",
    include_full_code: bool = True,
    preview_length: int = 500,
    similarity_threshold: float = 0.0,
) -> SearchCodeResponse:
    """Search decompiled pseudo-C code.

    Modes: semantic (vector similarity, default) or literal (exact match).
    Results include both mode counts.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    return tools.search_code(
        query=query,
        limit=limit,
        offset=offset,
        search_mode=SearchMode(search_mode),
        include_full_code=include_full_code,
        preview_length=preview_length,
        similarity_threshold=similarity_threshold,
    )


@mcp_error_handler
def search_data_types(
    program_name: str,
    ctx: Context,
    query: str = ".*",
    kinds: list[DataTypeKind] | None = None,
    category: str | None = None,
    offset: int = 0,
    limit: int = 100,
) -> SearchDataTypesResponse:
    """Search the program's data type manager.

    ``query`` is a case-insensitive regex matched against the type name
    (defaults to ``".*"`` so you can list every type with no pattern).

    ``kinds`` is an optional list restricting the result to one or more
    classifications. Omit or pass ``null`` to include every kind. Allowed
    values:

      - ``"struct"`` / ``"union"`` / ``"enum"`` — composite types
      - ``"array"`` / ``"pointer"`` / ``"typedef"`` — derived types
      - ``"function"`` — function signature (prototype) types
      - ``"primitive"`` — Ghidra built-in types (``int``, ``char``, ``void``, ...)

    ``category`` is an optional Data Type Manager folder path (e.g.
    ``"/MyLib"``, ``"/GNU/g++/std"``). When set, results are restricted to
    that folder and any descendants. ``"/"`` matches every category.
    Matching is case-sensitive and respects directory boundaries:
    ``"/MyLib"`` matches ``/MyLib`` and ``/MyLib/Sub`` but NOT
    ``/MyLibrary``. The leading slash is added if you omit it.

    Examples:
      - ``kinds=["struct","union","enum"]`` — just composites
      - ``kinds=["primitive"]`` — only Ghidra built-ins
      - ``category="/GNU/g++/std"`` — every type under that namespace
      - ``query="Point", category="/MyLib"`` — disambiguate among types named ``Point``
      - omitted — everything, including primitives
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    data_types = tools.search_data_types(
        query=query,
        kinds=kinds,
        category=category,
        offset=offset,
        limit=limit,
    )
    return SearchDataTypesResponse(data_types=data_types)


@mcp_error_handler
def search_data_items(
    program_name: str,
    ctx: Context,
    query: str = ".*",
    min_refcount: int | None = None,
    max_refcount: int | None = None,
    offset: int = 0,
    limit: int = 100,
) -> SearchDataItemsResponse:
    """Search defined data items laid down in the program's memory.

    Distinct from ``search_data_types`` (which returns DTM type
    *definitions*) — this tool returns concrete data *instances* at memory
    addresses, with their applied type and refcount. Useful for finding
    interesting globals: strings, lookup tables, magic numbers,
    function-pointer tables.

    ``query`` is a case-insensitive regex matched against the data item's
    label (defaults to ``".*"`` so the call lists everything unless
    narrowed). When a data item has no user label, the synthesized name
    ``"DAT_<address>"`` is used and is matched against the query.

    Optional refcount filters:
      - ``min_refcount`` — find heavily-referenced globals (e.g. ``10`` to
        surface lookup tables / shared config blocks).
      - ``max_refcount`` — find rarely-referenced data (e.g. ``0`` for dead
        constants).

    Each result reports ``{name, address, type, length, refcount}``.
    Results are returned in address order. Pagination via ``offset`` /
    ``limit`` (default 100). For type definitions, see ``search_data_types``.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    data_items = tools.search_data_items(
        query=query,
        min_refcount=min_refcount,
        max_refcount=max_refcount,
        offset=offset,
        limit=limit,
    )
    return SearchDataItemsResponse(data_items=data_items)


@mcp_error_handler
def get_struct_layout(
    program_name: str,
    name_or_path: str,
    ctx: Context,
) -> GetStructLayoutResponse:
    """Return the field layout of a struct.

    ``name_or_path`` accepts either a bare name (``"Point"``) or a full DTM
    path (``"/MyLib/Point"``). Bare-name lookups try root first, then scan
    every category — pass the full path from ``search_data_types``'s
    ``path`` field if the name is ambiguous.

    Each field reports ``{offset, size, type, name?}``. ``type`` is the
    bare data-type name (e.g. ``"int"``, ``"char[16]"``); call
    ``get_struct_layout`` recursively to drill into nested structs.
    Padding/undefined components are skipped — only declared fields appear.
    Errors with INVALID_PARAMS if the type isn't found or isn't a struct.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.get_struct_layout(name_or_path),
    )
    result = cast(dict, result)
    return GetStructLayoutResponse(program_name=program_name, **result)


@mcp_error_handler
def get_union_layout(
    program_name: str,
    name_or_path: str,
    ctx: Context,
) -> GetUnionLayoutResponse:
    """Return the field layout of a union.

    Same lookup semantics as ``get_struct_layout`` (bare name or DTM path).
    Each field reports ``{size, type, name?}`` — there's no offset because
    union members all share offset 0. Errors with INVALID_PARAMS if the
    type isn't found or isn't a union.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.get_union_layout(name_or_path),
    )
    result = cast(dict, result)
    return GetUnionLayoutResponse(program_name=program_name, **result)


@mcp_error_handler
def get_enum_values(
    program_name: str,
    name_or_path: str,
    ctx: Context,
) -> GetEnumValuesResponse:
    """Return the ``{member_name: value}`` map of an enum.

    Same lookup semantics as ``get_struct_layout`` (bare name or DTM path).
    The response shape is symmetric with ``create_enum_type``'s ``values``
    parameter — pass the response back into ``create_enum_type`` to clone
    the enum elsewhere. Errors with INVALID_PARAMS if the type isn't found
    or isn't an enum.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.get_enum_values(name_or_path),
    )
    result = cast(dict, result)
    return GetEnumValuesResponse(program_name=program_name, **result)


@mcp_error_handler
def list_programs(ctx: Context) -> ListProgramsResponse:
    """List every program in the current Ghidra project with analysis/index status.

    Returns all programs stored in the project on disk — including ones that are
    not currently open in any CodeBrowser tab. See ``list_open_programs`` (GUI
    mode only) for the narrower "what's open right now" view.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    return ListProgramsResponse(programs=pyghidra_context.list_program_infos())


@mcp_error_handler
def rename_function(
    program_name: str,
    name_or_address: str,
    new_name: str,
    ctx: Context,
) -> RenameFunctionResponse:
    """Rename a function using a Ghidra transaction."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.rename_function(name_or_address, new_name),
    )
    result = cast(dict, result)
    return RenameFunctionResponse(program_name=program_name, **result)


@mcp_error_handler
def save_program(program_name: str, ctx: Context) -> SaveProgramResponse:
    """Persist all pending changes for ``program_name`` to the project on disk."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(pyghidra_context, lambda: tools.save_program())
    result = cast(dict, result)
    return SaveProgramResponse(program_name=program_name, **result)


@mcp_error_handler
def create_function(
    program_name: str,
    address: str,
    ctx: Context,
    name: str | None = None,
    disassemble_first: bool = True,
) -> CreateFunctionResponse:
    """Create a function at the given entry-point address.

    ``address`` is the entry point of the new function (accepts ``"0x1000"``
    or ``"space:offset"`` for multi-space programs).

    ``name`` optionally renames the new function — without it, Ghidra
    autogenerates a ``FUN_*`` placeholder.

    ``disassemble_first`` (default True) runs Ghidra's disassembler at the
    target address first if no instructions exist there. Set False if the
    bytes are already disassembled and you want to skip the extra step.

    Errors with INVALID_PARAMS if a function already exists at the address
    or if disassembly / creation fails. The mutation is wrapped in a Ghidra
    transaction that rolls back on any failure.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.create_function(address, name=name, disassemble_first=disassemble_first),
    )
    result = cast(dict, result)
    return CreateFunctionResponse(program_name=program_name, **result)


@mcp_error_handler
def set_function_prototype(
    program_name: str,
    name_or_address: str,
    prototype: str,
    ctx: Context,
    calling_convention: str | None = None,
) -> SetFunctionPrototypeResponse:
    """Apply a full C-style prototype to a function in a single call.

    ``prototype`` is a C-style signature string parsed by Ghidra's
    ``FunctionSignatureParser`` — for example
    ``"int handle_request(Request *req, size_t len)"``. Types referenced in
    the prototype must already exist in the program's data type manager
    (use ``create_struct_type`` / ``create_typedef`` / etc. first if needed).

    ``calling_convention`` is optional — pass a name like ``"__cdecl"`` or
    ``"__stdcall"`` to set it alongside the signature. The name must match
    one of the program's compiler-spec conventions (with or without leading
    underscores). Omit it to leave the function's current convention alone.

    The function's existing plate comment is preserved across the change.

    Errors with INVALID_PARAMS if the prototype fails to parse, the calling
    convention is unknown, or Ghidra rejects the signature command. The
    mutation is wrapped in a Ghidra transaction that rolls back on failure.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.set_function_prototype(
            name_or_address, prototype, calling_convention=calling_convention
        ),
    )
    result = cast(dict, result)
    return SetFunctionPrototypeResponse(program_name=program_name, **result)


@mcp_error_handler
def rename_variable(
    program_name: str,
    function_name_or_address: str,
    variable_name: str,
    new_name: str,
    ctx: Context,
) -> RenameVariableResponse:
    """Rename a function parameter or local variable by exact name within a function.

    Missing or ambiguous names return an error instead of guessing.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.rename_variable(function_name_or_address, variable_name, new_name),
    )
    result = cast(dict, result)
    return RenameVariableResponse(program_name=program_name, **result)


@mcp_error_handler
def set_variable_type(
    program_name: str,
    function_name_or_address: str,
    variable_name: str,
    type_name: str,
    ctx: Context,
) -> SetVariableTypeResponse:
    """Set the data type for a function parameter or local variable by exact name.

    Missing or ambiguous names return an error instead of guessing.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.set_variable_type(function_name_or_address, variable_name, type_name),
    )
    result = cast(dict, result)
    return SetVariableTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def set_comment(
    program_name: str,
    target: str,
    comment: str,
    comment_type: Literal["decompiler", "plate", "pre", "eol", "post", "repeatable"],
    ctx: Context,
) -> SetCommentResponse:
    """Set a comment in the decompiler or listing."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.set_comment(target, comment, comment_type),
    )
    result = cast(dict, result)
    return SetCommentResponse(program_name=program_name, **result)


@mcp_error_handler
def create_struct_type(
    program_name: str,
    name: str,
    fields: list[StructField],
    ctx: Context,
) -> CreateDataTypeResponse:
    """Create a new structure data type in the program's Data Type Manager.

    Each field is `{name, type, offset?}`. If any field has an explicit
    `offset`, all fields are placed at explicit offsets and the struct is
    sized to fit; otherwise fields are appended in order. Errors if a type
    with the same name already exists.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.create_struct_type(name, fields),
    )
    result = cast(dict, result)
    return CreateDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def create_enum_type(
    program_name: str,
    name: str,
    values: dict[str, int],
    ctx: Context,
    size: int = 4,
) -> CreateDataTypeResponse:
    """Create a new enumeration data type in the program's Data Type Manager.

    `values` is a {member_name: int_value} mapping. `size` must be 1, 2, 4,
    or 8 bytes (default 4). Errors if a type with the same name already
    exists.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.create_enum_type(name, values, size),
    )
    result = cast(dict, result)
    return CreateDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def create_union_type(
    program_name: str,
    name: str,
    fields: list[StructField],
    ctx: Context,
) -> CreateDataTypeResponse:
    """Create a new union data type in the program's Data Type Manager.

    Each field is `{name, type}`. The `offset` attribute is ignored for
    unions. Errors if a type with the same name already exists.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.create_union_type(name, fields),
    )
    result = cast(dict, result)
    return CreateDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def create_array_type(
    program_name: str,
    name: str,
    base_type: str,
    length: int,
    ctx: Context,
) -> CreateDataTypeResponse:
    """Create a named array data type in the program's Data Type Manager.

    `length` is the element count (must be > 0). Errors if a type with the
    same name already exists.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.create_array_type(name, base_type, length),
    )
    result = cast(dict, result)
    return CreateDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def create_pointer_type(
    program_name: str,
    name: str,
    base_type: str,
    ctx: Context,
) -> CreateDataTypeResponse:
    """Create a named pointer data type in the program's Data Type Manager.

    Pass ``base_type="void"`` for a void pointer. Errors if a type with the
    same name already exists.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.create_pointer_type(name, base_type),
    )
    result = cast(dict, result)
    return CreateDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def create_typedef(
    program_name: str,
    name: str,
    base_type: str,
    ctx: Context,
) -> CreateDataTypeResponse:
    """Create a typedef alias for an existing data type.

    ``base_type`` is parsed by Ghidra, so pointer/array syntax (``Foo *``,
    ``int[10]``) is supported natively. Errors if a type with the same name
    already exists.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.create_typedef(name, base_type),
    )
    result = cast(dict, result)
    return CreateDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def import_c_types(
    program_name: str,
    source: str,
    ctx: Context,
    category_path: str | None = None,
    replace: bool = False,
) -> ImportCTypesResponse:
    """Parse a C-declarations string and add the resulting types to the DTM.

    ``source`` is a free-form string of C declarations: ``struct``,
    ``union``, ``enum``, ``typedef``, function prototypes. Forward and
    circular references between types in the same call are resolved by
    Ghidra's ``CParser``. Bitfields and anonymous nested aggregates are
    supported.

    Note: ``#include`` directives are NOT resolved — paste flat
    declarations only. For pre-built type libraries, use the ``--gdt``
    CLI flag at server startup instead.

    ``category_path`` (e.g. ``"/MyLib"``) places newly-added types in
    the given DTM category, creating it if needed. ``None`` keeps them
    at the DTM root.

    ``replace=False`` (default) preserves any pre-existing type with the
    same name and reports it under ``skipped``. ``replace=True`` overwrites
    the existing type with the parsed one (existing references are
    re-bound by Ghidra's conflict handler).

    Returns ``imported`` (per-type kind/size/path), ``skipped`` (paths kept
    intact), and ``errors`` (parser warnings or per-type post-processing
    failures). Raises INVALID_PARAMS if the C source fails to parse.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.import_c_types(source, category_path=category_path, replace=replace),
    )
    result = cast(dict, result)
    return ImportCTypesResponse(
        program_name=program_name,
        imported=[DataTypeInfo(**dt) for dt in result["imported"]],
        skipped=result["skipped"],
        errors=result["errors"],
    )


@mcp_error_handler
def apply_data_type(
    program_name: str,
    address: str,
    type_name: str,
    ctx: Context,
    clear_existing: bool = True,
) -> ApplyDataTypeResponse:
    """Apply a data type at a memory address.

    When ``clear_existing`` is True (default), any existing code/data in the
    target range is cleared first. When False, the call raises if the range
    is occupied. ``address`` accepts plain hex (e.g. ``0x1000``) or
    ``space:offset`` syntax (e.g. ``mem:1000``) for programs with multiple
    address spaces.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.apply_data_type(address, type_name, clear_existing),
    )
    result = cast(dict, result)
    return ApplyDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def run_inline_script(
    program_name: str,
    code: str,
    ctx: Context,
    args: list[str] | None = None,
) -> RunInlineScriptResponse:
    """Execute inline Python code against a program via pyghidra.

    The script namespace exposes `program` (alias `currentProgram`), `monitor`
    (a TaskMonitor), and `args` (the list passed in). Assign to `result` to
    return a value; its repr is captured.

    Execution runs inside a Ghidra transaction that commits on normal
    completion and rolls back if the code raises. stdout and stderr are
    captured and returned; exceptions are reported via the `error` field
    rather than propagated.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.run_inline_script(code, args=args),
    )
    result = cast(dict, result)
    return RunInlineScriptResponse(program_name=program_name, **result)


@mcp_error_handler
def run_script(
    program_name: str,
    script: str,
    ctx: Context,
    args: list[str] | None = None,
) -> RunScriptResponse:
    """Load a Python script from disk and execute it against the program.

    ``script`` accepts an absolute path, a CWD-relative path, or a name
    resolved through this search chain:

      1. ``script`` as-given.
      2. ``~/ghidra_scripts/<script>`` and ``~/ghidra_scripts/<basename>``.
      3. ``./ghidra_scripts/<script>`` and ``./ghidra_scripts/<basename>``.
      4. If ``script`` has no extension, each of the above is also tried
         with ``.py`` appended.

    Execution mirrors ``run_inline_script``: the file is exec'd inside a
    Ghidra transaction with ``program`` (alias ``currentProgram``),
    ``monitor``, and ``args`` in the namespace. Assigning to ``result`` in
    the script returns a value via ``result_repr``. stdout/stderr are
    captured and exceptions land in the ``error`` field.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.run_script(script, args=args),
    )
    result = cast(dict, result)
    return RunScriptResponse(program_name=program_name, **result)


@mcp_error_handler
def run_inline_script_async(
    program_name: str,
    code: str,
    ctx: Context,
    args: list[str] | None = None,
) -> ScriptJobHandle:
    """Submit ``run_inline_script`` to a background worker and return a job handle.

    Use this instead of ``run_inline_script`` when the script may exceed the
    MCP request timeout (typically 2 minutes). The script begins running
    immediately on a single-worker queue (jobs serialize to keep Ghidra
    transactions safe). Poll ``poll_script_job(job_id)`` to retrieve the
    result once status is ``"completed"`` or ``"failed"``.

    Same execution semantics as ``run_inline_script``: the script namespace
    exposes ``program``/``currentProgram``/``monitor``/``args``; assigning to
    ``result`` returns a value via ``result_repr``; the script runs in a
    Ghidra transaction that commits on normal completion and rolls back on
    exception.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    job = pyghidra_context.script_jobs.submit(
        program_name,
        lambda: cast(
            dict,
            _run_for_context(
                pyghidra_context,
                lambda: tools.run_inline_script(code, args=args),
            ),
        ),
    )
    return ScriptJobHandle(
        program_name=program_name,
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
    )


@mcp_error_handler
def run_script_async(
    program_name: str,
    script: str,
    ctx: Context,
    args: list[str] | None = None,
) -> ScriptJobHandle:
    """Submit ``run_script`` to a background worker and return a job handle.

    Use this instead of ``run_script`` when the script may exceed the MCP
    request timeout (typically 2 minutes). The script begins running
    immediately on a single-worker queue. Poll ``poll_script_job(job_id)``
    to retrieve the result once status is ``"completed"`` or ``"failed"``.

    Script-resolution and execution semantics match ``run_script`` exactly
    (same path search chain, same namespace, same transaction behavior).
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    job = pyghidra_context.script_jobs.submit(
        program_name,
        lambda: cast(
            dict,
            _run_for_context(
                pyghidra_context,
                lambda: tools.run_script(script, args=args),
            ),
        ),
    )
    return ScriptJobHandle(
        program_name=program_name,
        job_id=job.job_id,
        status=job.status,
        submitted_at=job.submitted_at,
    )


@mcp_error_handler
def poll_script_job(job_id: str, ctx: Context) -> ScriptJobStatus:
    """Look up an async script job's current status and result.

    Returns the job's lifecycle state plus per-stage timestamps. When
    ``status`` is ``"completed"`` or ``"failed"`` the result fields
    (``stdout``/``stderr``/``result_repr``/``committed``/``error``) are
    populated; while still ``queued``/``running`` they are ``None``.

    Errors with INVALID_PARAMS if ``job_id`` is unknown — either bad id
    or evicted (the registry retains the most recent jobs only).
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    job = pyghidra_context.script_jobs.get(job_id)
    if job is None:
        raise ValueError(
            f"Unknown script job id: {job_id!r}. "
            "It may have been evicted, or the server was restarted."
        )
    return ScriptJobStatus(**job.to_dict())


# @mcp_error_handler
# async def delete_project_binary(program_name: str, ctx: Context) -> str:
#     """Delete a binary from the project."""
#     pyghidra_context: MCPContext = ctx.request_context.lifespan_context
#     if pyghidra_context.delete_program(program_name):
#         return f"Successfully deleted binary: {program_name}"
#     else:
#         raise McpError(
#             ErrorData(
#                 code=INVALID_PARAMS,
#                 message=f"Binary '{program_name}' not found or could not be deleted.",
#             )
#         )
#
#
# @mcp_error_handler
# def list_exports(
#     program_name: str,
#     ctx: Context,
#     query: str = ".*",
#     offset: int = 0,
#     limit: int = 25,
# ) -> ExportInfos:
#     """List exported symbols, optionally filtered by regex query."""
#     pyghidra_context: MCPContext = ctx.request_context.lifespan_context
#     program_info = pyghidra_context.get_program_info(program_name)
#     tools = GhidraTools(program_info)
#     exports = tools.list_exports(query=query, offset=offset, limit=limit)
#     return ExportInfos(exports=exports)
#
#
# @mcp_error_handler
# def list_imports(
#     program_name: str,
#     ctx: Context,
#     query: str = ".*",
#     offset: int = 0,
#     limit: int = 25,
# ) -> ImportInfos:
#     """List imported symbols, optionally filtered by regex query."""
#     pyghidra_context: MCPContext = ctx.request_context.lifespan_context
#     program_info = pyghidra_context.get_program_info(program_name)
#     tools = GhidraTools(program_info)
#     imports = tools.list_imports(query=query, offset=offset, limit=limit)
#     return ImportInfos(imports=imports)


@mcp_error_handler
def list_callees(
    program_name: str,
    name_or_address: str,
    ctx: Context,
    offset: int = 0,
    limit: int = 100,
) -> ListFunctionCallsResponse:
    """List functions called by the given function, sorted by address.

    ``name_or_address`` accepts either a function name (e.g. ``main``) or an
    entry-point address (e.g. ``0x1000``, ``0x00400530``).
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    functions = tools.list_callees(name_or_address, offset=offset, limit=limit)
    return ListFunctionCallsResponse(functions=functions)


@mcp_error_handler
def list_callers(
    program_name: str,
    name_or_address: str,
    ctx: Context,
    offset: int = 0,
    limit: int = 100,
) -> ListFunctionCallsResponse:
    """List functions that call the given function, sorted by address.

    ``name_or_address`` accepts either a function name (e.g. ``main``) or an
    entry-point address (e.g. ``0x1000``, ``0x00400530``).
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    functions = tools.list_callers(name_or_address, offset=offset, limit=limit)
    return ListFunctionCallsResponse(functions=functions)


@mcp_error_handler
def list_xrefs(
    program_name: str, name_or_address: str | list[str], ctx: Context
) -> list[CrossReferenceInfos]:
    """List cross-references to function(s), symbol(s), or address(es).

    Accepts a single target or a list for batch lookup.
    Suggests close matches on no exact hit.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    targets = [name_or_address] if isinstance(name_or_address, str) else name_or_address
    results: list[CrossReferenceInfos] = []
    for target in targets:
        try:
            cross_references = tools.list_xrefs(target)
            results.append(CrossReferenceInfos(target=target, cross_references=cross_references))
        except Exception as e:
            results.append(CrossReferenceInfos(target=target, cross_references=[], error=str(e)))
    return results


@mcp_error_handler
def search_strings(
    program_name: str,
    ctx: Context,
    query: str,
    limit: int = 100,
) -> SearchStringsResponse:
    """Search for strings within a binary."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    strings = tools.search_strings(query=query, limit=limit)
    return SearchStringsResponse(strings=strings)


@mcp_error_handler
def read_bytes(program_name: str, ctx: Context, address: str, size: int = 32) -> ReadBytesResponse:
    """Read raw bytes at an address. Hex format supported (0x prefix optional)."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    return tools.read_bytes(address=address, size=size)


@mcp_error_handler
def gen_callgraph(
    program_name: str,
    function_name: str,
    ctx: Context,
    direction: Literal["calling", "called"] = "calling",
    display_type: Literal["flow", "flow_ends"] = "flow",
    condense_threshold: int = 50,
    top_layers: int = 3,
    bottom_layers: int = 3,
    max_run_time: int = 120,
) -> GenCallgraphResponse:
    """Generate a MermaidJS call graph for a function."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    return tools.gen_callgraph(
        function_name_or_address=function_name,
        cg_direction=CallGraphDirection(direction),
        cg_display_type=CallGraphDisplayType(display_type),
        include_refs=True,
        max_depth=None,
        max_run_time=max_run_time,
        condense_threshold=condense_threshold,
        top_layers=top_layers,
        bottom_layers=bottom_layers,
    )
