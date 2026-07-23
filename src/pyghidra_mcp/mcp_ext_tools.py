"""
Extended MCP tool handlers for pyghidra-mcp.

Project- and GUI-management tools split out from mcp_tools. Registered only
when the server is launched with --gui.
"""

import logging
from typing import Literal, cast

from mcp.server.fastmcp import Context

from pyghidra_mcp.context_protocol import MCPContext
from pyghidra_mcp.mcp_tools import _run_for_context, mcp_error_handler
from pyghidra_mcp.models import (
    AddStructFieldResponse,
    DeleteDataTypeResponse,
    DeleteFunctionResponse,
    DeleteStructFieldResponse,
    GotoResponse,
    ImportBinaryResponse,
    ListOpenProgramsResponse,
    OpenProgramInfo,
    SetStructFieldResponse,
)
from pyghidra_mcp.tools import GhidraTools

logger = logging.getLogger(__name__)


def _require_gui_context(ctx: Context):
    from pyghidra_mcp.gui_context import GuiPyGhidraContext

    pyghidra_context = ctx.request_context.lifespan_context
    if not isinstance(pyghidra_context, GuiPyGhidraContext):
        raise ValueError("This tool requires pyghidra-mcp to be running with --gui")
    return pyghidra_context


@mcp_error_handler
def get_program_metadata(program_name: str, ctx: Context) -> dict:
    """Get a program's metadata: architecture, compiler, endianness, hashes, analysis counts."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    return program_info.metadata


@mcp_error_handler
def import_binary(binary_path: str, ctx: Context) -> ImportBinaryResponse:
    """Import a binary into the project from a file path."""
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    return pyghidra_context.import_binary_backgrounded(binary_path)


@mcp_error_handler
def list_open_programs(ctx: Context) -> ListOpenProgramsResponse:
    """List programs currently open in the Ghidra GUI."""
    gui_context = _require_gui_context(ctx)
    programs = [OpenProgramInfo(**info) for info in gui_context.list_open_programs()]
    return ListOpenProgramsResponse(programs=programs)


@mcp_error_handler
def open_program_in_gui(program_name: str, ctx: Context) -> OpenProgramInfo:
    """Open a project binary in the Ghidra GUI CodeBrowser."""
    gui_context = _require_gui_context(ctx)
    return OpenProgramInfo(**gui_context.open_program_in_gui(program_name))


@mcp_error_handler
def set_current_program(program_name: str, ctx: Context) -> OpenProgramInfo:
    """Set the active/current program in the Ghidra GUI CodeBrowser."""
    gui_context = _require_gui_context(ctx)
    return OpenProgramInfo(**gui_context.set_current_program(program_name))


@mcp_error_handler
def goto(
    program_name: str,
    target: str,
    target_type: Literal["address", "function"],
    ctx: Context,
) -> GotoResponse:
    """Navigate the Ghidra GUI CodeBrowser to an address or function."""
    gui_context = _require_gui_context(ctx)
    return GotoResponse(**gui_context.goto(program_name, target, target_type))


@mcp_error_handler
def delete_function(
    program_name: str,
    name_or_address: str,
    ctx: Context,
) -> DeleteFunctionResponse:
    """Delete the function identified by ``name_or_address``.

    Destructive — this tool is only registered with ``--gui``, so it's
    intended for human-driven workflows where the operator can confirm
    against the GUI listing before issuing the delete.

    Removes the function via ``FunctionManager.removeFunction`` inside a
    Ghidra transaction. Custom signatures, comments, parameter names, and
    any other attached metadata are lost. Errors with INVALID_PARAMS if no
    function matches the given name or address.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.delete_function(name_or_address),
    )
    result = cast(dict, result)
    return DeleteFunctionResponse(program_name=program_name, **result)


@mcp_error_handler
def delete_data_type(
    program_name: str,
    name_or_path: str,
    ctx: Context,
) -> DeleteDataTypeResponse:
    """Delete a data type from the program's data type manager.

    Destructive — registered only when the server is launched with ``--gui``,
    so the operator can review the DTM before issuing the delete.

    ``name_or_path`` accepts a full DTM path (``"/MyLib/Foo"``) or a bare
    name (``"Foo"``). Errors with INVALID_PARAMS if the type is not found
    or if Ghidra refuses to remove it (typically because the type is still
    referenced by code/data — clear those references first).
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.delete_data_type(name_or_path),
    )
    result = cast(dict, result)
    return DeleteDataTypeResponse(program_name=program_name, **result)


@mcp_error_handler
def add_struct_field(
    program_name: str,
    struct_name_or_path: str,
    field_name: str,
    field_type: str,
    ctx: Context,
    offset: int | None = None,
) -> AddStructFieldResponse:
    """Add a field to an existing struct.

    Refinement tool — intended for human-driven workflows after the agent's
    first-pass struct creation. ``struct_name_or_path`` accepts a bare name
    (``"Point"``) or a full DTM path (``"/MyLib/Point"``).

    ``offset=null`` (default) appends the field at the end. A specific
    integer overlays the field into existing padding (or grows the struct
    if needed).

    ``field_type`` is parsed via Ghidra's DataTypeParser, so pointer/array
    syntax is supported (``"int*"``, ``"char[16]"``).

    Returns the affected field's offset and the struct's new total size.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.add_struct_field(struct_name_or_path, field_name, field_type, offset=offset),
    )
    result = cast(dict, result)
    return AddStructFieldResponse(program_name=program_name, **result)


@mcp_error_handler
def set_struct_field(
    program_name: str,
    struct_name_or_path: str,
    ctx: Context,
    field_name: str | None = None,
    field_offset: int | None = None,
    new_name: str | None = None,
    new_type: str | None = None,
) -> SetStructFieldResponse:
    """Set the name and/or type of an existing struct field.

    Refinement tool. Identify the target field by EITHER ``field_name`` OR
    ``field_offset`` (exactly one — passing both or neither errors with
    INVALID_PARAMS). Use ``field_offset`` for unnamed/padding fields.

    At least one of ``new_name`` / ``new_type`` must be provided; a no-op
    call errors. ``new_type`` follows Ghidra's DataTypeParser syntax.

    Returns the modified field's offset and the struct's total size.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.set_struct_field(
            struct_name_or_path,
            field_name=field_name,
            field_offset=field_offset,
            new_name=new_name,
            new_type=new_type,
        ),
    )
    result = cast(dict, result)
    return SetStructFieldResponse(program_name=program_name, **result)


@mcp_error_handler
def delete_struct_field(
    program_name: str,
    struct_name_or_path: str,
    ctx: Context,
    field_name: str | None = None,
    field_offset: int | None = None,
) -> DeleteStructFieldResponse:
    """Delete a field from a struct.

    Destructive refinement tool. Identify the target via EITHER
    ``field_name`` OR ``field_offset`` (exactly one). The struct's total
    size shrinks accordingly.

    Returns the removed field's prior offset and the struct's new total size.
    """
    pyghidra_context: MCPContext = ctx.request_context.lifespan_context
    program_info = pyghidra_context.get_program_info(program_name)
    tools = GhidraTools(program_info)
    result = _run_for_context(
        pyghidra_context,
        lambda: tools.delete_struct_field(
            struct_name_or_path,
            field_name=field_name,
            field_offset=field_offset,
        ),
    )
    result = cast(dict, result)
    return DeleteStructFieldResponse(program_name=program_name, **result)
