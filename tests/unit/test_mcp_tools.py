import asyncio
from unittest.mock import Mock

import pytest

from pyghidra_mcp.gui_context import GuiPyGhidraContext
from pyghidra_mcp.mcp_ext_tools import (
    add_struct_field,
    delete_data_type,
    delete_function,
    delete_struct_field,
    goto,
    set_struct_field,
)
from pyghidra_mcp.mcp_tools import (
    apply_data_type,
    create_array_type,
    create_enum_type,
    create_function,
    create_pointer_type,
    create_struct_type,
    create_typedef,
    create_union_type,
    decompile_function,
    disassemble_function,
    get_enum_values,
    get_struct_layout,
    get_union_layout,
    import_c_types,
    list_callees,
    list_callers,
    list_programs,
    poll_script_job,
    rename_variable,
    run_inline_script,
    run_inline_script_async,
    run_script,
    run_script_async,
    search_data_items,
    search_data_types,
    search_functions,
    search_symbols,
    set_comment,
    set_function_prototype,
    set_variable_type,
)
from pyghidra_mcp.models import (
    DataItemInfo,
    DataTypeInfo,
    DataTypeKind,
    FunctionInfo,
    FunctionRef,
    ProgramInfo,
    StructField,
    SymbolInfo,
    SymbolKind,
)


def test_list_programs_uses_project_wide_context_listing():
    program_info = ProgramInfo(
        name="/folder/sample",
        file_path=None,
        load_time=None,
        analysis_complete=False,
        metadata={},
        code_indexed=False,
        strings_indexed=False,
    )
    pyghidra_context = Mock()
    pyghidra_context.list_program_infos.return_value = [program_info]

    ctx = Mock()
    ctx.request_context.lifespan_context = pyghidra_context

    response = list_programs(ctx)

    assert response.programs == [program_info]


def test_set_comment_uses_tool_path(monkeypatch):
    pyghidra_context = Mock()
    pyghidra_context.get_program_info.return_value = Mock()

    fake_tools = Mock()
    fake_tools.set_comment.return_value = {
        "address": "1000042e3",
        "comment": "function summary",
        "comment_type": "decompiler",
    }

    ctx = Mock()
    ctx.request_context.lifespan_context = pyghidra_context

    monkeypatch.setattr("pyghidra_mcp.mcp_tools.GhidraTools", lambda _program_info: fake_tools)

    response = set_comment(
        program_name="sample",
        target="entry",
        comment="function summary",
        comment_type="decompiler",
        ctx=ctx,
    )

    fake_tools.set_comment.assert_called_once_with("entry", "function summary", "decompiler")
    assert response.program_name == "sample"
    assert response.address == "1000042e3"
    assert response.comment_type == "decompiler"


def test_rename_variable_uses_tool_path(monkeypatch):
    pyghidra_context = Mock()
    pyghidra_context.get_program_info.return_value = Mock()

    fake_tools = Mock()
    fake_tools.rename_variable.return_value = {
        "function_name": "helper",
        "function_address": "100001000",
        "variable_kind": "parameter",
        "old_name": "count",
        "new_name": "item_count",
    }

    ctx = Mock()
    ctx.request_context.lifespan_context = pyghidra_context

    monkeypatch.setattr("pyghidra_mcp.mcp_tools.GhidraTools", lambda _program_info: fake_tools)

    response = rename_variable(
        program_name="sample",
        function_name_or_address="helper",
        variable_name="count",
        new_name="item_count",
        ctx=ctx,
    )

    fake_tools.rename_variable.assert_called_once_with("helper", "count", "item_count")
    assert response.program_name == "sample"
    assert response.function_name == "helper"
    assert response.function_address == "100001000"
    assert response.variable_kind == "parameter"
    assert response.old_name == "count"
    assert response.new_name == "item_count"


def test_set_variable_type_uses_tool_path(monkeypatch):
    pyghidra_context = Mock()
    pyghidra_context.get_program_info.return_value = Mock()

    fake_tools = Mock()
    fake_tools.set_variable_type.return_value = {
        "function_name": "helper",
        "function_address": "100001000",
        "variable_kind": "local",
        "variable_name": "total",
        "old_type": "int",
        "new_type": "long",
    }

    ctx = Mock()
    ctx.request_context.lifespan_context = pyghidra_context

    monkeypatch.setattr("pyghidra_mcp.mcp_tools.GhidraTools", lambda _program_info: fake_tools)

    response = set_variable_type(
        program_name="sample",
        function_name_or_address="helper",
        variable_name="total",
        type_name="long",
        ctx=ctx,
    )

    fake_tools.set_variable_type.assert_called_once_with("helper", "total", "long")
    assert response.program_name == "sample"
    assert response.function_name == "helper"
    assert response.function_address == "100001000"
    assert response.variable_kind == "local"
    assert response.variable_name == "total"
    assert response.old_type == "int"
    assert response.new_type == "long"


@pytest.mark.asyncio
async def test_decompile_function_offloads_with_timeout(monkeypatch):
    pyghidra_context = Mock()
    pyghidra_context.get_program_info.return_value = Mock()

    fake_tools = Mock()
    decompiled = Mock()
    decompiled.callees = None
    decompiled.referenced_strings = None
    decompiled.xrefs = None
    fake_tools.decompile_function_by_name_or_addr.return_value = decompiled

    ctx = Mock()
    ctx.request_context.lifespan_context = pyghidra_context

    monkeypatch.setattr("pyghidra_mcp.mcp_tools.GhidraTools", lambda _program_info: fake_tools)

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    response = await decompile_function(
        program_name="sample",
        name_or_address="entry",
        timeout_sec=17,
        ctx=ctx,
    )

    fake_tools.decompile_function_by_name_or_addr.assert_called_once_with("entry", timeout=17)
    assert response == [decompiled]


@pytest.mark.asyncio
async def test_decompile_does_not_block_other_tool_calls(monkeypatch):
    pyghidra_context = Mock()
    pyghidra_context.get_program_info.return_value = Mock()

    fake_tools = Mock()
    decompiled = Mock()
    fake_tools.search_symbols.return_value = [
        SymbolInfo(
            name="entry",
            address="1000",
            type="Function",
            namespace="Global",
            source="USER_DEFINED",
            refcount=1,
            external=False,
        )
    ]

    ctx = Mock()
    ctx.request_context.lifespan_context = pyghidra_context

    monkeypatch.setattr("pyghidra_mcp.mcp_tools.GhidraTools", lambda _program_info: fake_tools)

    decompile_started = asyncio.Event()
    release_decompile = asyncio.Event()

    async def fake_to_thread(fn, *args, **kwargs):
        decompile_started.set()
        await release_decompile.wait()
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    fake_tools.decompile_function_by_name_or_addr.return_value = decompiled

    decompile_task = asyncio.create_task(
        decompile_function(
            program_name="sample",
            name_or_address="entry",
            timeout_sec=30,
            ctx=ctx,
        )
    )

    await decompile_started.wait()

    symbols = search_symbols(
        program_name="sample",
        query="entry",
        ctx=ctx,
    )

    fake_tools.search_symbols.assert_called_once_with("entry", kinds=None, offset=0, limit=100)
    assert symbols.symbols[0].name == "entry"
    assert not decompile_task.done()

    release_decompile.set()
    response = await decompile_task
    assert response == [decompiled]


def _make_fake_tools_context(monkeypatch, fake_tools):
    """Shared scaffolding for wrapper tests: mock GhidraTools + return ctx.

    Patches the ``GhidraTools`` reference in both ``mcp_tools`` and
    ``mcp_ext_tools`` so wrapper tests for either module work uniformly.
    """
    pyghidra_context = Mock()
    pyghidra_context.get_program_info.return_value = Mock()
    ctx = Mock()
    ctx.request_context.lifespan_context = pyghidra_context
    monkeypatch.setattr("pyghidra_mcp.mcp_tools.GhidraTools", lambda _program_info: fake_tools)
    monkeypatch.setattr("pyghidra_mcp.mcp_ext_tools.GhidraTools", lambda _program_info: fake_tools)
    return ctx


def test_run_inline_script_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.run_inline_script.return_value = {
        "stdout": "hello\n",
        "stderr": "",
        "result_repr": "'hello'",
        "committed": True,
        "error": None,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = run_inline_script(
        program_name="sample",
        code="result = 'hello'\nprint(result)",
        ctx=ctx,
    )

    fake_tools.run_inline_script.assert_called_once_with(
        "result = 'hello'\nprint(result)", args=None
    )
    assert response.program_name == "sample"
    assert response.stdout == "hello\n"
    assert response.result_repr == "'hello'"
    assert response.committed is True
    assert response.error is None


def test_run_script_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.run_script.return_value = {
        "script": "helper.py",
        "resolved_path": "/home/user/ghidra_scripts/helper.py",
        "stdout": "hi\n",
        "stderr": "",
        "result_repr": None,
        "committed": True,
        "error": None,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = run_script(
        program_name="sample",
        script="helper.py",
        ctx=ctx,
    )

    fake_tools.run_script.assert_called_once_with("helper.py", args=None)
    assert response.program_name == "sample"
    assert response.script == "helper.py"
    assert response.resolved_path == "/home/user/ghidra_scripts/helper.py"
    assert response.stdout == "hi\n"
    assert response.committed is True


def test_run_script_passes_args(monkeypatch):
    fake_tools = Mock()
    fake_tools.run_script.return_value = {
        "script": "helper",
        "resolved_path": "/tmp/helper.py",
        "stdout": "",
        "stderr": "",
        "result_repr": None,
        "committed": True,
        "error": None,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    run_script(program_name="sample", script="helper", args=["a", "b"], ctx=ctx)

    fake_tools.run_script.assert_called_once_with("helper", args=["a", "b"])


def test_run_inline_script_passes_args_through(monkeypatch):
    fake_tools = Mock()
    fake_tools.run_inline_script.return_value = {
        "stdout": "",
        "stderr": "",
        "result_repr": None,
        "committed": True,
        "error": None,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    run_inline_script(program_name="sample", code="pass", args=["a", "b"], ctx=ctx)

    fake_tools.run_inline_script.assert_called_once_with("pass", args=["a", "b"])


def _stub_job(job_id="abc123", status="queued", submitted_at=1700000000.0):
    """Build a Mock that quacks like script_jobs.ScriptJob."""
    job = Mock()
    job.job_id = job_id
    job.status = status
    job.submitted_at = submitted_at
    return job


def test_run_inline_script_async_submits_job(monkeypatch):
    fake_tools = Mock()
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)
    pyghidra_context = ctx.request_context.lifespan_context
    pyghidra_context.script_jobs.submit.return_value = _stub_job(
        job_id="job-1", status="queued", submitted_at=1700000000.0
    )

    response = run_inline_script_async(
        program_name="sample",
        code="result = 1 + 1",
        args=["x"],
        ctx=ctx,
    )

    # The wrapper must hand the registry a (program_name, callable) pair.
    assert pyghidra_context.script_jobs.submit.call_count == 1
    call_args = pyghidra_context.script_jobs.submit.call_args
    assert call_args.args[0] == "sample"
    assert callable(call_args.args[1])

    assert response.program_name == "sample"
    assert response.job_id == "job-1"
    assert response.status == "queued"
    assert response.submitted_at == 1700000000.0


def test_run_script_async_submits_job(monkeypatch):
    fake_tools = Mock()
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)
    pyghidra_context = ctx.request_context.lifespan_context
    pyghidra_context.script_jobs.submit.return_value = _stub_job(
        job_id="job-2", status="running", submitted_at=1700000001.0
    )

    response = run_script_async(
        program_name="sample",
        script="rename_helpers.py",
        ctx=ctx,
    )

    assert pyghidra_context.script_jobs.submit.call_count == 1
    assert response.job_id == "job-2"
    assert response.status == "running"


def test_poll_script_job_returns_terminal_state(monkeypatch):
    ctx = _make_fake_tools_context(monkeypatch, Mock())
    pyghidra_context = ctx.request_context.lifespan_context
    pyghidra_context.script_jobs.get.return_value = Mock(
        to_dict=lambda: {
            "job_id": "job-1",
            "program_name": "sample",
            "status": "completed",
            "submitted_at": 1.0,
            "started_at": 2.0,
            "completed_at": 3.0,
            "stdout": "hello\n",
            "stderr": "",
            "result_repr": "'hello'",
            "committed": True,
            "error": None,
        }
    )

    response = poll_script_job(job_id="job-1", ctx=ctx)

    pyghidra_context.script_jobs.get.assert_called_once_with("job-1")
    assert response.status == "completed"
    assert response.stdout == "hello\n"
    assert response.committed is True
    assert response.error is None


def test_poll_script_job_unknown_id_raises(monkeypatch):
    from mcp.shared.exceptions import McpError

    ctx = _make_fake_tools_context(monkeypatch, Mock())
    ctx.request_context.lifespan_context.script_jobs.get.return_value = None

    with pytest.raises(McpError) as excinfo:
        poll_script_job(job_id="nope", ctx=ctx)
    assert "Unknown script job id" in str(excinfo.value)


def test_create_struct_type_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_struct_type.return_value = {"name": "MyStruct", "size": 8}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    fields = [StructField(name="a", type="int"), StructField(name="b", type="int")]
    response = create_struct_type(
        program_name="sample",
        name="MyStruct",
        fields=fields,
        ctx=ctx,
    )

    fake_tools.create_struct_type.assert_called_once_with("MyStruct", fields)
    assert response.program_name == "sample"
    assert response.name == "MyStruct"
    assert response.size == 8


def test_create_enum_type_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_enum_type.return_value = {"name": "Colors", "size": 4}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    values = {"RED": 0, "GREEN": 1, "BLUE": 2}
    response = create_enum_type(
        program_name="sample",
        name="Colors",
        values=values,
        ctx=ctx,
    )

    fake_tools.create_enum_type.assert_called_once_with("Colors", values, 4)
    assert response.program_name == "sample"
    assert response.name == "Colors"
    assert response.size == 4


def test_create_enum_type_passes_size(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_enum_type.return_value = {"name": "Short", "size": 2}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    create_enum_type(
        program_name="sample",
        name="Short",
        values={"A": 0},
        ctx=ctx,
        size=2,
    )

    fake_tools.create_enum_type.assert_called_once_with("Short", {"A": 0}, 2)


def test_create_union_type_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_union_type.return_value = {"name": "MyUnion", "size": 4}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    fields = [StructField(name="i", type="int"), StructField(name="f", type="float")]
    response = create_union_type(
        program_name="sample",
        name="MyUnion",
        fields=fields,
        ctx=ctx,
    )

    fake_tools.create_union_type.assert_called_once_with("MyUnion", fields)
    assert response.program_name == "sample"
    assert response.name == "MyUnion"
    assert response.size == 4


def test_create_array_type_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_array_type.return_value = {"name": "IntArr10", "size": 40}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = create_array_type(
        program_name="sample",
        name="IntArr10",
        base_type="int",
        length=10,
        ctx=ctx,
    )

    fake_tools.create_array_type.assert_called_once_with("IntArr10", "int", 10)
    assert response.program_name == "sample"
    assert response.name == "IntArr10"
    assert response.size == 40


def test_create_pointer_type_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_pointer_type.return_value = {"name": "IntPtr", "size": 8}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = create_pointer_type(
        program_name="sample",
        name="IntPtr",
        base_type="int",
        ctx=ctx,
    )

    fake_tools.create_pointer_type.assert_called_once_with("IntPtr", "int")
    assert response.program_name == "sample"
    assert response.name == "IntPtr"
    assert response.size == 8


def test_create_typedef_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_typedef.return_value = {"name": "MyInt", "size": 4}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = create_typedef(
        program_name="sample",
        name="MyInt",
        base_type="int",
        ctx=ctx,
    )

    fake_tools.create_typedef.assert_called_once_with("MyInt", "int")
    assert response.program_name == "sample"
    assert response.name == "MyInt"
    assert response.size == 4


def test_apply_data_type_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.apply_data_type.return_value = {"address": "0x1000", "applied_size": 4}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = apply_data_type(
        program_name="sample",
        address="0x1000",
        type_name="int",
        ctx=ctx,
    )

    # clear_existing defaults to True
    fake_tools.apply_data_type.assert_called_once_with("0x1000", "int", True)
    assert response.program_name == "sample"
    assert response.address == "0x1000"
    assert response.applied_size == 4


def test_apply_data_type_passes_clear_existing_false(monkeypatch):
    fake_tools = Mock()
    fake_tools.apply_data_type.return_value = {"address": "0x2000", "applied_size": 8}
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    apply_data_type(
        program_name="sample",
        address="0x2000",
        type_name="long",
        ctx=ctx,
        clear_existing=False,
    )

    fake_tools.apply_data_type.assert_called_once_with("0x2000", "long", False)


def test_search_data_types_defaults(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_data_types.return_value = [
        DataTypeInfo(name="MyStruct", kind="struct", size=8, path="/MyStruct"),
    ]
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = search_data_types(program_name="sample", ctx=ctx)

    fake_tools.search_data_types.assert_called_once_with(
        query=".*", kinds=None, category=None, offset=0, limit=100
    )
    assert response.data_types[0].name == "MyStruct"
    assert response.data_types[0].kind == "struct"


def test_search_data_types_with_kinds_and_query(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_data_types.return_value = []
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    search_data_types(
        program_name="sample",
        query="^Msg",
        kinds=[DataTypeKind.STRUCT, DataTypeKind.UNION],
        offset=10,
        limit=25,
        ctx=ctx,
    )

    fake_tools.search_data_types.assert_called_once_with(
        query="^Msg",
        kinds=[DataTypeKind.STRUCT, DataTypeKind.UNION],
        category=None,
        offset=10,
        limit=25,
    )


def test_search_functions_defaults(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_functions.return_value = [
        FunctionInfo(
            name="main",
            address="0x1000",
            refcount=2,
            is_thunk=False,
            is_external=False,
            is_user_defined=True,
        ),
    ]
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = search_functions(program_name="sample", ctx=ctx)

    fake_tools.search_functions.assert_called_once_with(
        query=".*",
        min_refcount=None,
        max_refcount=None,
        user_defined_only=False,
        include_thunks=True,
        include_externals=True,
        offset=0,
        limit=100,
    )
    assert response.functions[0].name == "main"
    assert response.functions[0].refcount == 2
    assert response.functions[0].is_user_defined is True


def test_search_functions_with_filters(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_functions.return_value = []
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    search_functions(
        program_name="sample",
        query="^init_",
        min_refcount=5,
        max_refcount=100,
        user_defined_only=True,
        include_thunks=False,
        include_externals=False,
        offset=10,
        limit=25,
        ctx=ctx,
    )

    fake_tools.search_functions.assert_called_once_with(
        query="^init_",
        min_refcount=5,
        max_refcount=100,
        user_defined_only=True,
        include_thunks=False,
        include_externals=False,
        offset=10,
        limit=25,
    )


def test_create_function_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_function.return_value = {
        "name": "FUN_00001000",
        "entry_point": "0x00001000",
        "body_size": 32,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = create_function(
        program_name="sample",
        address="0x1000",
        ctx=ctx,
    )

    fake_tools.create_function.assert_called_once_with("0x1000", name=None, disassemble_first=True)
    assert response.program_name == "sample"
    assert response.name == "FUN_00001000"
    assert response.entry_point == "0x00001000"
    assert response.body_size == 32


def test_create_function_passes_name_and_disassemble_flag(monkeypatch):
    fake_tools = Mock()
    fake_tools.create_function.return_value = {
        "name": "init_handler",
        "entry_point": "0x00001000",
        "body_size": 64,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    create_function(
        program_name="sample",
        address="0x1000",
        name="init_handler",
        disassemble_first=False,
        ctx=ctx,
    )

    fake_tools.create_function.assert_called_once_with(
        "0x1000", name="init_handler", disassemble_first=False
    )


def test_delete_function_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.delete_function.return_value = {
        "name": "stale_helper",
        "entry_point": "0x00002000",
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = delete_function(
        program_name="sample",
        name_or_address="stale_helper",
        ctx=ctx,
    )

    fake_tools.delete_function.assert_called_once_with("stale_helper")
    assert response.program_name == "sample"
    assert response.name == "stale_helper"
    assert response.entry_point == "0x00002000"


def test_set_function_prototype_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.set_function_prototype.return_value = {
        "name": "handle_request",
        "entry_point": "0x00001000",
        "signature": "int handle_request(Request * req, size_t len)",
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = set_function_prototype(
        program_name="sample",
        name_or_address="handle_request",
        prototype="int handle_request(Request *req, size_t len)",
        ctx=ctx,
    )

    fake_tools.set_function_prototype.assert_called_once_with(
        "handle_request",
        "int handle_request(Request *req, size_t len)",
        calling_convention=None,
    )
    assert response.program_name == "sample"
    assert response.name == "handle_request"
    assert response.entry_point == "0x00001000"
    assert response.signature == "int handle_request(Request * req, size_t len)"


def test_delete_data_type_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.delete_data_type.return_value = {
        "name": "Foo",
        "path": "/MyLib/Foo",
        "kind": "struct",
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = delete_data_type(
        program_name="sample",
        name_or_path="/MyLib/Foo",
        ctx=ctx,
    )

    fake_tools.delete_data_type.assert_called_once_with("/MyLib/Foo")
    assert response.program_name == "sample"
    assert response.name == "Foo"
    assert response.path == "/MyLib/Foo"
    assert response.kind == "struct"


def test_import_c_types_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.import_c_types.return_value = {
        "imported": [
            {"name": "Point", "kind": "struct", "size": 8, "path": "/Point"},
            {"name": "color_t", "kind": "typedef", "size": 4, "path": "/color_t"},
        ],
        "skipped": [],
        "errors": [],
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = import_c_types(
        program_name="sample",
        source="struct Point { int x; int y; }; typedef int color_t;",
        ctx=ctx,
    )

    fake_tools.import_c_types.assert_called_once_with(
        "struct Point { int x; int y; }; typedef int color_t;",
        category_path=None,
        replace=False,
    )
    assert response.program_name == "sample"
    assert [dt.name for dt in response.imported] == ["Point", "color_t"]
    assert response.imported[0].kind == "struct"
    assert response.imported[1].kind == "typedef"
    assert response.skipped == []
    assert response.errors == []


def test_import_c_types_passes_category_and_replace(monkeypatch):
    fake_tools = Mock()
    fake_tools.import_c_types.return_value = {
        "imported": [{"name": "Foo", "kind": "struct", "size": 4, "path": "/MyLib/Foo"}],
        "skipped": ["/MyLib/Bar"],
        "errors": ["warning: redefinition of 'baz'"],
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = import_c_types(
        program_name="sample",
        source="struct Foo { int x; };",
        category_path="/MyLib",
        replace=True,
        ctx=ctx,
    )

    fake_tools.import_c_types.assert_called_once_with(
        "struct Foo { int x; };",
        category_path="/MyLib",
        replace=True,
    )
    assert response.imported[0].path == "/MyLib/Foo"
    assert response.skipped == ["/MyLib/Bar"]
    assert response.errors == ["warning: redefinition of 'baz'"]


def test_set_function_prototype_passes_calling_convention(monkeypatch):
    fake_tools = Mock()
    fake_tools.set_function_prototype.return_value = {
        "name": "win32_entry",
        "entry_point": "0x00401000",
        "signature": "void __stdcall win32_entry(void)",
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    set_function_prototype(
        program_name="sample",
        name_or_address="0x00401000",
        prototype="void win32_entry(void)",
        calling_convention="__stdcall",
        ctx=ctx,
    )

    fake_tools.set_function_prototype.assert_called_once_with(
        "0x00401000",
        "void win32_entry(void)",
        calling_convention="__stdcall",
    )


def test_get_struct_layout_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.get_struct_layout.return_value = {
        "name": "Point",
        "path": "/MyLib/Point",
        "size": 12,
        "fields": [
            {"offset": 0, "size": 4, "type": "int", "name": "x"},
            {"offset": 4, "size": 4, "type": "int", "name": "y"},
            {"offset": 8, "size": 4, "type": "float", "name": "z"},
        ],
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = get_struct_layout(
        program_name="sample",
        name_or_path="/MyLib/Point",
        ctx=ctx,
    )

    fake_tools.get_struct_layout.assert_called_once_with("/MyLib/Point")
    assert response.program_name == "sample"
    assert response.name == "Point"
    assert response.path == "/MyLib/Point"
    assert response.size == 12
    assert [f.name for f in response.fields] == ["x", "y", "z"]
    assert response.fields[0].offset == 0
    assert response.fields[1].offset == 4


def test_get_union_layout_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.get_union_layout.return_value = {
        "name": "Variant",
        "path": "/Variant",
        "size": 8,
        "fields": [
            {"size": 4, "type": "int", "name": "i"},
            {"size": 8, "type": "double", "name": "d"},
        ],
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = get_union_layout(
        program_name="sample",
        name_or_path="Variant",
        ctx=ctx,
    )

    fake_tools.get_union_layout.assert_called_once_with("Variant")
    assert response.name == "Variant"
    assert response.size == 8
    assert [f.name for f in response.fields] == ["i", "d"]
    # Sanity: union fields don't expose offset
    assert not hasattr(response.fields[0], "offset")


def test_get_enum_values_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.get_enum_values.return_value = {
        "name": "Color",
        "path": "/Color",
        "size": 4,
        "values": {"RED": 0, "GREEN": 1, "BLUE": 2},
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = get_enum_values(
        program_name="sample",
        name_or_path="Color",
        ctx=ctx,
    )

    fake_tools.get_enum_values.assert_called_once_with("Color")
    assert response.values == {"RED": 0, "GREEN": 1, "BLUE": 2}
    assert response.size == 4


def test_search_data_items_defaults(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_data_items.return_value = [
        DataItemInfo(
            name="login_msg",
            address="0x1000",
            type="char[20]",
            length=20,
            refcount=3,
        ),
    ]
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = search_data_items(program_name="sample", ctx=ctx)

    fake_tools.search_data_items.assert_called_once_with(
        query=".*",
        min_refcount=None,
        max_refcount=None,
        offset=0,
        limit=100,
    )
    assert response.data_items[0].name == "login_msg"
    assert response.data_items[0].refcount == 3
    assert response.data_items[0].type == "char[20]"


def test_search_data_items_with_filters(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_data_items.return_value = []
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    search_data_items(
        program_name="sample",
        query="^g_",
        min_refcount=10,
        max_refcount=1000,
        offset=20,
        limit=50,
        ctx=ctx,
    )

    fake_tools.search_data_items.assert_called_once_with(
        query="^g_",
        min_refcount=10,
        max_refcount=1000,
        offset=20,
        limit=50,
    )


def test_search_data_types_with_category_filter(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_data_types.return_value = []
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    search_data_types(
        program_name="sample",
        query="Point",
        category="/MyLib",
        ctx=ctx,
    )

    fake_tools.search_data_types.assert_called_once_with(
        query="Point",
        kinds=None,
        category="/MyLib",
        offset=0,
        limit=100,
    )


@pytest.mark.asyncio
async def test_disassemble_function_single_target(monkeypatch):
    fake_tools = Mock()
    fake_tools.disassemble_function.return_value = {
        "name": "main",
        "entry": "0x1000",
        "signature": "int main()",
        "listing": "0x1000: PUSH EBP\n0x1001: MOV EBP,ESP",
        "instruction_count": 2,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    response = await disassemble_function(
        program_name="sample",
        name_or_address="main",
        ctx=ctx,
    )

    fake_tools.disassemble_function.assert_called_once_with("main")
    assert len(response) == 1
    assert response[0].name == "main"
    assert response[0].entry == "0x1000"
    assert response[0].instruction_count == 2
    assert "PUSH EBP" in response[0].listing
    assert response[0].error is None


@pytest.mark.asyncio
async def test_disassemble_function_batch(monkeypatch):
    fake_tools = Mock()
    fake_tools.disassemble_function.side_effect = [
        {
            "name": "func_a",
            "entry": "0x1000",
            "signature": None,
            "listing": "0x1000: NOP",
            "instruction_count": 1,
        },
        {
            "name": "func_b",
            "entry": "0x2000",
            "signature": None,
            "listing": "0x2000: RET",
            "instruction_count": 1,
        },
    ]
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    response = await disassemble_function(
        program_name="sample",
        name_or_address=["func_a", "func_b"],
        ctx=ctx,
    )

    assert [r.name for r in response] == ["func_a", "func_b"]
    assert [r.entry for r in response] == ["0x1000", "0x2000"]
    assert fake_tools.disassemble_function.call_count == 2


@pytest.mark.asyncio
async def test_disassemble_function_reports_error_per_target(monkeypatch):
    fake_tools = Mock()
    fake_tools.disassemble_function.side_effect = ValueError("Function not found")
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    async def fake_to_thread(fn, *args, **kwargs):
        return fn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    response = await disassemble_function(
        program_name="sample",
        name_or_address="missing",
        ctx=ctx,
    )

    assert len(response) == 1
    assert response[0].name == "missing"
    assert response[0].error == "Function not found"
    assert response[0].listing == ""
    assert response[0].instruction_count == 0


def test_list_callees_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.list_callees.return_value = [
        FunctionRef(name="helper_a", address="0x1100"),
        FunctionRef(name="helper_b", address="0x1200"),
    ]
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = list_callees(
        program_name="sample",
        name_or_address="main",
        ctx=ctx,
    )

    fake_tools.list_callees.assert_called_once_with("main", offset=0, limit=100)
    assert [f.name for f in response.functions] == ["helper_a", "helper_b"]


def test_list_callees_passes_pagination(monkeypatch):
    fake_tools = Mock()
    fake_tools.list_callees.return_value = []
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    list_callees(
        program_name="sample",
        name_or_address="main",
        ctx=ctx,
        offset=25,
        limit=10,
    )

    fake_tools.list_callees.assert_called_once_with("main", offset=25, limit=10)


def test_list_callers_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.list_callers.return_value = [
        FunctionRef(name="caller_one", address="0x2000"),
    ]
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = list_callers(
        program_name="sample",
        name_or_address="entry",
        ctx=ctx,
    )

    fake_tools.list_callers.assert_called_once_with("entry", offset=0, limit=100)
    assert len(response.functions) == 1
    assert response.functions[0].name == "caller_one"
    assert response.functions[0].address == "0x2000"


def test_search_symbols_with_kinds_filter(monkeypatch):
    fake_tools = Mock()
    fake_tools.search_symbols.return_value = [
        SymbolInfo(
            name="main",
            address="1000",
            type="Function",
            namespace="Global",
            source="USER_DEFINED",
            refcount=1,
            external=False,
        )
    ]
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = search_symbols(
        program_name="sample",
        query="^main$",
        kinds=[SymbolKind.FUNCTIONS],
        ctx=ctx,
    )

    fake_tools.search_symbols.assert_called_once_with(
        "^main$", kinds=[SymbolKind.FUNCTIONS], offset=0, limit=100
    )
    assert response.symbols[0].name == "main"


def test_add_struct_field_uses_tool_path(monkeypatch):
    fake_tools = Mock()
    fake_tools.add_struct_field.return_value = {
        "struct_name": "Point",
        "offset": 12,
        "struct_size": 16,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = add_struct_field(
        program_name="sample",
        struct_name_or_path="Point",
        field_name="w",
        field_type="int",
        ctx=ctx,
    )

    fake_tools.add_struct_field.assert_called_once_with("Point", "w", "int", offset=None)
    assert response.program_name == "sample"
    assert response.struct_name == "Point"
    assert response.offset == 12
    assert response.struct_size == 16


def test_add_struct_field_with_explicit_offset(monkeypatch):
    fake_tools = Mock()
    fake_tools.add_struct_field.return_value = {
        "struct_name": "Point",
        "offset": 16,
        "struct_size": 20,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    add_struct_field(
        program_name="sample",
        struct_name_or_path="Point",
        field_name="extra",
        field_type="int",
        offset=16,
        ctx=ctx,
    )

    fake_tools.add_struct_field.assert_called_once_with("Point", "extra", "int", offset=16)


def test_set_struct_field_by_name(monkeypatch):
    fake_tools = Mock()
    fake_tools.set_struct_field.return_value = {
        "struct_name": "Point",
        "offset": 0,
        "struct_size": 12,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = set_struct_field(
        program_name="sample",
        struct_name_or_path="Point",
        field_name="x",
        new_name="x_pos",
        new_type="long",
        ctx=ctx,
    )

    fake_tools.set_struct_field.assert_called_once_with(
        "Point",
        field_name="x",
        field_offset=None,
        new_name="x_pos",
        new_type="long",
    )
    assert response.struct_name == "Point"
    assert response.offset == 0


def test_set_struct_field_by_offset(monkeypatch):
    fake_tools = Mock()
    fake_tools.set_struct_field.return_value = {
        "struct_name": "Point",
        "offset": 8,
        "struct_size": 12,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    set_struct_field(
        program_name="sample",
        struct_name_or_path="Point",
        field_offset=8,
        new_name="z",
        ctx=ctx,
    )

    fake_tools.set_struct_field.assert_called_once_with(
        "Point",
        field_name=None,
        field_offset=8,
        new_name="z",
        new_type=None,
    )


def test_delete_struct_field_by_name(monkeypatch):
    fake_tools = Mock()
    fake_tools.delete_struct_field.return_value = {
        "struct_name": "Point",
        "offset": 8,
        "struct_size": 8,
    }
    ctx = _make_fake_tools_context(monkeypatch, fake_tools)

    response = delete_struct_field(
        program_name="sample",
        struct_name_or_path="Point",
        field_name="z",
        ctx=ctx,
    )

    fake_tools.delete_struct_field.assert_called_once_with(
        "Point", field_name="z", field_offset=None
    )
    assert response.offset == 8
    assert response.struct_size == 8


def test_goto_uses_gui_context():
    gui_context = GuiPyGhidraContext.__new__(GuiPyGhidraContext)
    gui_context.goto = Mock()
    gui_context.goto.return_value = {
        "program_name": "sample",
        "address": "1000042e3",
        "success": True,
    }

    ctx = Mock()
    ctx.request_context.lifespan_context = gui_context

    response = goto(
        program_name="sample",
        target="entry",
        target_type="function",
        ctx=ctx,
    )

    gui_context.goto.assert_called_once_with("sample", "entry", "function")
    assert response.program_name == "sample"
    assert response.address == "1000042e3"
    assert response.success is True
