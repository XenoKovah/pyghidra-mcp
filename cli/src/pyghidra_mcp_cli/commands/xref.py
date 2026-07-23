"""Cross-reference commands for pyghidra-mcp CLI."""

import asyncio

import click

from ..client import PyGhidraMcpClient
from ..utils import format_output, handle_command_error


def program_option(func):
    """Common --program option for commands that target a specific program."""
    return click.option(
        "-b",
        "--program",
        "program_name",
        required=True,
        help="Program name in the project (use 'list programs' to see available programs).",
    )(func)


@click.command()
@program_option
@click.argument("name_or_address")
@click.pass_context
def xref(ctx: click.Context, program_name: str, name_or_address: str) -> None:
    """List cross-references to a symbol or address in a binary."""

    client = PyGhidraMcpClient(
        host=ctx.obj["HOST"],
        port=ctx.obj["PORT"],
    )

    async def run():
        async with client:
            result = await client.list_xrefs(program_name, name_or_address)
            format_output(result, ctx.obj["OUTPUT_FORMAT"], ctx.obj["VERBOSE"])

    try:
        from ..utils import run_async

        run_async(run())
    except (asyncio.exceptions.CancelledError, Exception) as e:
        handle_command_error(e, ctx)
