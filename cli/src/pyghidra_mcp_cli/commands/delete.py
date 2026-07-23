"""Delete binary commands for pyghidra-mcp CLI."""

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
@click.pass_context
def delete(ctx: click.Context, program_name: str) -> None:
    """Delete a binary from the project."""

    client = PyGhidraMcpClient(
        host=ctx.obj["HOST"],
        port=ctx.obj["PORT"],
    )

    async def run():
        async with client:
            result = await client.delete_binary(program_name)
            format_output(result, ctx.obj["OUTPUT_FORMAT"], ctx.obj["VERBOSE"])

    try:
        from ..utils import run_async

        run_async(run())
    except (asyncio.exceptions.CancelledError, Exception) as e:
        handle_command_error(e, ctx)
