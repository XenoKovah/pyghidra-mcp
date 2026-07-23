"""Read memory commands for pyghidra-mcp CLI."""

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
@click.argument("address")
@click.option("-s", "--size", type=int, default=32, help="Number of bytes to read.")
@click.pass_context
def read(ctx: click.Context, program_name: str, address: str, size: int) -> None:
    """Read bytes from memory at an address in a binary."""

    client = PyGhidraMcpClient(
        host=ctx.obj["HOST"],
        port=ctx.obj["PORT"],
    )

    async def run():
        async with client:
            result = await client.read_bytes(program_name, address, size=size)
            format_output(result, ctx.obj["OUTPUT_FORMAT"], ctx.obj["VERBOSE"])

    try:
        from ..utils import run_async

        run_async(run())
    except (asyncio.exceptions.CancelledError, Exception) as e:
        handle_command_error(e, ctx)
