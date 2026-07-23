from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from .context import ProgramInfo
    from .models import ImportBinaryResponse, ProgramInfo as ProgramInfoModel
    from .script_jobs import ScriptJobRegistry


class MCPContext(Protocol):
    """Tool-facing context contract shared by headless and GUI modes."""

    programs: dict[str, "ProgramInfo"]
    script_jobs: "ScriptJobRegistry"

    def get_program_info(self, program_name: str) -> "ProgramInfo": ...

    def list_binaries(self) -> list[str]: ...

    def list_binary_domain_files(self) -> list[Any]: ...

    def list_program_infos(self) -> list["ProgramInfoModel"]: ...

    def delete_program(self, program_name: str) -> bool: ...

    def import_binary_backgrounded(self, binary_path: str | Path) -> "ImportBinaryResponse": ...

    def close(self, save: bool = True) -> None: ...
