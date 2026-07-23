"""Unit tests for tools._resolve_script_path."""

from pathlib import Path

from pyghidra_mcp.tools import _resolve_script_path


def test_resolve_absolute_path(tmp_path: Path) -> None:
    """An absolute path to an existing file resolves to itself."""
    script = tmp_path / "direct.py"
    script.write_text("result = 1\n")

    resolved = _resolve_script_path(str(script))

    assert resolved == script.resolve()


def test_returns_none_when_missing(tmp_path: Path) -> None:
    """Unknown script paths resolve to None (not raise)."""
    missing = tmp_path / "nope.py"

    assert _resolve_script_path(str(missing)) is None


def test_cwd_ghidra_scripts_bare_name(tmp_path: Path, monkeypatch) -> None:
    """A bare filename finds files under ``./ghidra_scripts``."""
    scripts_dir = tmp_path / "ghidra_scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "foo.py"
    script.write_text("result = 1\n")
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_script_path("foo.py")

    assert resolved == script.resolve()


def test_extension_fallback_appends_py(tmp_path: Path, monkeypatch) -> None:
    """When the caller omits ``.py``, the resolver appends it."""
    scripts_dir = tmp_path / "ghidra_scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "bar.py"
    script.write_text("result = 1\n")
    monkeypatch.chdir(tmp_path)

    resolved = _resolve_script_path("bar")

    assert resolved == script.resolve()


def test_home_ghidra_scripts_takes_over_when_cwd_missing(tmp_path: Path, monkeypatch) -> None:
    """If ./ghidra_scripts doesn't have the file, ~/ghidra_scripts is consulted."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    home_scripts = fake_home / "ghidra_scripts"
    home_scripts.mkdir()
    script = home_scripts / "hub.py"
    script.write_text("result = 1\n")

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("HOME", str(fake_home))

    resolved = _resolve_script_path("hub.py")

    assert resolved == script.resolve()


def test_basename_fallback(tmp_path: Path, monkeypatch) -> None:
    """A user-supplied path whose basename exists under ghidra_scripts resolves."""
    scripts_dir = tmp_path / "ghidra_scripts"
    scripts_dir.mkdir()
    script = scripts_dir / "zap.py"
    script.write_text("result = 1\n")
    monkeypatch.chdir(tmp_path)

    # Caller supplies a non-existent full path; only the basename lives under
    # ./ghidra_scripts/. The resolver should pick it up via basename fallback.
    resolved = _resolve_script_path("/does/not/exist/zap.py")

    assert resolved == script.resolve()
