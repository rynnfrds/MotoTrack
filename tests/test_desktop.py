"""Desktop launcher tests: no GUI is opened, only the wiring is checked."""

from __future__ import annotations

import inspect

import pytest

from mototrack import cli, config


def test_desktop_command_is_registered():
    parser = cli.build_parser()
    args = parser.parse_args(["desktop"])
    assert args.func is cli.cmd_desktop


def test_desktop_accepts_a_port_override():
    parser = cli.build_parser()
    args = parser.parse_args(["desktop", "--port", "8799"])
    assert args.port == 8799


def test_desktop_defaults_to_no_explicit_port():
    args = cli.build_parser().parse_args(["desktop"])
    assert args.port is None, "port should fall through to config.resolve_port"


def test_desktop_missing_pywebview_is_a_clear_error(monkeypatch, capsys):
    """A missing optional dependency must explain itself, not traceback."""

    real_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "webview":
            raise ImportError("simulated missing pywebview")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    code = cli.main(["desktop"])
    assert code == 2
    message = capsys.readouterr().err
    assert "pywebview is not installed" in message
    assert "uv pip install pywebview" in message


def test_desktop_never_touches_port_8000(monkeypatch):
    """The desktop window must not collide with the other local application."""
    source = inspect.getsource(cli.cmd_desktop)
    assert "8000" not in source


def test_desktop_uses_loopback_only():
    source = inspect.getsource(cli.cmd_desktop)
    assert '127.0.0.1' in source, "the desktop window should not bind a public interface"


def test_launcher_script_exists():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    launcher = os.path.join(root, "Start MotoTrack.bat")
    assert os.path.exists(launcher)
    body = open(launcher, encoding="utf-8", errors="replace").read()
    assert "mototrack.cli desktop" in body
    assert "8000" not in body