"""Port and config tests: the demo must never take port 8000."""

from __future__ import annotations

import pytest

from mototrack import config
from mototrack.cli import port_in_use, suggest_ports


def test_default_port_is_not_8000():
    assert config.DEFAULT_PORT != 8000
    assert config.DEFAULT_PORT == 8770


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv(config.PORT_ENV_VAR, "9123")
    assert config.resolve_port() == 9123


def test_cli_flag_beats_env_var(monkeypatch):
    monkeypatch.setenv(config.PORT_ENV_VAR, "9123")
    assert config.resolve_port(9555) == 9555


def test_bad_env_var_is_a_clear_error(monkeypatch):
    monkeypatch.setenv(config.PORT_ENV_VAR, "not-a-port")
    with pytest.raises(ValueError, match=config.PORT_ENV_VAR):
        config.resolve_port()


def test_busy_port_is_detected():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        port = sock.getsockname()[1]
        assert port_in_use("127.0.0.1", port)


def test_suggest_ports_returns_free_ones():
    free = suggest_ports("127.0.0.1", 49152, count=3)
    assert len(free) == 3
    assert not any(port_in_use("127.0.0.1", port) for port in free)
