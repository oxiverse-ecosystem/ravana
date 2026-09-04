import socket

import pytest

from ravana.agent import tool_registry


def _resolved(ip):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]


@pytest.mark.parametrize("ip", [
    "127.0.0.1",
    "10.0.0.1",
    "169.254.1.1",
    "240.0.0.1",
    "::1",
    "fc00::1",
    "fe80::1",
])
def test_read_website_rejects_non_public_destinations(monkeypatch, ip):
    monkeypatch.setattr(
        tool_registry.socket, "getaddrinfo", lambda *args, **kwargs: _resolved(ip))

    def _unexpected_opener(*args, **kwargs):
        pytest.fail("network opener was built before destination validation")

    monkeypatch.setattr(tool_registry.urllib.request, "build_opener", _unexpected_opener)
    with pytest.raises(PermissionError):
        tool_registry._read_website("http://example.test/")


def test_redirect_handler_validates_redirect_destination(monkeypatch):
    monkeypatch.setattr(
        tool_registry.socket,
        "getaddrinfo",
        lambda host, *args, **kwargs: _resolved(
            "127.0.0.1" if host == "localhost" else "93.184.216.34"),
    )
    handler = tool_registry._ValidatingRedirectHandler()

    with pytest.raises(PermissionError):
        handler.redirect_request(
            None, None, 302, "Found", {}, "http://localhost/admin")
