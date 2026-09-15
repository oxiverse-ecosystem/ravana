import socket
import urllib.error

import pytest

from ravana.agent import tool_registry


def _resolved(ip):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 80))]


class TestStripHtml:
    def test_strips_tags(self):
        html = '<html><body><p>Hello world</p></body></html>'
        assert tool_registry._strip_html(html) == "Hello world"

    def test_removes_script_style(self):
        html = '<html><body><script>alert(1)</script><style>.x{}</style>Content</body></html>'
        result = tool_registry._strip_html(html)
        assert 'alert' not in result
        assert '.x' not in result
        assert 'Content' in result

    def test_removes_nav_footer_header(self):
        html = '<html><body><nav>Menu</nav><main>Main</main><footer>Foot</footer></body></html>'
        result = tool_registry._strip_html(html)
        assert 'Menu' not in result
        assert 'Foot' not in result
        assert 'Main' in result

    def test_extracts_title(self):
        html = '<html><head><title>My Title</title></head><body>Body</body></html>'
        result = tool_registry._strip_html(html)
        assert result.startswith('[My Title]')
        assert 'Body' in result

    def test_collapses_whitespace(self):
        html = '<html><body><p>Line   one</p>\n\n<p>Line two</p></body></html>'
        assert tool_registry._strip_html(html) == "Line one Line two"

    def test_empty_page(self):
        html = '<html><head></head><body></body></html>'
        assert tool_registry._strip_html(html) == ""


class TestReadWebsiteExtraction:
    """Integration-level: confirm _read_website returns clean text, not raw HTML."""

    def test_returns_clean_text_not_html(self, monkeypatch):
        raw_html = b'<!doctype html><html><head><title>Test</title></head><body><h1>Welcome</h1><p>Content here.</p></body></html>'

        class FakeResp:
            def read(self):
                return raw_html
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        class FakeOpener:
            def open(self, url, timeout=8):
                return FakeResp()

        monkeypatch.setattr(tool_registry, "_validate_public_url", lambda url: None)
        monkeypatch.setattr(tool_registry.urllib.request, "build_opener", lambda *a, **kw: FakeOpener())

        result = tool_registry._read_website("https://example.com")
        assert '<!doctype' not in result
        assert '<html' not in result
        assert '[Test]' in result
        assert 'Welcome' in result
        assert 'Content here.' in result

    def test_handles_fetch_failure_gracefully(self, monkeypatch):
        class FakeOpener:
            def open(self, url, timeout=8):
                raise urllib.error.URLError("connection refused")

        monkeypatch.setattr(tool_registry, "_validate_public_url", lambda url: None)
        monkeypatch.setattr(tool_registry.urllib.request, "build_opener", lambda *a, **kw: FakeOpener())

        result = tool_registry._read_website("https://example.com")
        assert 'fetch failed' in result

    def test_empty_page(self, monkeypatch):
        class FakeResp:
            def read(self):
                return b'<html><head></head><body></body></html>'
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass

        class FakeOpener:
            def open(self, url, timeout=8):
                return FakeResp()

        monkeypatch.setattr(tool_registry, "_validate_public_url", lambda url: None)
        monkeypatch.setattr(tool_registry.urllib.request, "build_opener", lambda *a, **kw: FakeOpener())

        result = tool_registry._read_website("https://example.com")
        assert 'empty page' in result


# Keep the original security tests below
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
