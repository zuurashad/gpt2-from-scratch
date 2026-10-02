"""Helpers for the browser tests: a local static server and a headless browser.

The page loads transformers.js from its CDN and the GPT-2 tokenizer from the Hugging Face
Hub, so these tests need network access."""

import contextlib
import functools
import http.server
import os
import threading
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0"


class _Handler(http.server.SimpleHTTPRequestHandler):
    mounts = {}        # URL prefix -> directory, checked before the repository root
    isolate = False    # send the headers that make the page cross-origin isolated

    def translate_path(self, path):
        path = urllib.parse.unquote(path.split("?", 1)[0].split("#", 1)[0])
        for prefix, directory in self.mounts.items():
            if path.startswith(prefix):
                return os.path.join(directory, *[p for p in path[len(prefix):].split("/") if p])
        return super().translate_path(path)

    def end_headers(self):
        if self.isolate:
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def log_message(self, *args):
        pass


@contextlib.contextmanager
def serve(mounts=None, isolate=False):
    """Serve the repository (so /web/index.html is the page) on a free local port."""
    handler = type("Handler", (_Handler,), {"mounts": dict(mounts or {}), "isolate": isolate})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(handler, directory=ROOT))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


@contextlib.contextmanager
def browser():
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        try:
            b = p.chromium.launch(channel="msedge", args=["--disable-gpu"])
        except Exception:  # noqa: BLE001 - CI has Playwright's own Chromium instead of Edge
            b = p.chromium.launch(args=["--disable-gpu"])
        try:
            yield b
        finally:
            b.close()
