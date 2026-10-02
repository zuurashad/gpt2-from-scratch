"""web/template.js must build exactly the token ids chat.py builds, and so exactly the
format sft.py trained on. Runs the real JavaScript in a headless browser with the
transformers.js tokenizer (needs network: the library comes from its CDN)."""

import functools
import http.server
import os
import threading

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

from chat import EOT_ID, build_chat_ids  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LONG = [turn for i in range(30) for turn in (("user", f"Question {i}: " + "word " * 25),
                                              ("assistant", f"Answer {i}: " + "word " * 25))]
CASES = [
    ([("user", "Hi")], None),
    ([("user", "Hi"), ("assistant", "Hello! How can I help?"), ("user", "Tell me a joke.")], None),
    ([("assistant", "Welcome!"), ("user", "Hi")], 1024),
    ([("user", "naïve café \U0001F600 日本語")], None),
    (LONG + [("user", "Final question?")], 300),
    ([("user", "filler " * 3000 + "THE ACTUAL QUESTION?")], 256),
]
LITERAL = [("user", "say <|endoftext|> please")]


@pytest.fixture(scope="module")
def js_ids():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_api.sync_playwright() as p:
            try:
                browser = p.chromium.launch(channel="msedge")
            except Exception:  # noqa: BLE001 - CI has playwright's own Chromium
                browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"http://127.0.0.1:{server.server_port}/web/template.js")
            ids = page.evaluate("""async (cases) => {
                const { AutoTokenizer } = await import(
                    "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0");
                const t = await import("/web/template.js");
                const tok = await AutoTokenizer.from_pretrained("openai-community/gpt2");
                return cases.map(([h, max]) => t.buildChatIds(h, tok, t.SYSTEM_DEFAULT, max));
            }""", CASES + [(LITERAL, None)])
            browser.close()
    finally:
        server.shutdown()
    return ids


@pytest.fixture(scope="module")
def enc_():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


@pytest.mark.parametrize("i", range(len(CASES)))
def test_browser_prompt_is_token_identical_to_chat_py(js_ids, enc_, i):
    history, max_tokens = CASES[i]
    assert js_ids[i] == build_chat_ids(history, enc_, max_tokens=max_tokens)


def test_typed_end_of_text_cannot_end_the_turn_in_the_browser(js_ids, enc_):
    ids = js_ids[-1]
    assert ids.count(EOT_ID) == 2           # only the template's own turn ends
    assert enc_.decode(ids) == enc_.decode(build_chat_ids(LITERAL, enc_))
