"""The demo page end to end, in a headless browser, with tiny local models (?local).

Needs Playwright plus optimum/onnxruntime (requirements-web.txt) to build the models,
and network access for transformers.js and the GPT-2 tokenizer."""

import os
import time

import pytest

pytest.importorskip("playwright.sync_api")
pytest.importorskip("optimum.exporters.onnx")

from chat import build_chat_ids, encode_text  # noqa: E402
from tiny_web_model import build_web_model  # noqa: E402
from webpage import browser, serve  # noqa: E402

BLOCK = 1024
WAIT = 120_000      # ms; the first load fetches transformers.js and its WebAssembly runtime


@pytest.fixture(scope="module")
def model_sets(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    sets = {"q8": os.path.join(root, "q8"), "fp32": os.path.join(root, "fp32")}
    build_web_model(os.path.join(sets["q8"], "chat"), seed=0, quantise=True)
    build_web_model(os.path.join(sets["q8"], "base"), seed=1, quantise=True)
    build_web_model(os.path.join(sets["fp32"], "chat"), seed=0, quantise=False)
    return sets


@pytest.fixture(scope="module")
def context():
    with browser() as b:
        ctx = b.new_context()
        yield ctx
        ctx.close()


@pytest.fixture(scope="module")
def site(model_sets):
    with serve({"/web/models/": model_sets["q8"]}) as url:
        yield url


@pytest.fixture(scope="module")
def fp32_site(model_sets):
    with serve({"/web/models/": model_sets["fp32"]}) as url:
        yield url


class Page:
    def __init__(self, context, url, query="?local", init=None, block_cdn=False):
        self.page = context.new_page()
        self.errors, self.requests = [], []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("request", lambda r: self.requests.append(r.url))
        if init:
            self.page.add_init_script(init)
        if block_cdn:
            self.page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
        self.page.goto(f"{url}/web/index.html{query}")

    def status(self):
        return self.page.text_content("#status")

    def wait_status(self, text):
        self.page.wait_for_function("t => document.getElementById('status').textContent.includes(t)",
                                    arg=text, timeout=WAIT)

    def settings(self, **values):
        for key, value in values.items():
            self.page.eval_on_selector(f"#{key}", "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }",
                                       str(value))

    def runs(self):
        return self.page.evaluate("window.demoStats.runs")

    def send(self, text):
        self.page.fill("#chat-input", text)
        self.page.press("#chat-input", "Enter")

    def wait_runs(self, n, button="#chat-send"):
        self.page.wait_for_function(
            "([n, b]) => window.demoStats.runs.length >= n && !document.querySelector(b).disabled",
            arg=[n, button], timeout=WAIT)
        return self.runs()

    def wait_first_token(self):
        self.page.wait_for_function(
            "() => { const m = document.querySelectorAll('.msg.assistant'); "
            "return m.length && m[m.length - 1].textContent !== '…'; }", timeout=WAIT)

    def replies(self):
        return self.page.eval_on_selector_all(".msg.assistant", "els => els.map(e => e.textContent)")

    def notices(self):
        return self.page.eval_on_selector_all(".msg.notice", "els => els.map(e => e.textContent)")

    def close(self):
        assert not self.errors, self.errors
        self.page.close()


def greedy(page, max_new=16):
    page.settings(temperature=0, max_new_tokens=max_new)


def test_chat_prompt_is_token_identical_to_chat_py(context, site, enc):
    p = Page(context, site)
    p.wait_status("ready")
    greedy(p)
    p.send("Hello there")
    run = p.wait_runs(1)[0]
    assert run["kind"] == "chat"
    assert run["promptIds"] == build_chat_ids([("user", "Hello there")], enc, max_tokens=BLOCK - 16)
    assert len(p.page.query_selector_all(".msg.user")) == 1 and len(p.replies()) == 1
    p.close()


def chat_session(context, url, query, messages, max_new=16):
    p = Page(context, url, query=query)
    p.wait_status("ready")
    greedy(p, max_new)
    for i, m in enumerate(messages):
        p.send(m)
        p.wait_runs(i + 1)
    assert not p.notices()                          # real replies, so comparing them means something
    out = (p.replies(), p.runs())
    p.close()
    return out


def test_follow_ups_reuse_the_cache_and_match_a_full_recompute(context, fp32_site):
    # 256 new tokens leave a 748-token prompt budget, so the ~790-token third message
    # (inside the box's 4,000-character limit) is cut, and the history before it dropped
    messages = ["Hello there", "And a second question?", "word " * 790 + "end?"]
    replies, runs = chat_session(context, fp32_site, "?local", messages, max_new=256)
    fresh, fresh_runs = chat_session(context, fp32_site, "?local&nocache", messages, max_new=256)
    assert runs[2]["promptTokens"] <= BLOCK - 256
    assert replies == fresh
    assert [r["promptIds"] for r in runs] == [r["promptIds"] for r in fresh_runs]
    assert all(r["reusedTokens"] == 0 for r in fresh_runs)
    assert runs[0]["reusedTokens"] == 0
    for prev, cur in zip(runs, runs[1:]):
        # the cache holds the previous prompt plus its reply, so at least the prompts' shared
        # start is reused, and never the whole new prompt (its last token must be read)
        shared = next((i for i, (a, b) in enumerate(zip(prev["promptIds"], cur["promptIds"])) if a != b),
                      min(prev["promptTokens"], cur["promptTokens"]))
        assert shared <= cur["reusedTokens"] < cur["promptTokens"]
    assert runs[1]["reusedTokens"] > runs[0]["promptTokens"]     # the first reply was reused too
    assert runs[2]["reusedTokens"] < 40              # trimmed: little beyond the system prompt matches


def test_a_full_context_still_reuses_the_cache(context, fp32_site, enc):
    # 256-token replies fill the 768-token budget by the fourth message; the window then
    # jumps to half the budget, so later turns extend one prompt instead of each dropping
    # a turn (which would change the prompt's start and void the cache every time)
    messages = [f"Message number {i}: tell me something new." for i in range(7)]
    replies, runs = chat_session(context, fp32_site, "?local", messages, max_new=256)
    fresh, fresh_runs = chat_session(context, fp32_site, "?local&nocache", messages, max_new=256)
    assert replies == fresh
    assert [r["promptIds"] for r in runs] == [r["promptIds"] for r in fresh_runs]
    assert all(r["promptTokens"] <= BLOCK - 256 for r in runs)
    jumps = [i for i in range(1, len(runs)) if runs[i]["reusedTokens"] < 40]
    assert jumps and len(jumps) < len(runs) - 3          # the window moved, but not every turn
    for i in jumps:
        assert runs[i]["promptTokens"] <= (BLOCK - 256) // 2
        assert enc.decode(runs[i]["promptIds"]).count("<|user|>") >= 1
    later = [r for i, r in enumerate(runs) if i > jumps[0] and i not in jumps]
    assert later and all(r["reusedTokens"] > 256 for r in later)   # previous prompt + reply reused


def test_page_stays_responsive_and_streams_while_writing(context, site):
    p = Page(context, site)
    p.wait_status("ready")
    p.settings(temperature=1, max_new_tokens=512)
    p.send("go")
    t = time.perf_counter()
    p.page.fill("#chat-input", "typing still works")   # handled between generation steps
    took = time.perf_counter() - t
    # text appears while the reply is still being written (polled once per frame)
    p.page.wait_for_function("document.getElementById('chat-send').disabled && "
                             "document.querySelector('.msg.assistant').textContent.length > 20", timeout=WAIT)
    run = p.wait_runs(1)[0]
    assert took < 0.5 < run["totalMs"] / 1000
    assert p.page.input_value("#chat-input") == "typing still works"
    p.close()


def test_one_reply_at_a_time_and_the_draft_survives(context, site):
    p = Page(context, site)
    p.wait_status("ready")
    p.settings(temperature=1, max_new_tokens=512)
    p.send("first")
    p.page.fill("#chat-input", "draft")
    p.page.press("#chat-input", "Enter")           # while the first reply is still running
    p.page.click("#chat-send", force=True)
    p.wait_runs(1)
    assert len(p.runs()) == 1 and len(p.page.query_selector_all(".msg.user")) == 1
    assert p.page.input_value("#chat-input") == "draft"
    p.close()


def test_stop_ends_the_reply_and_the_next_one_still_works(context, site):
    p = Page(context, site)
    p.wait_status("ready")
    p.settings(temperature=1, max_new_tokens=512)
    p.send("tell me everything")
    p.wait_first_token()
    p.page.click("#chat-stop")
    run = p.wait_runs(1)[0]
    assert run["newTokens"] < 512
    greedy(p)
    p.send("next")
    assert len(p.wait_runs(2)) == 2
    p.close()


def test_clear_during_a_reply_starts_a_clean_chat(context, site, enc):
    p = Page(context, site)
    p.wait_status("ready")
    p.settings(temperature=1, max_new_tokens=512)
    p.send("old conversation")
    p.wait_first_token()
    p.page.click("#chat-clear")
    p.wait_runs(1)
    assert p.page.eval_on_selector("#chat-log", "el => el.children.length") == 0
    greedy(p)
    p.send("new")
    run = p.wait_runs(2)[-1]
    assert run["promptIds"] == build_chat_ids([("user", "new")], enc, max_tokens=BLOCK - 16)
    assert run["reusedTokens"] == 0
    p.close()


def test_typed_end_of_text_reaches_the_model_as_plain_text(context, site, enc):
    p = Page(context, site)
    p.wait_status("ready")
    greedy(p)
    message = "say <|endoftext|> now"
    p.send(message)
    run = p.wait_runs(1)[0]
    assert run["promptIds"] == build_chat_ids([("user", message)], enc, max_tokens=BLOCK - 16)
    p.close()


def test_a_paste_longer_than_the_context_is_cut_to_fit(context, site, enc):
    p = Page(context, site)
    p.wait_status("ready")
    p.settings(temperature=0, max_new_tokens=512)    # leaves 512 tokens for an ~800-token paste
    message = "word " * 795 + "THE QUESTION?"
    p.send(message)
    run = p.wait_runs(1)[0]
    assert run["promptIds"] == build_chat_ids([("user", message)], enc, max_tokens=BLOCK - 512)
    assert run["promptTokens"] <= BLOCK - 512
    assert "THE QUESTION?" in enc.decode(run["promptIds"])     # the end of a paste is what survives
    p.close()


def test_completion_tab_uses_the_base_model_and_drops_trailing_spaces(context, site, enc):
    p = Page(context, site)
    p.wait_status("ready")
    p.page.click("#tab-complete")
    p.wait_status("base model is ready")
    greedy(p)
    p.page.fill("#complete-input", "Once upon a time   ")
    p.page.click("#complete-go")
    run = p.wait_runs(1, "#complete-go")[0]
    assert run["kind"] == "base" and run["promptIds"] == encode_text(enc, "Once upon a time")
    assert p.page.text_content("#complete-output").startswith("Once upon a time")
    p.page.click("#complete-go")                    # the same prompt again reuses its cache
    again = p.wait_runs(2, "#complete-go")[-1]
    assert again["reusedTokens"] == len(run["promptIds"]) - 1
    p.close()


def test_a_blocked_cdn_gets_an_explanation(context, site):
    p = Page(context, site, block_cdn=True)
    p.wait_status("could not download transformers.js")
    p.errors.clear()                                # the blocked modulepreload is expected
    p.close()


def test_a_browser_without_simd_is_told_so(context, site):
    p = Page(context, site, init="WebAssembly.validate = () => false;")
    assert p.status().startswith("Sorry:")
    assert p.page.is_disabled("#chat-send")
    p.close()


def test_data_saver_downloads_nothing_until_the_first_message(context, site):
    p = Page(context, site, init="Object.defineProperty(navigator, 'connection', "
                                 "{ value: { saveData: true }, configurable: true });")
    p.page.wait_for_timeout(1500)
    assert "Data saver" in p.status()
    assert not [u for u in p.requests if "/web/models/" in u]
    greedy(p)
    p.send("hello")
    assert len(p.wait_runs(1)) == 1
    p.close()


def test_cross_origin_isolated_page_still_runs(context, model_sets):
    # the Space sends these headers so ONNX Runtime can use several threads
    with serve({"/web/models/": model_sets["q8"]}, isolate=True) as url:
        p = Page(context, url)
        assert p.page.evaluate("crossOriginIsolated")
        p.wait_status("ready")
        greedy(p)
        p.send("hello")
        assert len(p.wait_runs(1)) == 1
        p.close()
