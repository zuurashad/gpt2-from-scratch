"""web/template.js must build exactly the token ids chat.py builds, and so exactly the
format sft.py trained on. Runs the real JavaScript in a headless browser with the
transformers.js tokenizer (needs network: the library comes from its CDN)."""

import pytest

pytest.importorskip("playwright.sync_api")

from app import EMPTY_TURN as APP_EMPTY_TURN  # noqa: E402
from app import to_turns  # noqa: E402
from chat import EOT_ID, build_chat_ids, encode_text  # noqa: E402
from webpage import LIBRARY, browser, serve  # noqa: E402

LONG = [turn for i in range(30) for turn in (("user", f"Question {i}: " + "word " * 25),
                                              ("assistant", f"Answer {i}: " + "word " * 25))]
CASES = [
    ([("user", "Hi")], None),
    ([("user", "Hi"), ("assistant", "Hello! How can I help?"), ("user", "Tell me a joke.")], None),
    ([("assistant", "Welcome!"), ("user", "Hi")], 1024),
    ([("user", "naïve café \U0001F600 日本語")], None),
    (LONG + [("user", "Final question?")], 300),
    ([("user", "filler " * 3000 + "THE ACTUAL QUESTION?")], 256),
    ([("user", "\U0001F600" * 400)], 200),            # the cut lands inside multi-token emoji
]
# a typed "<|endoftext|>" in every position that changes how BPE splits its neighbours
LITERALS = [
    "say <|endoftext|> please", "<|endoftext|>", "<|endoftext|><|endoftext|>",
    "a<|endoftext|>b", "x <|endoftext|>!!", "  <|endoftext|>  ", "'<|endoftext|>'s",
    "<|endoftext|>\n<|endoftext|>", "emoji \U0001F600<|endoftext|>日本", "(<|endoftext|>)",
    "<|endoftext", "endoftext|>", "<|endoftext|>endoftext", "<<|endoftext|>>",
    "12<|endoftext|>34", "tab\t<|endoftext|>\tend",
]
EMPTY = "\x00EMPTY"                                    # placeholder for each side's own notice
TURNS = [  # (role, text) as the page logs them; the last one is the new message
    ("user", "  hi  "), ("assistant", EMPTY), ("user", "again"), ("assistant", "   "),
    ("assistant", "ok"), ("system", "ignored"), ("assistant", "\x1c\x85both strip these　"),
    ("user", "a" * 3999 + "\U0001F600" + "b"),        # the 4,000-character cap meets an emoji
    ("user", "﻿kept by Python's strip"),
]
SPACES = "abc  \t"
EMOJI_LONG = "\U0001F600 words " * 300


@pytest.fixture(scope="module")
def js():
    with serve() as url, browser() as b:
        page = b.new_page()
        page.goto(f"{url}/web/template.js")
        result = page.evaluate("""async ({ lib, cases, literals, turns, empty, spaces, emojiLong }) => {
            const { AutoTokenizer } = await import(lib);
            const t = await import("/web/template.js");
            const tok = await AutoTokenizer.from_pretrained("openai-community/gpt2");
            const lone = String.fromCharCode(0xd83d) + " broken " + String.fromCharCode(0xdc00) + "x";
            const log = turns.map(([role, text]) => ({ role, text: text === empty ? t.EMPTY_TURN : text }));
            const long = t.encodeText(tok, emojiLong);
            return {
                chat: cases.map(([h, max]) => t.buildChatIds(h, tok, t.SYSTEM_DEFAULT, max)),
                literal: literals.map((s) => t.encodeText(tok, s)),
                literalChat: literals.map((s) => t.buildChatIds([["user", s]], tok)),
                lone: t.buildChatIds([["user", lone]], tok),
                turns: t.toTurns(log),
                clean: [t.cleanCompletionPrompt(spaces), t.cleanCompletionPrompt("line\\n"),
                        t.cleanCompletionPrompt("  lead kept")],
                keep: [long, t.keepEnd(long, 51, tok), t.keepEnd(long, 50, tok), t.keepEnd(long, 10000, tok)],
            };
        }""", {"lib": LIBRARY, "cases": CASES, "literals": LITERALS, "turns": TURNS, "empty": EMPTY,
               "spaces": SPACES, "emojiLong": EMOJI_LONG})
    return result


@pytest.mark.parametrize("i", range(len(CASES)))
def test_browser_prompt_is_token_identical_to_chat_py(js, enc, i):
    history, max_tokens = CASES[i]
    assert js["chat"][i] == build_chat_ids(history, enc, max_tokens=max_tokens)


@pytest.mark.parametrize("i", range(len(LITERALS)))
def test_typed_end_of_text_is_plain_text_with_tiktoken_ids(js, enc, i):
    text = LITERALS[i]
    assert js["literal"][i] == encode_text(enc, text)
    ids = js["literalChat"][i]
    assert ids == build_chat_ids([("user", text)], enc)
    assert ids.count(EOT_ID) == 2              # only the template's own turn ends


def test_half_of_a_surrogate_pair_becomes_the_replacement_character(js, enc):
    assert js["lone"] == build_chat_ids([("user", "� broken �x")], enc)


def test_chat_log_becomes_the_same_turns_as_app_py(js):
    *history, (_, message) = TURNS
    history = [{"role": r, "content": APP_EMPTY_TURN if t == EMPTY else t} for r, t in history]
    assert [tuple(t) for t in js["turns"]] == to_turns(history, message)


def test_completion_prompt_loses_trailing_spaces_only(js):
    assert js["clean"] == ["abc", "line\n", "  lead kept"]


def test_long_completion_prompt_keeps_its_end_on_a_character_boundary(js, enc):
    full, *kept = js["keep"]
    for ids, limit in zip(kept, (51, 50, 10000)):
        assert len(ids) <= limit and ids == full[len(full) - len(ids):]
        assert not enc.decode(ids).startswith("�")
    assert kept[2] == full
