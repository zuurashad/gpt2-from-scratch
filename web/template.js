// The chat template, ported line for line from chat.py so the browser prompts the model
// with exactly the token sequence sft.py trained it on. tests/test_web_template.py runs
// this file in a headless browser and compares its output with chat.build_chat_ids.

export const SYSTEM_DEFAULT = "You are a helpful assistant.";
export const B_SYS = "<|system|>\n";
export const B_USER = "<|user|>\n";
export const B_ASSISTANT = "<|assistant|>\n";
export const EOT_ID = 50256;

const LITERAL_EOT = "<|endoftext|>";

// A JS string can hold half of a surrogate pair (a pasted, broken emoji); UTF-8 cannot,
// so Python would only ever receive U+FFFD in its place. Do the same here.
export function wellFormed(text) {
  if (typeof text.toWellFormed === "function") return text.toWellFormed();
  let out = "";
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (c >= 0xd800 && c <= 0xdbff && i + 1 < text.length) {
      const d = text.charCodeAt(i + 1);
      if (d >= 0xdc00 && d <= 0xdfff) { out += text[i] + text[i + 1]; i++; continue; }
    }
    out += c >= 0xd800 && c <= 0xdfff ? "�" : text[i];
  }
  return out;
}

const plain = (tokenizer, text) => (text ? tokenizer.encode(text, { add_special_tokens: false }) : []);

// Encode user-controlled text with no special tokens at all (chat.encode_text). The
// tokenizer would turn a typed "<|endoftext|>" into the special token, letting a visitor
// end the model's turn, so the text is cut just before and after "endoftext". That word
// is always a pre-token of its own (it sits between two "|"), so the cuts never change how
// the surrounding text splits: the ids equal tiktoken's with disallowed_special=(), even
// when the string touches spaces, letters or punctuation.
export function encodeText(tokenizer, text) {
  text = wellFormed(text);
  if (!text.includes(LITERAL_EOT)) return plain(tokenizer, text);
  const parts = text.split(LITERAL_EOT);
  const ids = [];
  parts.forEach((part, i) => {
    const last = i === parts.length - 1;
    ids.push(...plain(tokenizer, (i > 0 ? "|>" : "") + part + (last ? "" : "<|")));
    if (!last) ids.push(...plain(tokenizer, "endoftext"));
  });
  return ids;
}

// history: [[role, text], ...] -> token ids ending where the model continues.
// maxTokens trims WHOLE turns from the front, oldest first, never the system prompt.
export function buildChatIds(history, tokenizer, system = SYSTEM_DEFAULT, maxTokens = null) {
  const heads = { user: B_USER, assistant: B_ASSISTANT };
  for (const [role] of history) {
    if (!(role in heads)) throw new Error(`unknown role ${role}; expected 'user' or 'assistant'`);
  }
  const prefix = [...encodeText(tokenizer, B_SYS + system), EOT_ID];
  let turns = history.map(([role, text]) => [...encodeText(tokenizer, heads[role] + text), EOT_ID]);
  const tail = encodeText(tokenizer, B_ASSISTANT);

  if (maxTokens !== null) {
    const budget = maxTokens - prefix.length - tail.length;
    let start = turns.length;
    let used = 0;
    while (start > 0 && used + turns[start - 1].length <= budget) {
      start -= 1;
      used += turns[start].length;
    }
    // a trimmed window opening on an assistant turn answers a question the model can't see
    while (start > 0 && start < turns.length - 1 && history[start][0] !== "user") start += 1;
    if (turns.length && start === turns.length) {
      // even the newest turn alone is too long: keep its end
      const [role, text] = history[history.length - 1];
      const head = encodeText(tokenizer, heads[role]);
      const room = budget - head.length - 1;
      if (room <= 0) throw new Error(`maxTokens=${maxTokens} leaves no room for a message`);
      let body = encodeText(tokenizer, text).slice(-room);
      while (body.length > 1 && tokenizer.decode(body).startsWith("�")) body = body.slice(1);
      turns[turns.length - 1] = [...head, ...body, EOT_ID];
      start = turns.length - 1;
    }
    turns = turns.slice(start);
  }
  return [...prefix, ...turns.flat(), ...tail];
}

// The page's own notices are not model output and must never be fed back to it.
export const EMPTY_TURN = "(The model ended its turn without writing anything. Try again, or raise the temperature.)";

// Python's str.strip(): JS trim() also removes U+FEFF and keeps U+001C-U+001F and U+0085.
const PY_SPACE = "[\\t\\n\\v\\f\\r\\x1c-\\x20\\x85\\xa0\\u1680\\u2000-\\u200a\\u2028\\u2029\\u202f\\u205f\\u3000]";
const PY_STRIP = new RegExp(`^${PY_SPACE}+|${PY_SPACE}+$`, "g");

// Chat log -> strictly alternating [role, text] turns (app.to_turns). The length cap
// counts characters, not UTF-16 units, so it can never cut an emoji in half.
export function toTurns(messages, maxChars = 4000) {
  const turns = [];
  for (const { role, text: raw } of messages) {
    const text = Array.from(wellFormed(raw || "").replace(PY_STRIP, "")).slice(0, maxChars).join("");
    if (!(role === "user" || role === "assistant") || !text || text === EMPTY_TURN) continue;
    if (turns.length && turns[turns.length - 1][0] === role) {
      turns[turns.length - 1] = [role, turns[turns.length - 1][1] + "\n\n" + text];
    } else {
      turns.push([role, text]);
    }
  }
  return turns;
}

// The base model's prompt. Trailing spaces are dropped: GPT-2's tokens carry their
// leading space (" Paris"), so a prompt ending in one forces the rare space-less variant
// of the next word and the continuation reads badly.
export function cleanCompletionPrompt(text) {
  return wellFormed(text).replace(/[ \t]+$/, "");
}

// Fit a completion prompt into maxTokens by keeping its end, where the model continues,
// without opening on the tail bytes of a character the cut went through.
export function keepEnd(ids, maxTokens, tokenizer) {
  if (ids.length <= maxTokens) return ids;
  let body = ids.slice(-maxTokens);
  while (body.length > 1 && tokenizer.decode(body.slice(0, 4)).startsWith("�")) body = body.slice(1);
  return body;
}
