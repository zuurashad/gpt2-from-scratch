// The chat template, ported line for line from chat.py so the browser prompts the model
// with exactly the token sequence sft.py trained it on. tests/test_web_template.py runs
// this file under Node and compares its output with chat.build_chat_ids.

export const SYSTEM_DEFAULT = "You are a helpful assistant.";
export const B_SYS = "<|system|>\n";
export const B_USER = "<|user|>\n";
export const B_ASSISTANT = "<|assistant|>\n";
export const EOT_ID = 50256;

const LITERAL_EOT = "<|endoftext|>";
// "<|endoftext|>" encoded as ordinary characters, as tiktoken does with disallowed_special=()
const LITERAL_EOT_IDS = [27, 91, 437, 1659, 5239, 91, 29];

// Encode user-controlled text with no special tokens at all (chat.encode_text). The
// tokenizer treats "<|endoftext|>" as the special token, so a typed copy of it is split
// out and inserted as plain characters: a visitor can never end the model's turn.
export function encodeText(tokenizer, text) {
  const parts = text.split(LITERAL_EOT);
  const ids = [];
  parts.forEach((part, i) => {
    if (i > 0) ids.push(...LITERAL_EOT_IDS);
    if (part) ids.push(...tokenizer.encode(part, { add_special_tokens: false }));
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

// Chat log -> strictly alternating [role, text] turns (app.to_turns).
export function toTurns(messages, maxChars = 4000) {
  const turns = [];
  for (const { role, text: raw } of messages) {
    const text = (raw || "").trim().slice(0, maxChars);
    if (!(role === "user" || role === "assistant") || !text || text === EMPTY_TURN) continue;
    if (turns.length && turns[turns.length - 1][0] === role) {
      turns[turns.length - 1] = [role, turns[turns.length - 1][1] + "\n\n" + text];
    } else {
      turns.push([role, text]);
    }
  }
  return turns;
}
