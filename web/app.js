// In-browser demo: transformers.js runs the int8 ONNX export of each model on the
// visitor's CPU (WebAssembly). Prompts are built by template.js, the exact chat
// template the model was fine-tuned on.
import {
  AutoModelForCausalLM, AutoTokenizer, BaseStreamer, InterruptableStoppingCriteria, Tensor, env,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
import { EMPTY_TURN, EOT_ID, SYSTEM_DEFAULT, buildChatIds, encodeText, toTurns } from "./template.js";

const BLOCK_SIZE = 1024;
const LOCAL = new URLSearchParams(location.search).has("local");   // ?local: serve ./models/
const MODELS = LOCAL
  ? { chat: "chat", base: "base" }
  : { chat: "zuu007/gpt2-from-scratch-chat", base: "zuu007/gpt2-from-scratch" };
if (LOCAL) {
  env.allowLocalModels = true;          // off by default in browsers
  env.allowRemoteModels = false;
  env.localModelPath = new URL("./models/", location.href).pathname;   // a path, not a URL
}

const $ = (id) => document.getElementById(id);
const status = (text) => { $("status").textContent = text; };

// ---- model loading (each model once, on first use) ----
const loaded = {};
function load(kind) {
  if (!loaded[kind]) {
    const files = {};
    const progress = (p) => {
      if (p.status === "progress" && p.total) {
        files[p.file] = [p.loaded, p.total];
        const [done, total] = Object.values(files).reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
        status(`Downloading the ${kind} model: ${(done / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`);
      }
    };
    loaded[kind] = (async () => {
      const tokenizer = await AutoTokenizer.from_pretrained(MODELS[kind]);
      const model = await AutoModelForCausalLM.from_pretrained(MODELS[kind], {
        dtype: "q8", device: "wasm", progress_callback: progress,
      });
      status(`The ${kind} model is ready.`);
      return { tokenizer, model };
    })();
    loaded[kind].catch((e) => { delete loaded[kind]; status(`Could not load the model: ${e.message}`); });
  }
  return loaded[kind];
}

// ---- generation ----
function settings() {
  const v = (id) => Number($(id).value);
  const temperature = v("temperature") < 1e-3 ? 0 : v("temperature");
  return {
    max_new_tokens: v("max_new_tokens"), temperature, do_sample: temperature > 0,
    top_k: v("top_k"), top_p: v("top_p"), repetition_penalty: v("repetition_penalty"),
  };
}

// Streams decoded text, holding output while it ends mid-character (an emoji is
// several byte-level tokens, each U+FFFD on its own) - the same rule as chat.py.
class TextUpdates extends BaseStreamer {
  constructor(tokenizer, onText) {
    super();
    Object.assign(this, { tokenizer, onText, ids: [], text: "", prompt: true });
  }
  put(value) {
    if (this.prompt) { this.prompt = false; return; }      // the first call carries the prompt
    for (const t of value[0]) if (Number(t) !== EOT_ID) this.ids.push(Number(t));
    const full = this.tokenizer.decode(this.ids);
    if (!full.endsWith("�") && full !== this.text) { this.text = full; this.onText(full); }
  }
  end() {
    const full = this.tokenizer.decode(this.ids);
    if (full !== this.text) { this.text = full; this.onText(full); }
  }
}

let stopper = null;
async function generate(kind, ids, onText) {
  const { tokenizer, model } = await load(kind);
  const s = settings();
  ids = ids.slice(-(BLOCK_SIZE - s.max_new_tokens));
  stopper = new InterruptableStoppingCriteria();
  const streamer = new TextUpdates(tokenizer, onText);
  const t0 = performance.now();
  await model.generate({
    inputs: new Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
    attention_mask: new Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
    ...s, eos_token_id: EOT_ID, streamer, stopping_criteria: stopper,
  });
  const n = streamer.ids.length;
  status(`${n} tokens in ${((performance.now() - t0) / 1000).toFixed(1)} s ` +
         `(${(n / ((performance.now() - t0) / 1000)).toFixed(1)} tokens/s, on your CPU)`);
  stopper = null;
  return streamer.text;
}

function busy(kind, on) {
  const [go, stop] = kind === "chat" ? ["chat-send", "chat-stop"] : ["complete-go", "complete-stop"];
  $(go).disabled = on;
  $(stop).disabled = !on;
}

// ---- chat tab ----
const log = [];      // [{role, text}] as shown on the page
function bubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
  return div;
}

async function send(message) {
  message = message.trim();
  if (!message) return;
  $("chat-log").querySelectorAll(".placeholder, .examples").forEach((el) => el.remove());
  log.push({ role: "user", text: message });
  bubble("user", message);
  const reply = bubble("assistant", "…");
  busy("chat", true);
  try {
    const { tokenizer } = await load("chat");
    const max = Number($("max_new_tokens").value);
    const ids = buildChatIds(toTurns(log), tokenizer, SYSTEM_DEFAULT, BLOCK_SIZE - max);
    const text = await generate("chat", ids, (t) => { reply.textContent = t; });
    if (text.trim()) {
      log.push({ role: "assistant", text });
    } else {
      reply.textContent = EMPTY_TURN;
      reply.classList.add("notice");
    }
  } catch (e) {
    reply.textContent = `Error: ${e.message}`;
    reply.classList.add("notice");
  } finally {
    busy("chat", false);
  }
}

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("chat-input").value;
  $("chat-input").value = "";
  send(text);
});
$("chat-input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("chat-form").requestSubmit(); }
});
document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
$("chat-stop").addEventListener("click", () => stopper?.interrupt());
$("chat-clear").addEventListener("click", () => {
  log.length = 0;
  $("chat-log").replaceChildren();
});

// ---- completion tab ----
$("complete-go").addEventListener("click", async () => {
  const prompt = $("complete-input").value;
  if (!prompt.trim()) return;
  const out = $("complete-output");
  out.textContent = prompt;
  busy("complete", true);
  try {
    const { tokenizer } = await load("base");
    await generate("base", encodeText(tokenizer, prompt), (t) => { out.textContent = prompt + t; });
  } catch (e) {
    out.textContent += `\n\nError: ${e.message}`;
  } finally {
    busy("complete", false);
  }
});
$("complete-stop").addEventListener("click", () => stopper?.interrupt());

// ---- tabs and settings ----
for (const [tab, panel, kind] of [["tab-chat", "panel-chat", "chat"], ["tab-complete", "panel-complete", "base"]]) {
  $(tab).addEventListener("click", () => {
    document.querySelectorAll('[role="tab"]').forEach((t) => t.setAttribute("aria-selected", String(t.id === tab)));
    document.querySelectorAll('[role="tabpanel"]').forEach((p) => { p.hidden = p.id !== panel; });
    load(kind);
  });
}
for (const id of ["temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"]) {
  $(id).addEventListener("input", () => { $(`${id}-v`).textContent = $(id).value; });
}

load("chat");
