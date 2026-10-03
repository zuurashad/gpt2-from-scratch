// In-browser demo: transformers.js runs the int8 ONNX export of each model on the
// visitor's CPU (WebAssembly). template.js builds the prompts in exactly the format the
// model was fine-tuned on, and sampling.js picks tokens the way chat.py does.
import {
  EMPTY_TURN, EOT_ID, SYSTEM_DEFAULT, buildChatIds, cleanCompletionPrompt, encodeText, keepEnd, toTurns,
} from "./template.js";
import { commonPrefixLength, makeRule } from "./sampling.js";

const LIBRARY = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0";
const BLOCK_SIZE = 1024;
const MODEL_MB = 125;
const query = new URLSearchParams(location.search);
const LOCAL = query.has("local");             // ?local: load ./models/{chat,base}/ instead of the Hub
const REUSE = !query.has("nocache");          // ?nocache: re-read the whole prompt every turn
const MODELS = LOCAL
  ? { chat: "chat", base: "base" }
  : { chat: "zuu007/gpt2-from-scratch-chat", base: "zuu007/gpt2-from-scratch" };
const NAMES = { chat: "chat model", base: "base model" };

const $ = (id) => document.getElementById(id);
const status = (text) => { $("status").textContent = text; };
const stats = (window.demoStats = { runs: [] });   // what each generation did (read by the tests)

// ---- can this browser run it at all? ----
// ONNX Runtime Web needs WebAssembly with SIMD; this 31-byte module uses i8x16 instructions
const SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0,
  10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
function unsupported() {
  if (typeof WebAssembly !== "object" || typeof BigInt64Array !== "function") {
    return "this browser does not support WebAssembly";
  }
  try {
    if (!WebAssembly.validate(SIMD_PROBE)) return "this browser's WebAssembly has no SIMD support";
  } catch {
    return "this browser's WebAssembly has no SIMD support";
  }
  return null;
}

// Phones and small devices keep one model in memory at a time: switching tabs reloads the
// other one from the browser's cache (a second or two) instead of holding both.
const CONSTRAINED = (navigator.deviceMemory !== undefined && navigator.deviceMemory <= 4)
  || navigator.userAgentData?.mobile === true || /Android|iPhone|iPad|iPod|Mobile/i.test(navigator.userAgent);

// ---- the library: imported on demand, so a blocked CDN gives a message, not a dead page ----
let library = null;
function loadLibrary() {
  library ??= import(LIBRARY).then((lib) => {
    if (LOCAL) {
      lib.env.allowLocalModels = true;         // off by default in browsers
      lib.env.allowRemoteModels = false;
      lib.env.localModelPath = new URL("./models/", location.href).pathname;   // a path, not a URL
    }
    return lib;
  }, (e) => {
    library = null;
    throw new Error(`could not download transformers.js from cdn.jsdelivr.net (${e.message}). `
      + "A content blocker or network filter may be stopping it.");
  });
  return library;
}

// ONNX Runtime's WebAssembly backend computes on this thread, and transformers.js chains
// the steps of a generation with promises that never give the browser a turn. Without a
// yield before each step the page would not repaint until the reply was finished (no
// visible streaming), and a click on Stop, or typing, would wait just as long. A message
// to ourselves is an ordinary task queued behind whatever is already waiting (unlike
// scheduler.yield(), whose continuation jumps that queue, or setTimeout, clamped to 4ms).
const yieldToBrowser = (() => {
  const channel = new MessageChannel();
  const waiting = [];
  channel.port1.onmessage = () => waiting.shift()();
  return () => new Promise((resolve) => { waiting.push(resolve); channel.port2.postMessage(null); });
})();

function yielding(model) {
  const forward = model.forward.bind(model);
  model.forward = async (inputs) => {
    await yieldToBrowser();
    return forward(inputs);
  };
  return model;
}

// ---- models (each loaded once, on first use) ----
const models = {};     // kind -> Promise<{ tokenizer, model }>
const caches = {};     // kind -> { ids, pkv }: the attention cache left by the last generation
let active = null;     // { kind, stopper, stopped } while a reply is being written

function load(kind) {
  if (!models[kind]) {
    if (CONSTRAINED) {
      for (const other of Object.keys(models)) if (other !== kind && active?.kind !== other) release(other);
    }
    const files = {};
    const progress = (p) => {
      if (p.status !== "progress" || !p.total) return;
      files[p.file] = [p.loaded, p.total];
      const [done, total] = Object.values(files).reduce(([a, b], [x, y]) => [a + x, b + y], [0, 0]);
      status(`Downloading the ${NAMES[kind]}: ${(done / 1e6).toFixed(0)} / ${(total / 1e6).toFixed(0)} MB`
        + " (only on the first visit: your browser keeps a copy)");
    };
    const pending = (async () => {
      const lib = await loadLibrary();
      const [tokenizer, model] = await Promise.all([
        lib.AutoTokenizer.from_pretrained(MODELS[kind]),
        lib.AutoModelForCausalLM.from_pretrained(MODELS[kind], {
          dtype: "q8", device: "wasm", progress_callback: progress,
        }),
      ]);
      status(`The ${NAMES[kind]} is ready.`);
      return { tokenizer, model: yielding(model) };
    })();
    models[kind] = pending;
    pending.catch((e) => {
      if (models[kind] === pending) delete models[kind];
      status(`Could not load the ${NAMES[kind]}: ${e.message}`);
    });
  }
  return models[kind];
}

async function release(kind) {
  const pending = models[kind];
  delete models[kind];
  delete caches[kind];
  try {
    const { model } = await pending;
    await model.dispose();
  } catch { /* it never loaded, or is already gone */ }
}

// ---- generation ----
function settings() {
  const v = (id) => Number($(id).value);
  const temperature = v("temperature") < 1e-3 ? 0 : v("temperature");   // 0 = greedy
  return {
    temperature, top_p: v("top_p"), top_k: Math.round(v("top_k")),
    repetition_penalty: v("repetition_penalty"), max_new_tokens: Math.round(v("max_new_tokens")),
  };
}

// Streams decoded text, holding output while it ends mid-character (an emoji is several
// byte-level tokens, each U+FFFD on its own): the same rule as chat.py.
function makeStreamer(BaseStreamer, tokenizer, onText, onFirst) {
  return new (class extends BaseStreamer {
    ids = [];
    text = "";
    prompt = true;

    put(value) {
      if (this.prompt) { this.prompt = false; return; }      // the first call carries the prompt
      if (!this.ids.length) onFirst();
      for (const t of value[0]) if (Number(t) !== EOT_ID) this.ids.push(Number(t));
      const full = tokenizer.decode(this.ids);
      if (!full.endsWith("�") && full !== this.text) { this.text = full; onText(full); }
    }

    end() {
      const full = tokenizer.decode(this.ids);
      if (full !== this.text) { this.text = full; onText(full); }
    }
  })();
}

// The cache holds keys and values for every position read so far; keep the first `length`.
function crop(lib, pkv, length) {
  const parts = {};
  for (const [name, tensor] of Object.entries(pkv)) parts[name] = tensor.slice(null, null, [0, length], null);
  return new lib.DynamicCache(parts);
}

async function generate(kind, ids, s, onText) {
  const lib = await loadLibrary();
  const { tokenizer, model } = await load(kind);
  if (active.stopped) return "";

  // A follow-up prompt repeats the previous one plus the reply and the new message, so
  // the attention cache of the last generation already covers most of it, and only the
  // part after the shared start is read again. Without this, every turn of a long chat
  // re-reads the whole conversation, which is what makes a slow device feel stuck.
  let pkv = null;
  let reused = 0;
  const cached = caches[kind];
  delete caches[kind];                            // generate() mutates the cache it is given
  if (REUSE && cached) {
    reused = Math.min(commonPrefixLength(cached.ids, ids), ids.length - 1);
    if (reused > 0) pkv = reused === cached.ids.length ? cached.pkv : crop(lib, cached.pkv, reused);
    else reused = 0;
  }

  const greedy = s.temperature === 0;
  const Rule = makeRule(lib.LogitsProcessor);
  const rule = new Rule(ids.length, {
    topK: greedy ? 0 : s.top_k, topP: greedy ? 0 : s.top_p, repetitionPenalty: s.repetition_penalty,
  });
  active.stopper = new lib.InterruptableStoppingCriteria();
  const t0 = performance.now();
  let firstAt = null;
  const streamer = makeStreamer(lib.BaseStreamer, tokenizer, onText, () => { firstAt = performance.now(); });

  let out;
  try {
    out = await model.generate({
      inputs: new lib.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]),
      attention_mask: new lib.Tensor("int64", new BigInt64Array(ids.length).fill(1n), [1, ids.length]),
      max_new_tokens: s.max_new_tokens,
      do_sample: !greedy,
      temperature: greedy ? 1 : s.temperature,
      // the library's own top-k only narrows what its sampler sorts (the rule has already
      // applied it); its repetition penalty would also punish the prompt, so it is off
      top_k: greedy ? 0 : s.top_k,
      repetition_penalty: 1,
      logits_processor: [rule],
      eos_token_id: EOT_ID,
      streamer,
      stopping_criteria: active.stopper,
      ...(pkv ? { past_key_values: pkv } : {}),
      return_dict_in_generate: true,
    });
  } catch (e) {
    release(kind);                                // a failed session may be unusable: reload next time
    throw new Error(`the model stopped with an error (${e.message}). If this keeps happening, `
      + "the device may be short of memory: close other tabs and try again.");
  }

  const seq = Array.from(out.sequences.data, Number);
  // the last sampled token was never fed back, so the cache covers everything before it
  if (REUSE && out.past_key_values?.get_seq_length() === seq.length - 1) {
    caches[kind] = { ids: seq.slice(0, -1), pkv: out.past_key_values };
  }

  const n = streamer.ids.length;
  const end = performance.now();
  stats.runs.push({
    kind, promptIds: ids, promptTokens: ids.length, reusedTokens: reused, newTokens: n,
    firstTokenMs: firstAt === null ? null : Math.round(firstAt - t0), totalMs: Math.round(end - t0),
  });
  const rate = n > 1 && firstAt !== null ? (n - 1) / ((end - firstAt) / 1000) : null;
  status(`${n} tokens` + (rate ? `, ${rate.toFixed(1)} tokens/s` : "")
    + (firstAt !== null ? `; the first after ${((firstAt - t0) / 1000).toFixed(1)} s` : "")
    + " (all on your CPU)");
  return streamer.text;
}

function setBusy(kind) {
  active = kind ? { kind, stopper: null, stopped: false } : null;
  $("chat-send").disabled = !!kind;
  $("complete-go").disabled = !!kind;
  $("chat-stop").disabled = kind !== "chat";
  $("complete-stop").disabled = kind !== "base";
  document.querySelectorAll(".example").forEach((b) => { b.disabled = !!kind; });
}

function stop(kind) {
  if (active?.kind !== kind) return;
  active.stopped = true;
  active.stopper?.interrupt();
}

// ---- chat tab ----
const log = [];        // [{role, text}] as shown on the page
let chatEpoch = 0;     // bumped by Clear, so a reply still being written can't leak into a new chat
let chatFloor = 0;     // index of the oldest log entry the model still sees

// The prompt for the current log. When the conversation outgrows the context, dropping
// the oldest turn on every message (what buildChatIds' maxTokens does on its own) would
// change the start of the prompt each time, so the attention cache could never be reused
// once the context is full and every reply would re-read ~800 tokens. Instead the window
// jumps forward to a user message, far enough that what remains fills at most half of the
// budget; the next few messages then extend the same prompt and reuse its cache. The
// prompt is still exactly chat.py's format for the turns it contains.
function windowIds(tokenizer, budget) {
  let ids = buildChatIds(toTurns(log.slice(chatFloor)), tokenizer, SYSTEM_DEFAULT);
  if (ids.length <= budget) return ids;
  for (let i = chatFloor + 1; i < log.length; i++) {
    if (log[i].role !== "user") continue;
    chatFloor = i;
    ids = buildChatIds(toTurns(log.slice(i)), tokenizer, SYSTEM_DEFAULT);
    if (ids.length <= budget / 2) return ids;
  }
  // the newest messages alone are still too long: buildChatIds keeps what fits, ending
  // with the end of the newest message
  return buildChatIds(toTurns(log.slice(chatFloor)), tokenizer, SYSTEM_DEFAULT, budget);
}

function follow() {
  const el = $("chat-log");
  if (el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
}

function bubble(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  $("chat-log").appendChild(div);
  $("chat-log").scrollTop = $("chat-log").scrollHeight;
  return div;
}

async function send(message) {
  if (active || !message.trim()) return;
  const epoch = chatEpoch;
  $("chat-log").querySelectorAll(".placeholder, .examples").forEach((el) => el.remove());
  log.push({ role: "user", text: message });
  bubble("user", message.trim());
  const reply = bubble("assistant", "…");
  reply.setAttribute("aria-busy", "true");      // screen readers hear the reply once, when done
  setBusy("chat");
  const s = settings();
  try {
    const { tokenizer } = await load("chat");
    if (epoch !== chatEpoch) return;
    const ids = windowIds(tokenizer, BLOCK_SIZE - s.max_new_tokens);
    const text = await generate("chat", ids, s, (t) => { reply.textContent = t; follow(); });
    if (epoch !== chatEpoch) { delete caches.chat; return; }
    if (text.trim()) {
      log.push({ role: "assistant", text });
    } else {
      reply.textContent = active.stopped ? "(Stopped.)" : EMPTY_TURN;
      reply.classList.add("notice");
    }
  } catch (e) {
    reply.textContent = `Error: ${e.message}`;
    reply.classList.add("notice");
  } finally {
    reply.removeAttribute("aria-busy");
    setBusy(null);
  }
}

$("chat-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = $("chat-input").value;
  if (active || !text.trim()) return;            // keep the draft while a reply is running
  $("chat-input").value = "";
  send(text);
});
$("chat-input").addEventListener("keydown", (e) => {
  // Enter sends and Shift+Enter starts a new line; an Enter that confirms an IME
  // composition (Chinese, Japanese or Korean input) must do neither
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing && e.keyCode !== 229) {
    e.preventDefault();
    $("chat-form").requestSubmit();
  }
});
document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => send(b.textContent)));
$("chat-stop").addEventListener("click", () => stop("chat"));
$("chat-clear").addEventListener("click", () => {
  chatEpoch += 1;
  stop("chat");
  log.length = 0;
  chatFloor = 0;
  delete caches.chat;
  $("chat-log").replaceChildren();
});

// ---- completion tab ----
$("complete-go").addEventListener("click", async () => {
  const prompt = cleanCompletionPrompt($("complete-input").value);
  if (active || !prompt.trim()) return;
  const out = $("complete-output");
  out.textContent = prompt;
  setBusy("base");
  const s = settings();
  try {
    const { tokenizer } = await load("base");
    const ids = keepEnd(encodeText(tokenizer, prompt), BLOCK_SIZE - s.max_new_tokens, tokenizer);
    await generate("base", ids, s, (t) => { out.textContent = prompt + t; });
  } catch (e) {
    out.textContent += `\n\nError: ${e.message}`;
  } finally {
    setBusy(null);
  }
});
$("complete-stop").addEventListener("click", () => stop("base"));

// ---- tabs and settings ----
for (const [tab, panel, kind] of [["tab-chat", "panel-chat", "chat"], ["tab-complete", "panel-complete", "base"]]) {
  $(tab).addEventListener("click", () => {
    document.querySelectorAll('[role="tab"]').forEach((t) => t.setAttribute("aria-selected", String(t.id === tab)));
    document.querySelectorAll('[role="tabpanel"]').forEach((p) => { p.hidden = p.id !== panel; });
    if (!active && !unsupported()) load(kind);
  });
}
for (const id of ["temperature", "top_p", "top_k", "repetition_penalty", "max_new_tokens"]) {
  $(id).addEventListener("input", () => { $(`${id}-v`).textContent = $(id).value; });
}

// ---- start ----
const reason = unsupported();
if (reason) {
  status(`Sorry: ${reason}. A recent Chrome, Edge, Firefox or Safari can run this page.`);
  document.querySelectorAll("button, textarea").forEach((el) => { el.disabled = true; });
} else if (navigator.connection?.saveData) {
  status(`Data saver is on, so nothing downloads until you send a message (the model is about ${MODEL_MB} MB).`);
} else {
  load("chat");
}
