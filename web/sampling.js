// How the next token is chosen: chat.py's generate_stream rule, ported so the browser and
// the Python app behave the same. transformers.js 4.3 differs in two ways that matter
// for a chat model: its repetition penalty also covers the prompt (so the end-of-turn
// token and the visitor's own words become less likely), and it never applies top_p.
// app.js therefore turns both off and runs filterLogits as a logits processor instead.
// tests/test_web_sampling.py checks it against chat._filter_logits.

// k-th largest value of a Float32Array (quickselect on a copy; O(n) on average).
function kthLargest(values, k) {
  const a = Float32Array.from(values);
  let lo = 0;
  let hi = a.length - 1;
  const target = a.length - k;          // index of the k-th largest in ascending order
  while (lo < hi) {
    const pivot = a[(lo + hi) >> 1];
    let i = lo;
    let j = hi;
    while (i <= j) {
      while (a[i] < pivot) i++;
      while (a[j] > pivot) j--;
      if (i <= j) { const t = a[i]; a[i] = a[j]; a[j] = t; i++; j--; }
    }
    if (target <= j) hi = j;
    else if (target >= i) lo = i;
    else break;
  }
  return a[target];
}

// logits: Float32Array for one position (modified in place). generated: the ids sampled
// so far in this reply (never the prompt). topK 0 and topP 0 or 1 mean "off", as in
// chat.py; greedy decoding passes topK = topP = 0 and only the penalty applies.
export function filterLogits(logits, { topK = 0, topP = 0, repetitionPenalty = 1, generated = [] }) {
  if (repetitionPenalty !== 1 && generated.length) {
    // torch divides float32 logits by the penalty rounded to float32; doing the same
    // keeps every value bit-identical to chat.py
    const penalty = Math.fround(repetitionPenalty);
    for (const id of new Set(generated)) {
      const v = logits[id];
      // CTRL's rule: shrink positive logits, grow negative ones, so the penalty always
      // makes a token less likely
      logits[id] = v > 0 ? v / penalty : v * penalty;
    }
  }

  let keep = null;                      // candidate ids, best first, when a filter is on
  if (topK > 0 && topK < logits.length) {
    const kth = kthLargest(logits, topK);
    keep = [];
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] < kth) logits[i] = -Infinity;   // ties with the k-th value survive
      else keep.push(i);
    }
  }

  if (topP > 0 && topP < 1) {
    if (!keep) {
      keep = [];
      for (let i = 0; i < logits.length; i++) if (logits[i] > -Infinity) keep.push(i);
    }
    keep.sort((a, b) => logits[b] - logits[a]);
    const max = logits[keep[0]];
    let total = 0;
    for (const i of keep) total += Math.exp(logits[i] - max);
    // keep the first token that crosses p: a token is dropped only when the probability
    // mass BEFORE it already exceeds p, so the nucleus is never empty
    let before = 0;
    for (const i of keep) {
      const p = Math.exp(logits[i] - max) / total;
      if (before > topP) logits[i] = -Infinity;
      before += p;
    }
  }
  return logits;
}

// filterLogits as a transformers.js logits processor. The library's LogitsProcessor base
// class is passed in because the library itself is loaded lazily (see app.js).
// promptLength separates the prompt from the reply inside the ids the library passes.
export function makeRule(LogitsProcessor) {
  return class ChatPyRule extends LogitsProcessor {
    constructor(promptLength, options) {
      super();
      this.promptLength = promptLength;
      this.options = options;
    }

    _call(inputIds, logits) {
      for (let b = 0; b < inputIds.length; b++) {
        const generated = inputIds[b].slice(this.promptLength).map(Number);
        filterLogits(logits[b].data, { ...this.options, generated });
      }
      return logits;
    }
  };
}

// Length of the shared start of two id arrays: how much of a cached prompt still applies.
export function commonPrefixLength(a, b) {
  const n = Math.min(a.length, b.length);
  let i = 0;
  while (i < n && a[i] === b[i]) i++;
  return i;
}
