---
title: GPT-2 (124M) from scratch
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: static
app_file: index.html
pinned: true
license: mit
short_description: GPT-2 small trained from scratch, running in your browser
models:
  - {{BASE_REPO}}
  - {{CHAT_REPO}}
# cross-origin isolation lets ONNX Runtime use several CPU threads (SharedArrayBuffer)
# when the page is opened directly; inside the Hub's frame it runs single-threaded
custom_headers:
  cross-origin-embedder-policy: require-corp
  cross-origin-opener-policy: same-origin
  cross-origin-resource-policy: cross-origin
---

# GPT-2 (124M), trained from scratch

The live demo for **[{{REPO_DISPLAY}}]({{REPO_URL}})**: a PyTorch reproduction of
GPT-2 small, pretrained from scratch on FineWeb-Edu on a single 6GB laptop GPU, then
fine-tuned for chat.

The model runs entirely in the visitor's browser (transformers.js, int8 ONNX on
WebAssembly). Nothing typed into the page is sent anywhere.
- **Chat** talks to [{{CHAT_REPO}}](https://huggingface.co/{{CHAT_REPO}}).
- **Complete text** shows the base model, [{{BASE_REPO}}](https://huggingface.co/{{BASE_REPO}}).
