---
title: GPT-2 (124M) from scratch
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: {{SDK_VERSION}}
python_version: "3.12"
app_file: app.py
pinned: true
license: mit
short_description: GPT-2 small reproduced from scratch on one laptop GPU
{{PRELOAD}}---

# GPT-2 (124M), trained from scratch

The live demo for **[{{REPO_DISPLAY}}]({{REPO_URL}})**, a from-scratch PyTorch
reproduction of GPT-2 small. It was pretrained on FineWeb-Edu on a single 6GB laptop
GPU, then instruction-tuned for chat.

- **Chat** talks to the fine-tuned model.
- **Complete text** shows the base model straight out of pretraining.

Both run on this Space's CPU. Code, training logs and the write-up are on GitHub.
