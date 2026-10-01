---
license: mit
language:
  - en
tags:
  - gpt2
  - text-generation
  - from-scratch
  - pytorch
datasets:
  - HuggingFaceFW/fineweb-edu
  - {{SFT_DATASET}}
pipeline_tag: text-generation
---

# GPT-2 (124M), trained from scratch

Weights for **[{{REPO_DISPLAY}}]({{REPO_URL}})**, a from-scratch PyTorch reproduction of
GPT-2 small. Try it in the browser: **[live demo]({{SPACE_URL}})**.

| folder | what it is |
|---|---|
| `base/` | pretrained on {{TOKENS}} tokens of FineWeb-Edu (validation loss {{BASE_VAL_LOSS}}) |
| `chat/` | `base/` after supervised fine-tuning on `{{SFT_DATASET}}` |

Each folder holds `model.safetensors` and `config.json`.

The architecture is exactly GPT-2 small: 12 layers, 12 heads, width 768, 1024-token
context and the GPT-2 BPE tokenizer. The token embedding (tied to the output head) is
padded from 50,257 to 50,304 rows for tensor-core-friendly shapes. The padding rows are
never sampled.

## Evaluation

Zero-shot, scored by comparing the model's likelihood of each answer choice (the GPT-2
and GPT-3 protocol). The baseline is OpenAI's GPT-2 124M, run through the same harness.

{{EVAL_TABLE}}

## Use it

```bash
git clone {{REPO_URL}}
cd {{REPO_NAME}}
pip install -r requirements-app.txt
hf download {{MODEL_REPO}} --local-dir weights
python chat.py --checkpoint weights/chat --mode chat        # or weights/base --mode complete
```

## Limitations

This is a 124M-parameter model trained on {{TOKENS}} tokens. It writes fluent English,
but it is frequently wrong, especially about facts, arithmetic and recent events. Its
only alignment is supervised fine-tuning on a small instruction dataset, so it can
produce incorrect or inappropriate text. It is a learning project, not a product.

## Data and licences

- Pretraining: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (ODC-By 1.0)
- Fine-tuning: [{{SFT_DATASET}}](https://huggingface.co/datasets/{{SFT_DATASET}})
- Code: MIT, derived in part from Andrej Karpathy's
  [build-nanogpt](https://github.com/karpathy/build-nanogpt) (MIT)
