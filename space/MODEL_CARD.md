---
license: mit
language:
  - en
library_name: transformers
pipeline_tag: text-generation
tags:
  - gpt2
  - from-scratch
  - onnx
  - transformers.js
datasets:
{{DATASETS_YAML}}
---

# {{TITLE}}

{{INTRO}} Part of **[{{REPO_DISPLAY}}]({{REPO_URL}})**, a PyTorch reproduction of
GPT-2 small trained from scratch on a single 6GB laptop GPU.
**[Try it in your browser]({{SPACE_URL}})**.

The architecture is exactly GPT-2 small: 12 layers, 12 heads, width 768, a 1024-token
context and the GPT-2 BPE tokenizer. The weights load into Hugging Face's
`GPT2LMHeadModel` unchanged.

## Files

| file | what it is |
|---|---|
| `model.safetensors` | full-precision (fp32) weights |
| `onnx/model_quantized.onnx` | per-channel int8 ONNX for the browser demo. Validation loss vs fp32: +{{Q8_DECODING}} generating token by token ({{Q8_DECODING_TOKENS}} FineWeb-Edu tokens), +{{Q8_DELTA}} over one forward pass ({{Q8_TOKENS}} tokens) |
| `training.json` | training step, validation loss and the training arguments |

## Use it

```python
from transformers import pipeline

generate = pipeline("text-generation", model="{{REPO_ID}}")
print(generate({{EXAMPLE_PROMPT}}, max_new_tokens=60)[0]["generated_text"])
```
{{CHAT_NOTE}}
In the browser, with [transformers.js](https://huggingface.co/docs/transformers.js):
`await AutoModelForCausalLM.from_pretrained("{{REPO_ID}}", { dtype: "q8" })`.

{{EVAL_SECTION}}

## Limitations

This is a 124M-parameter model trained on {{TOKENS}} tokens. It writes fluent English,
but it is frequently wrong, especially about facts, arithmetic and recent events. Its
only alignment is supervised fine-tuning on a small instruction dataset, so it can
produce incorrect or inappropriate text.
{{CHAT_LIMITATION}}
## Data and licences

- Pretraining: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  (ODC-By 1.0)
- Fine-tuning (chat model only): [databricks-dolly-15k](https://huggingface.co/datasets/databricks/databricks-dolly-15k),
  licensed CC BY-SA 3.0. Treat the chat weights as share-alike.
- Code: MIT, derived in part from Andrej Karpathy's
  [build-nanogpt](https://github.com/karpathy/build-nanogpt) (MIT)
