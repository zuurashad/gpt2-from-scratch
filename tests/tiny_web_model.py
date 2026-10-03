"""A tiny random GPT-2 (2 layers, 64 wide, the real 50,257-token vocabulary and 1,024
positions) packaged the way export_web.py packages the real model, for the browser tests.

Needs optimum + onnxruntime (requirements-web.txt) and, once, network access for the
GPT-2 tokenizer files."""

import os
import shutil
import tempfile

import torch

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")


def build_web_model(out_dir, seed=0, quantise=True):
    """Write <out_dir>/{config.json, tokenizer files, onnx/model_quantized.onnx}.

    quantise=True runs the real pipeline (int8 + optimise_for_web) and also keeps the
    int8 graph from before the rewrite as onnx/model_quantized.orig.onnx. quantise=False
    stores the fp32 graph under the same name: no dynamic quantisation means no
    activation scales that depend on how a prompt is split into chunks, so generation
    is deterministic enough to compare the page's cache reuse token for token."""
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.exporters.onnx import main_export
    from transformers import GPT2Config, GPT2LMHeadModel

    from export_web import optimise_for_web

    torch.manual_seed(seed)
    config = GPT2Config(n_positions=1024, n_embd=64, n_layer=2, n_head=2, bos_token_id=50256,
                        eos_token_id=50256)
    model = GPT2LMHeadModel(config).eval()
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    # end-of-text and whitespace-only tokens: a random model that picks them greedily
    # writes replies the page (rightly) shows as empty
    quiet = [50256] + [i for i in range(50256) if not enc.decode_single_token_bytes(i).strip()]
    with torch.no_grad():
        # spread the logits out, so greedy picks are clear-cut, and shrink the quiet tokens'
        # rows (tied input/output embedding) so they are never the most likely
        model.transformer.wte.weight.mul_(10)
        model.transformer.wte.weight[quiet] *= 0.01
    os.makedirs(os.path.join(out_dir, "onnx"), exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)
    for name in TOKENIZER_FILES:
        shutil.copy(hf_hub_download("openai-community/gpt2", name), os.path.join(out_dir, name))
    with tempfile.TemporaryDirectory() as tmp:
        main_export(out_dir, output=tmp, task="text-generation-with-past")
        fp32 = os.path.join(tmp, "model.onnx")
        target = os.path.join(out_dir, "onnx", "model_quantized.onnx")
        if quantise:
            quantize_dynamic(fp32, target, weight_type=QuantType.QInt8, per_channel=True)
            shutil.copy(target, os.path.join(out_dir, "onnx", "model_quantized.orig.onnx"))
            optimise_for_web(target)
        else:
            shutil.copy(fp32, target)
    os.remove(os.path.join(out_dir, "model.safetensors"))     # the page only reads the ONNX file
    return out_dir
