"""Quantize the bf16 seq-cls checkpoint (convert.py output) to FP8 with llm-compressor.

- Scheme FP8_DYNAMIC: weights FP8 E4M3 with one scale per output channel (round-to-nearest),
  activations FP8 with one scale per token computed at runtime. No calibration data, so no bias
  towards a language or a document length. Served natively by vLLM on Ada and Hopper.
- Only the Linear layers of the decoder are quantized. score.weight (embed[9454] / 5) is ignored
  and kept in fp32 like in the bf16 checkpoint: transformers loads it in bf16, so the original
  fp32 tensor is put back before saving. Embeddings and norms are not Linear: untouched.
- The model is loaded straight onto the GPU (~8 GB bf16), so little CPU RAM is needed.

    python quantize_fp8.py [--src ./zerank-2-seq-cls] [--dst ./zerank-2-seq-cls-fp8]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification

COPIED_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "added_tokens.json", "special_tokens_map.json", "LICENSE",
                "score_template.jinja", "chat_template.jinja"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./zerank-2-seq-cls")
    ap.add_argument("--dst", default="./zerank-2-seq-cls-fp8")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)

    src_cfg = json.loads((src / "config.json").read_text())
    assert src_cfg["architectures"] == ["Qwen3ForSequenceClassification"] and src_cfg["num_labels"] == 1
    index = json.loads((src / "model.safetensors.index.json").read_text())
    with safe_open(src / index["weight_map"]["score.weight"], framework="pt") as f:
        score = f.get_tensor("score.weight")
    assert score.dtype == torch.float32 and score.shape == (1, src_cfg["hidden_size"])

    model = AutoModelForSequenceClassification.from_pretrained(src, dtype=torch.bfloat16, device_map="cuda")
    oneshot(model=model, recipe=QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["score"]))
    assert model.score.weight.dtype == torch.bfloat16 and not hasattr(model.score, "quantization_scheme")
    model.score.weight.data = score.to(model.score.weight.device)

    model.save_pretrained(dst, save_compressed=True)
    for name in COPIED_FILES:
        shutil.copy2(src / name, dst / name)

    # save_pretrained rewrites config.json: check that only the quantization config was added.
    cfg = json.loads((dst / "config.json").read_text())
    changed = {k for k in src_cfg.keys() | cfg.keys() if src_cfg.get(k) != cfg.get(k)}
    print(f"config keys changed by save_pretrained: {sorted(changed)}")

    out_index = dst / "model.safetensors.index.json"
    score_file = (json.loads(out_index.read_text())["weight_map"]["score.weight"]
                  if out_index.exists() else "model.safetensors")
    with safe_open(dst / score_file, framework="pt") as f:
        stored = f.get_tensor("score.weight")
        assert stored.dtype == torch.float32 and torch.equal(stored, score)
    print(f"OK -> {dst}")


if __name__ == "__main__":
    main()
