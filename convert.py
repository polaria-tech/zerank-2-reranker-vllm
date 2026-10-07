"""Convert zeroentropy/zerank-2 (Qwen3ForCausalLM + LogitScore) into a Qwen3ForSequenceClassification
checkpoint that vLLM serves on /score and /rerank without --hf-overrides.

- score.weight [1, hidden] = embed_tokens.weight[9454] / 5, stored in fp32.
  (Tied embeddings: there is no lm_head.weight. Single "Yes" token, no negative token.)
  vLLM applies a sigmoid by default when num_labels=1, so score = sigmoid(logit / 5),
  the calibrated 0-1 score published by ZeroEntropy. Raw logit = 5 * logit(score).
  fp32 because vLLM runs pooling-model heads in fp32: no extra bf16 rounding of the weight.
- Converted shard by shard (never more than one shard in memory).
- No sentence-transformers files: they describe the original causal-LM pipeline
  (including activation_fn=Identity, which vLLM would use instead of the sigmoid).
- The score template ships with the checkpoint (score_template.jinja, also copied to
  chat_template.jinja); vLLM only applies it when passed with --chat-template.

    python convert.py [--src ./upstream] [--dst ./zerank-2-seq-cls]
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

YES_TOKEN_ID = 9454
TEMPERATURE = 5.0
TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                   "added_tokens.json", "special_tokens_map.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="./upstream")
    ap.add_argument("--dst", default="./zerank-2-seq-cls")
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    cfg = json.loads((src / "config.json").read_text())
    assert cfg["architectures"] == ["Qwen3ForCausalLM"] and cfg["tie_word_embeddings"] is True
    st_head = json.loads((src / "1_LogitScore" / "config.json").read_text())
    assert st_head == {"true_token_id": YES_TOKEN_ID, "false_token_id": None}, st_head

    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    assert not any(k.startswith("lm_head") for k in index["weight_map"])

    new_map, total_size, score = {}, 0, None
    for i, shard in enumerate(shards):
        tensors = {}
        with safe_open(src / shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                tensors[name] = t
                if name == "model.embed_tokens.weight":
                    row = t[YES_TOKEN_ID].float()
                    score = (row / TEMPERATURE).unsqueeze(0).contiguous()
        out_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        if i == len(shards) - 1:
            assert score is not None, "embed_tokens not found"
            tensors["score.weight"] = score
        save_file(tensors, dst / out_name, metadata={"format": "pt"})
        for name, t in tensors.items():
            new_map[name] = out_name
            total_size += t.numel() * t.element_size()
        print(f"{shard} -> {out_name} ({len(tensors)} tensors)")
        del tensors

    # Exact check: stored row * 5 gives back the original embedding row.
    with safe_open(dst / new_map["score.weight"], framework="pt") as f:
        stored = f.get_tensor("score.weight")
    with safe_open(src / index["weight_map"]["model.embed_tokens.weight"], framework="pt") as f:
        orig = f.get_slice("model.embed_tokens.weight")[YES_TOKEN_ID : YES_TOKEN_ID + 1].float()
    err = ((stored * TEMPERATURE - orig).abs().max() / orig.abs().max()).item()
    print(f"score.weight {tuple(stored.shape)} {stored.dtype}, max relative error vs embed[{YES_TOKEN_ID}]: {err:.2e}")
    assert stored.shape == (1, cfg["hidden_size"]) and err < 1e-6

    (dst / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total_size}, "weight_map": dict(sorted(new_map.items()))}, indent=2))

    cfg.update({
        "architectures": ["Qwen3ForSequenceClassification"],
        "num_labels": 1,
        "tie_word_embeddings": False,
        # The prompt is fully built by the template: vLLM must not insert a separator.
        "use_sep_token": False,
    })
    # Keys that would trigger an online re-conversion from lm_head (or change the activation) at load time.
    for k in ("method", "classifier_from_token", "is_original_qwen3_reranker", "use_pad_token",
              "problem_type", "sentence_transformers", "sbert_ce_default_activation_function"):
        assert k not in cfg, k
    (dst / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    for name in TOKENIZER_FILES + ["LICENSE"]:
        shutil.copy2(src / name, dst / name)
    here = Path(__file__).parent
    shutil.copy2(here / "score_template.jinja", dst / "score_template.jinja")
    shutil.copy2(here / "score_template.jinja", dst / "chat_template.jinja")
    print(f"OK -> {dst}")


if __name__ == "__main__":
    main()
