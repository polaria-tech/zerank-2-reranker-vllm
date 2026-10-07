"""Reference scores from the official path: CrossEncoder("zeroentropy/zerank-2").predict.

Three references:
  - ST bf16: the official path as used in production. Its output goes through a bf16 lm_head,
    so the logit is rounded to the bf16 grid (0.03125 steps between 4 and 8);
  - "clean": same HF bf16 backbone, last hidden state x embed_tokens.weight[9454] in fp32,
    i.e. exactly what vLLM's fp32 head computes, without the output rounding;
  - fp32: numerical ground truth, used to measure the bf16 noise of the backbone. Computed as
    the fp32 HF backbone (batch of 1, no attention mask) x the 'Yes' row: identical to ST fp32
    (measured: 4e-6 logit) but uses the memory-efficient causal attention kernel, whereas ST's
    explicit mask makes fp32 sdpa materialize the full attention matrix (OOM at ~39k tokens).
This model's bf16 noise is significant (+/-0.1-0.3 logit depending on attention backend or
padding), so parity is judged against fp32, using ST bf16's own deviation as the yardstick.

Writes reference.json: for each pair, token ids, raw logits and the published 0-1 score
sigmoid(logit / 5).

    python reference.py [--model ./upstream] [--set main|long] [--out reference.json]
"""

import argparse
import contextlib
import gc
import json
import math

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sentence_transformers import CrossEncoder
from transformers import AutoModelForCausalLM
from transformers.integrations import sdpa_attention

from pairs import flat_pairs

YES_TOKEN_ID = 9454
# Model card values (bf16, unspecified GPU). Loose tolerance because of bf16 noise, see docstring.
CARD = [(("What is 2+2?", "4"), 5.4062), (("What is 2+2?", "The answer is definitely 1 million"), -4.5000)]


def sig5(x: float) -> float:
    return 1 / (1 + math.exp(-x / 5))


def run(model_path: str, dtype: torch.dtype, pairs, batch_size: int):
    model = CrossEncoder(model_path, model_kwargs={"dtype": dtype}, device="cuda")
    out = [float(x) for x in model.predict(pairs, batch_size=batch_size, convert_to_numpy=True)]
    tok = model.tokenizer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out, tok


def run_direct(model_path: str, dtype: torch.dtype, ids_list):
    """Hidden state at the last token (batch of 1, no padding) x 'Yes' row, product in fp32."""
    m = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype).cuda().eval()
    w = m.model.embed_tokens.weight[YES_TOKEN_ID].float()
    out = []
    if dtype == torch.float32:
        # Force memory-efficient attention: it supports fp32 and never materializes the
        # seq x seq matrix, so ~39k-token prompts fit. Errors out instead of silently falling back.
        # That kernel rejects enable_gqa, so make transformers expand K/V heads explicitly instead
        # (repeat_kv: same math). bf16 keeps the default kernel choice, i.e. the same as ST.
        orig_gqa = sdpa_attention.use_gqa_in_sdpa
        sdpa_attention.use_gqa_in_sdpa = lambda *args, **kwargs: False
        kernels = sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    else:
        orig_gqa, kernels = None, contextlib.nullcontext()
    with torch.no_grad(), kernels:
        for ids in ids_list:
            h = m.model(torch.tensor([ids], device="cuda"), use_cache=False).last_hidden_state[0, -1]
            out.append(float(h.float() @ w))
    if orig_gqa is not None:
        sdpa_attention.use_gqa_in_sdpa = orig_gqa
    del m
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./upstream")
    ap.add_argument("--set", choices=["main", "long"], default="main")
    ap.add_argument("--out", default=None, help="default: reference.json / reference_long.json")
    args = ap.parse_args()
    args.out = args.out or ("reference.json" if args.set == "main" else "reference_long.json")

    pairs = flat_pairs(args.set)

    card, _ = run(args.model, torch.bfloat16, [p for p, _ in CARD], 1)
    for (pair, expected), got in zip(CARD, card):
        print(f"card {pair!r}: bf16 {got:.4f}, card {expected:.4f}")
        assert abs(got - expected) < 0.25, "model card mismatch beyond bf16 noise"

    # batch_size=1: no padding, the cleanest path for ST.
    bf16, tok = run(args.model, torch.bfloat16, pairs, 1)
    # Batched pass (padding sensitivity). Skipped for the long set: ST materializes full-vocabulary
    # logits at every position, which does not fit in memory for 8 x ~39k tokens.
    bf16_b8 = run(args.model, torch.bfloat16, pairs, 8)[0] if args.set == "main" else [None] * len(pairs)

    # Token ids of the official path (original template, add_generation_prompt=True):
    # parity.py compares them with the prompt vLLM builds.
    ids_list = [tok.apply_chat_template(
        [{"role": "query", "content": q}, {"role": "document", "content": d}],
        add_generation_prompt=True, tokenize=True, return_dict=True,
    )["input_ids"] for q, d in pairs]
    clean = run_direct(args.model, torch.bfloat16, ids_list)
    fp32 = run_direct(args.model, torch.float32, ids_list)

    rows = []
    for (q, d), ids, lb, lb8, lf, lc in zip(pairs, ids_list, bf16, bf16_b8, fp32, clean):
        rows.append({"query": q, "document": d, "n_tokens": len(ids), "token_ids": ids,
                     "logit_bf16": lb, "logit_bf16_batch8": lb8, "logit_fp32": lf, "logit_clean": lc,
                     "score_bf16": sig5(lb), "score_clean": sig5(lc), "score_fp32": sig5(lf)})

    def mx(key_a, key_b):
        return max(abs(r[key_a] - r[key_b]) for r in rows)

    print(f"ST bf16 vs clean       : logit max {mx('logit_bf16', 'logit_clean'):.4f}")
    print(f"clean vs fp32          : logit max {mx('logit_clean', 'logit_fp32'):.4f}")
    print(f"ST bf16 vs fp32        : logit max {mx('logit_bf16', 'logit_fp32'):.4f}")
    if args.set == "main":
        print(f"ST bf16 b8 vs fp32     : logit max {mx('logit_bf16_batch8', 'logit_fp32'):.4f}")
        print(f"ST bf16 b1 vs b8       : logit max {mx('logit_bf16', 'logit_bf16_batch8'):.4f}")
    print(f"ST bf16 vs fp32 score  : max {mx('score_bf16', 'score_fp32'):.5f}")
    json.dump({"pairs": rows}, open(args.out, "w"), ensure_ascii=False, indent=1)
    print(f"{len(rows)} pairs -> {args.out} (tokens {min(r['n_tokens'] for r in rows)}..{max(r['n_tokens'] for r in rows)})")


if __name__ == "__main__":
    main()
