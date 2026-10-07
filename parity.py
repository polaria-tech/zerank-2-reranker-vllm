"""Parity check: vLLM (/score, /rerank) vs the official CrossEncoder.predict path (reference.json).

Checks (non-zero exit status if any fails):
  1. Prompt: token ids built by vLLM == those of the official path.
     - every pair: usage.prompt_tokens from /score (one request per pair) == len(ST ids);
     - one pair: exact ids via /tokenize, using the same server-side template as /score.
     This is the only check that catches a silently wrong template. Both halves are needed:
     without --chat-template, /tokenize falls back to the tokenizer's chat_template.jinja and
     still matches, while /score ignores that file and concatenates query + document.
  2. 0-1 scores and reconstructed logits (5 * logit(s)) against fp32, with a tolerance relative
     to the noise floor of the official path (ST bf16 vs fp32), overall and per length bucket.
     Deviations from "clean" (HF bf16 + fp32 head) and from ST bf16 are informational only:
     they include the HF bf16 backbone noise on top of vLLM's.
  3. vLLM self-consistency: single pair vs batch of very different lengths, /score vs /rerank.
  4. Per-query ranking (/rerank): no inversion between documents whose fp32 logits differ by
     more than 2x the noise floor (near-ties may legitimately swap in bf16).

    python parity.py [--url http://localhost:8000] [--ref reference.json]
"""

import argparse
import json
import math
import sys

import requests

# Relative criterion: vLLM may deviate from fp32 by up to FLOOR_MULT x the deviation of the
# official path (ST bf16) from fp32, measured on the same pairs.
FLOOR_MULT = 1.5
BUCKETS = [(0, 64), (64, 512), (512, 2048), (2048, 8192), (8192, 24576), (24576, 10**9)]


def logit_from_score(s: float) -> float:
    s = min(max(s, 1e-12), 1 - 1e-12)
    return 5 * math.log(s / (1 - s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--ref", default="reference.json")
    ap.add_argument("--model", default="zerank-2-seq-cls")
    args = ap.parse_args()
    ref = json.load(open(args.ref))["pairs"]
    url, model = args.url.rstrip("/"), args.model
    failures = []

    def post(path, body):
        r = requests.post(url + path, json={"model": model, **body}, timeout=600)
        if r.status_code != 200:
            sys.exit(f"{path} -> HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    # 1a. One /score request per pair: unbatched score + prompt token count.
    single = []
    tok_mismatch = []
    for r in ref:
        out = post("/score", {"text_1": r["query"], "text_2": r["document"]})
        single.append(out["data"][0]["score"])
        if out["usage"]["prompt_tokens"] != r["n_tokens"]:
            tok_mismatch.append((r["n_tokens"], out["usage"]["prompt_tokens"], r["document"][:40]))
    print(f"[prompt] vLLM prompt_tokens == ST on {len(ref) - len(tok_mismatch)}/{len(ref)} pairs")
    for m in tok_mismatch[:5]:
        print(f"   ST {m[0]} vs vLLM {m[1]}: {m[2]!r}")
    if tok_mismatch:
        failures.append("prompt token count mismatch")

    # 1b. Exact ids for one pair, via /tokenize with the server's template.
    r0 = ref[0]
    tk = requests.post(url + "/tokenize", json={
        "model": model,
        "messages": [{"role": "query", "content": r0["query"]}, {"role": "document", "content": r0["document"]}],
        "add_generation_prompt": False, "add_special_tokens": False,
    }, timeout=60)
    if tk.status_code == 200 and tk.json()["tokens"] == r0["token_ids"]:
        print(f"[prompt] /tokenize: ids identical to ST ({len(r0['token_ids'])} tokens)")
    else:
        print(f"[prompt] /tokenize: MISMATCH. HTTP {tk.status_code} {tk.text[:300]}\n   ST: {r0['token_ids']}")
        failures.append("/tokenize token ids mismatch")

    # 2/3. One /rerank and one /score request per query group: each is a batch mixing very
    #      different document lengths (a few tokens -> several thousand).
    batched = [None] * len(ref)
    groups: dict[str, list[int]] = {}
    for i, r in enumerate(ref):
        groups.setdefault(r["query"], []).append(i)
    rerank_order = {}
    score_vs_rerank = 0.0
    for q, idx in groups.items():
        docs = [ref[i]["document"] for i in idx]
        out = post("/rerank", {"query": q, "documents": docs})
        for res in out["results"]:
            batched[idx[res["index"]]] = res["relevance_score"]
        rerank_order[q] = [idx[res["index"]] for res in out["results"]]
        sc = post("/score", {"text_1": q, "text_2": docs})
        for d in sc["data"]:
            score_vs_rerank = max(score_vs_rerank, abs(
                logit_from_score(d["score"]) - logit_from_score(batched[idx[d["index"]]])))

    # Noise floor: deviation of the official path (ST bf16, batch 1) from fp32. A real bug
    # (template, token, padding, activation) yields errors of several logit units.
    floor_l = max(abs(r["logit_bf16"] - r["logit_fp32"]) for r in ref)
    floor_s = max(abs(r["score_bf16"] - r["score_fp32"]) for r in ref)
    tol_l, tol_s = FLOOR_MULT * floor_l, FLOOR_MULT * floor_s
    print(f"\nNoise floor ST-bf16 vs fp32: logit {floor_l:.4f}, score {floor_s:.5f}"
          f" -> {FLOOR_MULT}x tolerances: logit {tol_l:.4f}, score {tol_s:.5f}")

    def dev(vals, key):
        return [abs(logit_from_score(v) - r[key]) for v, r in zip(vals, ref)]

    def line(name, dl, ds=None):
        i = max(range(len(dl)), key=dl.__getitem__)
        out = f"  {name:<22} logit max {max(dl):.4f} mean {sum(dl) / len(dl):.4f} (worst: {ref[i]['n_tokens']} tokens)"
        return out + (f" | score max {max(ds):.5f}" if ds else "")

    st_l = [abs(r["logit_bf16"] - r["logit_fp32"]) for r in ref]
    for name, vals in (("vLLM single", single), ("vLLM batch", batched)):
        dl = dev(vals, "logit_fp32")
        ds = [abs(v - r["score_fp32"]) for v, r in zip(vals, ref)]
        bad = max(dl) > tol_l or max(ds) > tol_s
        print(line(f"{name} vs fp32", dl, ds) + ("  <-- OUT OF TOLERANCE" if bad else ""))
        if bad:
            failures.append(f"{name} vs fp32 out of tolerance")
    print(line("ST bf16 vs fp32", st_l))
    print("  informational (includes HF bf16 backbone noise):")
    print(line("vLLM batch vs clean", dev(batched, "logit_clean")))
    print(line("vLLM batch vs ST bf16", dev(batched, "logit_bf16")))

    # Per length bucket: noise grows with sequence length.
    print("\n  by length (logit max vs fp32)        vLLM     ST bf16   n")
    dlb = dev(batched, "logit_fp32")
    for lo, hi in BUCKETS:
        hi_s = "+" if hi >= 10**9 else str(hi)
        idx = [i for i, r in enumerate(ref) if lo <= r["n_tokens"] < hi]
        if not idx:
            continue
        v, st = max(dlb[i] for i in idx), max(st_l[i] for i in idx)
        flag = ""
        if v > tol_l:
            flag = "  <-- OUT OF TOLERANCE"
            failures.append(f"bucket {lo}-{hi_s} out of tolerance")
        elif v > FLOOR_MULT * st:
            flag = "  (vLLM drifts more than ST in this bucket, worth watching)"
        print(f"  {lo:>5}-{hi_s:<6} tokens                 {v:.4f}   {st:.4f}   {len(idx)}{flag}")

    # vLLM self-consistency: single vs batch, /score vs /rerank. bf16 inference is not
    # batch-invariant, but the gap must stay below the noise floor.
    self_d = max(abs(logit_from_score(a) - logit_from_score(b)) for a, b in zip(single, batched))
    print(f"\n[vLLM single vs batch] logit max {self_d:.4f}; [/score vs /rerank] logit max {score_vs_rerank:.4f}")
    if max(self_d, score_vs_rerank) > floor_l:
        failures.append("vLLM inconsistent with itself beyond the noise floor")

    # Ranking: only inversions between documents more than 2x the noise floor apart in fp32 count.
    inversions = []
    for q, order in rerank_order.items():
        pos = {i: k for k, i in enumerate(order)}
        for a in order:
            for b in order:
                if ref[a]["logit_fp32"] - ref[b]["logit_fp32"] > 2 * floor_l and pos[a] > pos[b]:
                    inversions.append((q, ref[a]["document"][:30], ref[b]["document"][:30]))
    top1 = sum(order[0] == max(groups[q], key=lambda i: ref[i]["logit_fp32"]) for q, order in rerank_order.items())
    print(f"[ranking] top-1 matches fp32: {top1}/{len(groups)}; inversions beyond 2x noise floor: {len(inversions)}")
    for inv in inversions[:5]:
        print("   ", inv)
    if inversions:
        failures.append("ranking inversion")

    print("\nRESULT:", "OK" if not failures else "FAIL -> " + "; ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
