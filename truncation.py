"""What happens when a (query, document) pair does not fit in the context?

zerank-2 reads its score at the last prompt token, which must be the template's
"<|im_start|>assistant\\n" suffix. Any truncation that cuts the end of the prompt (or the
query at the start) silently produces a wrong score. This script sends an over-long document
with each truncation option and checks the outcome is either an explicit error or a score
equal to scoring the properly truncated document (template intact).

It also checks a vLLM scheduler bug (0.29-0.31): a prompt of exactly max-model-len tokens is
never scheduled, because the scheduler reserves one slot for a sampled token even for pooling
models. The request hangs with no error; truncate_prompt_tokens=-1 truncates to exactly that
length. Each request therefore has a timeout, and a hang counts as a failure.

"Properly truncated" reference: the document cut to its first K tokens on the client side,
then sent as a normal, untruncated request. A matching score (within the bf16 noise floor)
means vLLM truncated the document only.

    python truncation.py [--url http://localhost:8000] [--max-model-len 40960]
"""

import argparse
import math
import sys

import requests
from transformers import AutoTokenizer

from pairs import long_text, _doc

NOISE = 0.25  # logit; bf16 single-vs-batch and kernel noise measured by parity.py is below this
TIMEOUT = 60  # seconds; a ~40k-token prompt takes under 2 s cold on an H100


def logit(s: float) -> float:
    s = min(max(s, 1e-12), 1 - 1e-12)
    return 5 * math.log(s / (1 - s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="zerank-2-seq-cls")
    ap.add_argument("--tokenizer", default="./zerank-2-seq-cls")
    ap.add_argument("--max-model-len", type=int, default=40960)
    args = ap.parse_args()
    url, model, L = args.url.rstrip("/"), args.model, args.max_model_len
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    query = "How do I add subcommands, each with their own arguments, to a command-line parser?"
    # Relevant content first, so a correct (document-tail) truncation keeps it.
    doc = long_text(L + 8000, _doc("argparse"), "start")
    doc_ids = tok(doc, add_special_tokens=False)["input_ids"]
    overhead = len(tok.apply_chat_template(
        [{"role": "query", "content": query}, {"role": "document", "content": ""}], tokenize=True,
        return_dict=True)["input_ids"])
    print(f"document {len(doc_ids)} tokens, template + query {overhead} tokens, max-model-len {L}\n", flush=True)

    def call(path, **extra):
        body = {"model": model, **extra}
        if path == "/score":
            body.update(text_1=query, text_2=doc if "text_2" not in extra else extra["text_2"])
        else:
            body.update(query=query, documents=[doc])
        try:
            r = requests.post(url + path, json=body, timeout=TIMEOUT)
        except requests.exceptions.Timeout:
            return None, None, "HANG"
        if r.status_code != 200:
            return None, None, f"HTTP {r.status_code}: {r.json().get('error', {}).get('message', r.text)[:160]}"
        j = r.json()
        if path == "/score":
            return j["data"][0]["score"], j["usage"]["prompt_tokens"], None
        return j["results"][0]["relevance_score"], j["usage"]["prompt_tokens"] if "usage" in j else None, None

    def expected(doc_tokens: int) -> float:
        cut = tok.decode(doc_ids[:doc_tokens])
        r = requests.post(url + "/score", json={"model": model, "text_1": query, "text_2": cut}, timeout=600)
        r.raise_for_status()
        return r.json()["data"][0]["score"]

    cases = [
        # (name, endpoint, request params, document tokens kept if vLLM truncates the document only)
        ("no truncation params", "/score", {}, None),
        ("truncate_prompt_tokens=-1", "/score", {"truncate_prompt_tokens": -1}, L - overhead),
        (f"truncate_prompt_tokens={L - 1}", "/score", {"truncate_prompt_tokens": L - 1}, L - 1 - overhead),
        ("truncate_prompt_tokens=8192", "/score", {"truncate_prompt_tokens": 8192}, 8192 - overhead),
        ("truncate_prompt_tokens=8192, side=left", "/score",
         {"truncate_prompt_tokens": 8192, "truncation_side": "left"}, 8192 - overhead),
        ("max_tokens_per_doc=8000", "/score", {"max_tokens_per_doc": 8000}, 8000),
        ("max_tokens_per_doc=8000", "/rerank", {"max_tokens_per_doc": 8000}, 8000),
        (f"truncate_prompt_tokens={L - 1}", "/rerank", {"truncate_prompt_tokens": L - 1}, L - 1 - overhead),
    ]
    silent_wrong, hangs = [], []
    print(f"{'case':<42}{'endpoint':<9}{'outcome':<36}{'logit':>8}{'expected':>10}")
    for name, path, params, keep in cases:
        score, n_prompt, err = call(path, **params)
        if err == "HANG":
            print(f"{name:<42}{path:<9}HANG: no response after {TIMEOUT} s (request stays 'running')", flush=True)
            hangs.append(f"{path} {name}")
            continue
        if err:
            print(f"{name:<42}{path:<9}{err}", flush=True)
            continue
        exp = logit(expected(keep)) if keep else None
        got = logit(score)
        ok = exp is not None and abs(got - exp) <= NOISE
        verdict = "document-only truncation" if ok else "WRONG SCORE (template cut)"
        print(f"{name:<42}{path:<9}{verdict:<36}{got:>8.3f}{exp if exp is None else round(exp, 3):>10}"
              f"   prompt_tokens={n_prompt}", flush=True)
        if not ok:
            silent_wrong.append(f"{path} {name}")

    print("\nRESULT:", "no silently wrong score" if not silent_wrong
          else "silently wrong scores -> " + "; ".join(silent_wrong))
    if hangs:
        print("HANGS:", "; ".join(hangs))
    sys.exit(1 if silent_wrong or hangs else 0)


if __name__ == "__main__":
    main()
