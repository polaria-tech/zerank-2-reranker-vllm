"""Benchmark: one query + N documents per request (a rerank call).

Engines (run one at a time, they each want the GPU):
  st       sentence-transformers CrossEncoder.predict, bf16 (the official path), batch-size sweep
  http     vLLM /rerank over HTTP against a running server, at several concurrency levels
  offline  vLLM LLM.score in-process (no HTTP), to isolate the server/HTTP overhead
  report   print a table of all results in bench_results/

Workload (deterministic): each request has a unique query and N documents of a random token
length in [--doc-min, --doc-max] (default 300-500), cut from Python stdlib pydoc text. Queries are
"short" (~15 tokens, typical search queries) or "long" (~300 tokens), measured separately: the
query sits at the head of every prompt of a rerank call, so only long queries span full
prefix-cache blocks (16 tokens) that the N documents can share. Unique queries per request mean
no cross-request cache hits, as in real traffic.

    python bench.py st --batch-sizes 16,32,64,128
    python bench.py http --label default --server-log serve.log --concurrency 1,8,32
    python bench.py offline --label offline --no-prefix-caching
    python bench.py report
"""

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MODEL_DIR = "./zerank-2-seq-cls"
UPSTREAM = "./upstream"
RESULTS = Path("bench_results")

SHORT_QUESTIONS = [
    "how to parse command line arguments", "read a json file into a dict", "rotate log files daily",
    "send an email with attachment", "difference between deque and list", "make an http request with headers",
    "walk a directory tree recursively", "run a shell command and capture output", "compare dates with timezones",
    "assert that an exception is raised in a test", "count word frequencies", "parse an http response status",
]


# ---------------------------------------------------------------------------------------- workload

def make_workload(n_requests: int, n_docs: int, doc_min: int, doc_max: int, query_kind: str, seed: int):
    """Returns [(query, [docs]), ...] and the total prompt tokens of each request."""
    import importlib
    import pydoc
    import re

    from transformers import AutoTokenizer

    # model_max_length raised: the text pool is tokenized in one go, documents are cut from it.
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, model_max_length=10**9)
    mods = ["logging", "pathlib", "http.client", "collections", "email.message", "unittest.case", "argparse",
            "json", "subprocess", "datetime", "re", "asyncio.tasks", "typing", "inspect", "socket", "tarfile",
            "zipfile", "csv", "decimal", "functools", "itertools", "threading", "multiprocessing.pool"]
    text = "\n\n".join(re.sub(r" at 0x[0-9a-f]+", "", pydoc.render_doc(importlib.import_module(m),
                                                                      renderer=pydoc.plaintext)) for m in mods)
    pool = tok(text, add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)

    def chunk(n):
        start = rng.randrange(0, len(pool) - n)
        return tok.decode(pool[start:start + n])

    requests = []
    for i in range(n_requests):
        q = f"{rng.choice(SHORT_QUESTIONS)} (case {seed}-{i})"
        if query_kind == "long":
            q = f"{q}. Context from the user's ticket: {chunk(280)}"
        docs = [chunk(rng.randint(doc_min, doc_max)) for _ in range(n_docs)]
        requests.append((q, docs))
    # Prompt sizes through the real template, for tokens/s.
    tokens = []
    for q, docs in requests:
        tokens.append(sum(len(tok.apply_chat_template(
            [{"role": "query", "content": q}, {"role": "document", "content": d}],
            tokenize=True, return_dict=True)["input_ids"]) for d in docs))
    return requests, tokens


def summarize(latencies, wall, n_pairs, n_tokens):
    lat = sorted(latencies)
    return {
        "requests": len(lat), "p50_s": statistics.median(lat),
        "p95_s": lat[min(len(lat) - 1, int(round(0.95 * (len(lat) - 1))))],
        "mean_s": statistics.fmean(lat), "wall_s": wall,
        "pairs_per_s": n_pairs / wall, "tokens_per_s": n_tokens / wall,
    }


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def save(name, payload):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=1))
    print(f"-> {RESULTS / name}.json")


# ---------------------------------------------------------------------------------------- engines

def bench_st(args):
    import torch
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(UPSTREAM, model_kwargs={"dtype": torch.bfloat16}, device="cuda")
    rows = []
    for kind in args.query_kinds:
        reqs, toks = make_workload(args.requests + args.warmup, args.docs, args.doc_min, args.doc_max, kind, args.seed)
        for bs in args.batch_sizes:
            for q, docs in reqs[:args.warmup]:
                model.predict([(q, d) for d in docs], batch_size=bs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            lat = []
            t0 = time.perf_counter()
            for q, docs in reqs[args.warmup:]:
                t = time.perf_counter()
                model.predict([(q, d) for d in docs], batch_size=bs)
                lat.append(time.perf_counter() - t)
            wall = time.perf_counter() - t0
            s = summarize(lat, wall, args.requests * args.docs, sum(toks[args.warmup:]))
            s.update(engine="st", query=kind, batch_size=bs, concurrency=1,
                     gpu_peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30)
            rows.append(s)
            print(f"st  {kind:<5} bs={bs:<4} p50 {s['p50_s']:.3f}s p95 {s['p95_s']:.3f}s "
                  f"{s['pairs_per_s']:.0f} pairs/s {s['tokens_per_s']:.0f} tok/s peak {s['gpu_peak_alloc_gib']:.1f} GiB",
                  flush=True)
    save(args.label or "st", {"engine": "st", "rows": rows})


def bench_http(args):
    import requests as rq

    url = args.url.rstrip("/") + "/rerank"
    actual_mem = None
    if args.server_log:
        for line in Path(args.server_log).read_text().splitlines():
            if "Actual usage is" in line:
                actual_mem = line.split("Actual usage is", 1)[1].strip()
    rows = []
    for kind in args.query_kinds:
        for c in args.concurrency:
            n = max(args.requests, 4 * c)
            # A fresh seed per cell: no request (hence no prefix) is ever repeated across cells.
            reqs, toks = make_workload(n + args.warmup, args.docs, args.doc_min, args.doc_max, kind,
                                       args.seed + 1000 * c + (0 if kind == "short" else 1))
            sessions = {}

            def one(item):
                import threading
                s = sessions.setdefault(threading.get_ident(), rq.Session())
                q, docs = item
                t = time.perf_counter()
                r = s.post(url, json={"model": args.model, "query": q, "documents": docs}, timeout=600)
                r.raise_for_status()
                return time.perf_counter() - t

            with ThreadPoolExecutor(c) as ex:
                list(ex.map(one, reqs[:args.warmup]))
                t0 = time.perf_counter()
                lat = list(ex.map(one, reqs[args.warmup:]))
                wall = time.perf_counter() - t0
            s = summarize(lat, wall, n * args.docs, sum(toks[args.warmup:]))
            s.update(engine="http", query=kind, concurrency=c, gpu_used_mib=gpu_used_mib())
            rows.append(s)
            print(f"http {kind:<5} c={c:<3} p50 {s['p50_s']:.3f}s p95 {s['p95_s']:.3f}s "
                  f"{s['pairs_per_s']:.0f} pairs/s {s['tokens_per_s']:.0f} tok/s", flush=True)
    save(args.label, {"engine": "http", "server_args": args.server_args, "server_memory": actual_mem, "rows": rows})


def bench_offline(args):
    from vllm import LLM

    template = Path(MODEL_DIR, "score_template.jinja").read_text()
    kw = {"max_num_batched_tokens": args.max_num_batched_tokens} if args.max_num_batched_tokens else {}
    llm = LLM(MODEL_DIR, runner="pooling", dtype="bfloat16", enable_prefix_caching=not args.no_prefix_caching, **kw)
    rows = []
    for kind in args.query_kinds:
        reqs, toks = make_workload(args.requests + args.warmup, args.docs, args.doc_min, args.doc_max, kind,
                                   args.seed + 7 + (0 if kind == "short" else 1))
        for q, docs in reqs[:args.warmup]:
            llm.score(q, docs, chat_template=template, use_tqdm=False)
        # Latency: one rerank call at a time (comparable to HTTP concurrency 1).
        lat = []
        t0 = time.perf_counter()
        for q, docs in reqs[args.warmup:]:
            t = time.perf_counter()
            llm.score(q, docs, chat_template=template, use_tqdm=False)
            lat.append(time.perf_counter() - t)
        wall = time.perf_counter() - t0
        s = summarize(lat, wall, args.requests * args.docs, sum(toks[args.warmup:]))
        s.update(engine="offline", query=kind, concurrency=1)
        rows.append(s)
        print(f"offline {kind:<5} sequential p50 {s['p50_s']:.3f}s p95 {s['p95_s']:.3f}s "
              f"{s['pairs_per_s']:.0f} pairs/s {s['tokens_per_s']:.0f} tok/s", flush=True)
        # Throughput: every pair of every request in one call (upper bound of the engine).
        reqs2, toks2 = make_workload(args.requests, args.docs, args.doc_min, args.doc_max, kind,
                                     args.seed + 9 + (0 if kind == "short" else 1))
        qs = [q for q, docs in reqs2 for _ in docs]
        ds = [d for _, docs in reqs2 for d in docs]
        t0 = time.perf_counter()
        llm.score(qs, ds, chat_template=template, use_tqdm=False)
        wall = time.perf_counter() - t0
        s = {"engine": "offline", "query": kind, "concurrency": "all-at-once", "requests": args.requests,
             "wall_s": wall, "pairs_per_s": len(ds) / wall, "tokens_per_s": sum(toks2) / wall}
        rows.append(s)
        print(f"offline {kind:<5} all-at-once {s['pairs_per_s']:.0f} pairs/s {s['tokens_per_s']:.0f} tok/s", flush=True)
    save(args.label or "offline", {"engine": "offline", "prefix_caching": not args.no_prefix_caching,
                                   "max_num_batched_tokens": args.max_num_batched_tokens, "rows": rows})


def report(_args):
    print(f"| run | engine | query | conc / bs | p50 s | p95 s | pairs/s | tok/s | memory |")
    print("|---|---|---|---|---|---|---|---|---|")
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text())
        for r in d["rows"]:
            cb = f"bs={r['batch_size']}" if "batch_size" in r else f"c={r['concurrency']}"
            mem = (f"{r['gpu_peak_alloc_gib']:.1f} GiB peak" if "gpu_peak_alloc_gib" in r
                   else f"{r['gpu_used_mib'] / 1024:.1f} GiB used" if "gpu_used_mib" in r else "")
            p50 = f"{r['p50_s']:.3f}" if "p50_s" in r else ""
            p95 = f"{r['p95_s']:.3f}" if "p95_s" in r else ""
            print(f"| {f.stem} | {r['engine']} | {r['query']} | {cb} | {p50} | {p95} | "
                  f"{r['pairs_per_s']:.0f} | {r['tokens_per_s']:.0f} | {mem} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=["st", "http", "offline", "report"])
    ap.add_argument("--label", default=None, help="result file name in bench_results/")
    ap.add_argument("--requests", type=int, default=32, help="measured requests per cell")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--docs", type=int, default=100, help="documents per request")
    ap.add_argument("--doc-min", type=int, default=300)
    ap.add_argument("--doc-max", type=int, default=500)
    ap.add_argument("--query-kinds", default="short,long")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-sizes", default="16,32,64,128", help="st only")
    ap.add_argument("--concurrency", default="1,8,32", help="http only")
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--model", default="zerank-2-seq-cls")
    ap.add_argument("--server-log", default=None, help="http only: vLLM log, to record its memory line")
    ap.add_argument("--server-args", default="", help="http only: free text recorded with the results")
    ap.add_argument("--no-prefix-caching", action="store_true", help="offline only")
    ap.add_argument("--max-num-batched-tokens", type=int, default=None, help="offline only")
    args = ap.parse_args()
    args.query_kinds = args.query_kinds.split(",")
    args.batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    args.concurrency = [int(x) for x in args.concurrency.split(",")]
    if args.engine == "http" and not args.label:
        sys.exit("--label is required for http (one file per server configuration)")
    {"st": bench_st, "http": bench_http, "offline": bench_offline, "report": report}[args.engine](args)


if __name__ == "__main__":
    main()
