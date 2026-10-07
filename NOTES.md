# zerank-2 → Qwen3ForSequenceClassification for vLLM

Converted checkpoint: `./zerank-2-seq-cls/` (from `zeroentropy/zerank-2`, Apache 2.0).
`/score` and `/rerank` return ZeroEntropy's calibrated 0-1 score `sigmoid(logit / 5)` directly.
Raw logit (what `CrossEncoder.predict()` returns) = `5 * log(s / (1 - s))`.

## Serve

```bash
vllm serve ./zerank-2-seq-cls \
  --runner pooling \
  --dtype bfloat16 \
  --chat-template ./zerank-2-seq-cls/score_template.jinja \
  --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.3
```

No `--hf-overrides`. `--chat-template` is **required** (see below); the last two flags are the
best H100 settings found (see "Benchmark"), optional for correctness. Tested with vLLM 0.29.0 and
0.31.0, with the default `--max-model-len` (40960, the base model's limit) and with 16384.
Prefix caching is on by default and should stay on.

**Clients must truncate with `max_tokens_per_doc`, never with `truncate_prompt_tokens`**, and keep
`query tokens + document tokens + 13 < max-model-len` (the template adds 13 tokens; strictly less:
see "Truncation"). Long queries can be capped with `max_tokens_per_query`.

## Conversion (`convert.py`)

- `score.weight` `[1, 2560]` = `model.embed_tokens.weight[9454] / 5`, stored in **fp32**.
  zerank-2 has tied embeddings (no `lm_head.weight`) and scores a single token (`Yes`, id 9454),
  with no negative token.
- vLLM's default activation for `num_labels=1` is a sigmoid, and pooling-model heads run in fp32
  (`head dtype: torch.float32` in the DEBUG log), so the stored weight is used unrounded.
- `config.json`: `architectures: ["Qwen3ForSequenceClassification"]`, `num_labels: 1`,
  `tie_word_embeddings: false`, `use_sep_token: false`. No `method` / `classifier_from_token` /
  `is_original_qwen3_reranker`: those trigger an online re-conversion from `lm_head`.
- The sentence-transformers files are deliberately **not** copied: `config_sentence_transformers.json`
  carries `activation_fn: Identity`, which vLLM would pick up instead of the sigmoid.
- Converted shard by shard, with an exact check that `score.weight * 5 == embed_tokens[9454]`.

## Parity (`reference.py` + `parity.py`, H100 80GB, bf16)

Ground truth = the official model in fp32 (HF backbone + `Yes` row; matches `CrossEncoder.predict`
in fp32 to 2e-5 logit). Tolerance = 1.5x the deviation of the official bf16 path
(`CrossEncoder.predict`) from fp32, measured on the same pairs.

**Main set**: 40 pairs, 9 queries, multilingual, 19 → 6051 tokens, mixed lengths in each batch.
**Long set**: 7 pairs, 48 → 38203 tokens (Python stdlib docs, relevant passage at start / middle / end).

| deviation from fp32 (max logit / max 0-1 score) | main set        | long set        |
|-------------------------------------------------|-----------------|-----------------|
| official ST bf16 (`predict`), the noise floor   | 0.183 / 0.0053  | 0.121 / 0.0042  |
| vLLM 0.31.0, batched `/rerank`                  | 0.095–0.136 / 0.0040 | 0.066 / 0.0032 |
| vLLM 0.31.0, one pair per request               | 0.117 / 0.0040  | 0.065 / 0.0033  |
| vLLM 0.29.0                                     | same range      | 0.057 / 0.0028  |

(The batched range depends on batch composition: bf16 inference is not batch-invariant.)

- Prompt token ids: identical to the official path on 47/47 pairs.
- By length bucket, vLLM stays below ST bf16 everywhere; no extra drift with length
  (24k–38k tokens: 0.066 vs 0.121).
- Ranking: top-1 identical to fp32 on every query; no inversion between documents more than
  2x the noise floor apart.
- vLLM single vs batched: up to 0.085 logit; the official path batch 1 vs batch 8: up to 0.19.
- Model card example `("What is 2+2?", "4")`: card 5.4062, ST bf16 here 5.4375 (one bf16 step).

Comparing vLLM directly to ST bf16 gives up to 0.25 logit: that is the sum of two different
bf16 backbones' noise (HF sdpa vs vLLM kernels), not a conversion error. HF sdpa vs eager
attention alone moves one logit by 0.28.

### Long documents (model behaviour, same in fp32)

Relevance is diluted in long documents, most when the relevant passage sits in the middle:

| document (query: argparse subcommands)  | tokens | 0-1 score |
|-----------------------------------------|--------|-----------|
| argparse docs alone                      | 6228   | 0.68      |
| argparse docs + unrelated text after     | 15187  | 0.41      |
| unrelated text + argparse docs at the end| 23207  | 0.52      |
| argparse docs in the middle              | 31561  | 0.22      |
| unrelated text only                      | 31810  | 0.11      |

Ordering stays correct, but chunking long documents is advisable.

## Instructions

vLLM 0.29+ accepts an `instruction` field on `/score` and `/rerank` (folded into
`chat_template_kwargs`), and only forwards it if the template references `instruction`.
`score_template.jinja` renders it as `<query>{query}</query>\n<instruction>{instruction}</instruction>`
in the system turn (ZeroEntropy's recommended format). Without an instruction (absent, null, empty
or blank), the rendered prompt is byte-identical to the template used for the parity runs (checked
on all reference pairs). The instruction path itself was not run against a live server.

## Truncation (`truncation.py`)

Over-long document (48k tokens, max-model-len 40960). Identical on 0.29.0 and 0.31.0.

| request | outcome |
|---|---|
| no truncation parameter | HTTP 400 with a clear message ✅ |
| `max_tokens_per_doc=8000` (`/score` and `/rerank`) | document truncated, template intact: score equals scoring the cut document ✅ |
| `truncate_prompt_tokens=N` | **silently wrong score** ❌: truncates the whole prompt from the end, cutting `<\|im_start\|>assistant\n` (logit 7.56 instead of 2.25 at N=8192) |
| `truncate_prompt_tokens=N`, `truncation_side=left` | **silently wrong score** ❌: cuts the query |
| `truncate_prompt_tokens=-1` | **hangs forever** ❌ (see below) |

**vLLM scheduler bug (0.29.0 and 0.31.0)**: a prompt of *exactly* `max-model-len` tokens is never
completed. `v1/core/sched/scheduler.py` caps each prefill step at
`max_model_len - num_computed_tokens - num_sampled_tokens_per_step`, and that reservation is 1 even
for pooling models, which never sample, so the last prompt token can never be scheduled. The
request stays "running" with no error and no GPU work until the client disconnects (other requests
are still served). `max-model-len - 1` tokens works (1.4 s cold for 40959 tokens). It does not
reproduce when the prompt's prefix is already in the prefix cache. `truncate_prompt_tokens=-1`
truncates to exactly `max-model-len`, so it always hits this. Hence the strict `<` above.

## What breaks the score

| mistake | effect (measured) |
|---|---|
| no `--chat-template` | vLLM concatenates `query + document` raw: mean error 11.4 logit, 29 ranking inversions |
| original zerank-2 `chat_template.jinja` as `--chat-template` | vLLM renders score templates without `add_generation_prompt`, so `<\|im_start\|>assistant\n` is missing: mean error 9.9 logit, 9 inversions |
| `truncate_prompt_tokens` | template suffix (or query) cut: silently wrong scores; `-1` hangs |
| prompt of exactly `max-model-len` tokens | request hangs forever (vLLM scheduler bug) |
| keeping `config_sentence_transformers.json` | `activation_fn: Identity` → raw logits instead of 0-1 scores (from source, not run) |
| loading the original checkpoint as `Qwen3ForSequenceClassification` | randomly initialized `score.weight`, no error |
| `no`/`yes` pair or `from_2_way_softmax` (Qwen3-Reranker recipe) | wrong formula for zerank-2, compressed scores |
| fp16 instead of bf16 | not tested |

vLLM ≤ 0.31 ignores the checkpoint's `chat_template.jinja` for scoring (explicit FIXME in
`entrypoints/pooling/scoring/io_processor.py`): the template only applies through `--chat-template`.
`chat_template.jinja` in the checkpoint is the same score template, for `apply_chat_template` users.
`/tokenize` does use it, so a passing `/tokenize` check does not prove `/score` is templated;
`parity.py` also checks `usage.prompt_tokens` on every `/score` call for that reason.

## Benchmark (`bench.py`, H100 80GB, vLLM 0.31.0, bf16)

Workload: one rerank call = 1 query + 100 documents of 300–500 tokens (Python stdlib docs).
"short" queries ~15 tokens, "long" queries ~300 tokens. Every request has a unique query, so the
prefix cache only helps within a call (the query is shared by its 100 prompts), never across calls.
tok/s counts all prompt tokens, including those served from the prefix cache. 32+ measured calls
per cell after 3 warm-up calls.

| engine / settings | query | concurrency | p50 latency | p95 latency | pairs/s | tok/s |
|---|---|---|---|---|---|---|
| ST `predict` bf16, batch 32 (best of 16–128) | short | 1 | 1.193 s | 1.230 s | 84 | 35.9k |
| ST `predict` bf16, batch 32 | long | 1 | 1.866 s | 1.949 s | 53 | 38.3k |
| vLLM defaults | short | 1 / 8 / 32 | 0.645 / 4.53 / 18.3 s | 0.659 / 4.64 / 18.4 s | 155 / 175 / 174 | 66.7k / 74.9k / 75.1k |
| vLLM defaults | long | 1 / 8 / 32 | 0.717 / 4.66 / 18.9 s | 0.737 / 4.79 / 19.0 s | 139 / 168 / 169 | 99.9k / 120k / 121k |
| vLLM `--no-enable-prefix-caching` | long | 1 / 8 / 32 | 1.110 / 7.91 / 31.8 s | 1.129 / 7.95 / 32.0 s | 90 / 101 / 101 | 64.4k / 72.1k / 72.2k |
| **vLLM recommended** (`--max-num-batched-tokens 65536 --gpu-memory-utilization 0.3`) | short | 1 / 8 / 32 | **0.622** / 4.30 / 17.3 s | 0.639 / 4.32 / 18.1 s | 161 / **184** / 183 | 69.3k / 78.7k / 78.8k |
| **vLLM recommended** | long | 1 / 8 / 32 | **0.689** / 4.40 / 17.7 s | 0.710 / 5.06 / 18.6 s | 145 / **177** / 177 | 104k / 127k / 127k |
| vLLM offline `LLM.score`, one call at a time | short / long | 1 | 0.608 / 0.675 s | 0.623 / 0.686 s | 164 / 148 | 69.7k / 106k |
| vLLM offline `LLM.score`, all calls at once | short / long | – | – | – | 169 / 155 | 72.1k / 111k |

Findings:
- vLLM is ~1.9x faster per call than `predict` (0.62 s vs 1.19 s) and ~2.2x (short queries) to
  3.3x (long queries) higher throughput. ST's own batch size barely matters past 32.
- The GPU saturates at concurrency ~8 (~180 pairs/s); beyond that, latency grows linearly with the
  queue (c=32: ~18 s per call). Size concurrency on the client side to the latency budget.
- **Prefix caching is supported** for this model in 0.29 and 0.31 (causal attention + LAST pooling;
  `ModelConfig.is_prefix_caching_supported`) and on by default. No effect for short queries
  (under one 16-token block); for ~300-token queries it cuts latency 35% and raises throughput 67%.
  Parity passes with it on.
- `--max-num-batched-tokens` (default 8192): 16k / 32k / 64k give +2% / +4% / +5% throughput and up
  to -4% latency; peak activation memory goes from 0.95 GiB (8k) to 1.5 / 2.7 / 5.1 GiB.
  Small but free.
- `--gpu-memory-utilization 0.3` (~24 GiB reserved, 10.9 GiB KV cache ≈ 77k tokens) performs the same
  as the default 0.92 (73 GiB). Below ~0.25 the KV cache could no longer hold one 40960-token
  prompt (~147 KiB/token in bf16; computed, not tested).
- HTTP + server overhead is ~2% at concurrency 1 (0.622 s vs 0.608 s offline). Offline "all at once"
  is slower than the server at c=8: the server overlaps tokenization of queued calls with GPU work.
- Memory: ST peak allocation 11–25 GiB depending on batch size (it materializes full-vocabulary
  logits at every position); vLLM 7.8 GiB weights + 1–5 GiB activations + the KV cache you allow.

## Reproduce

```bash
uv venv -p 3.12 --managed-python .venv-031   # managed Python: Triton needs Python.h
VIRTUAL_ENV=.venv-031 uv pip install vllm==0.31.0 "sentence-transformers>=5.4" requests
hf download zeroentropy/zerank-2 --local-dir upstream
.venv-031/bin/python reference.py              # writes reference.json (needs a free GPU)
.venv-031/bin/python reference.py --set long   # writes reference_long.json
.venv-031/bin/python convert.py                # writes ./zerank-2-seq-cls
# start the server as above, then (each exits 1 on failure):
.venv-031/bin/python parity.py
.venv-031/bin/python parity.py --ref reference_long.json
.venv-031/bin/python truncation.py             # currently fails: documents the issues above
.venv-031/bin/python bench.py http --label vllm-recommended   # server up; `bench.py st` / `offline` with GPU free
.venv-031/bin/python bench.py report
```

Versions used: vLLM 0.31.0 / 0.29.0, torch 2.13.0+cu130, transformers 5.17 (0.31 venv) / 5.19
(0.29 venv), sentence-transformers 6.1.0.
