# zerank-2 for vLLM

Conversion of [`zeroentropy/zerank-2-reranker`](https://huggingface.co/zeroentropy/zerank-2-reranker)
(formerly `zeroentropy/zerank-2`, revision `5eae30d5`) into a
`Qwen3ForSequenceClassification` checkpoint that vLLM serves on `/score` and `/rerank`
without `--hf-overrides`, returning ZeroEntropy's calibrated 0-1 score `sigmoid(logit / 5)`.

Converted checkpoint: [`polaria-tech/zerank-2-reranker-vllm`](https://huggingface.co/polaria-tech/zerank-2-reranker-vllm).
Below, `./zerank-2-seq-cls` is the local output of `convert.py`.

Scores are checked against the official `CrossEncoder.predict()` path (fp32 ground truth,
bf16 noise floor) on vLLM 0.29.0 and 0.31.0, H100, from 19 to 38k tokens.

```bash
vllm serve ./zerank-2-seq-cls \
  --runner pooling \
  --dtype bfloat16 \
  --chat-template ./zerank-2-seq-cls/score_template.jinja
```

`--chat-template` is required, and clients must truncate with `max_tokens_per_doc`, never
`truncate_prompt_tokens`. Method, measured deviations, benchmarks and the list of what silently
breaks the score: [NOTES.md](NOTES.md).

| file | purpose |
|---|---|
| `convert.py` | streaming conversion of the original checkpoint |
| `score_template.jinja` | score template to pass with `--chat-template` |
| `reference.py` | reference scores from the official path (bf16 and fp32) |
| `parity.py` | compares a running vLLM server against the reference; exits 1 on failure |
| `pairs.py` | test pairs (mixed lengths, multilingual, long documents) |
| `truncation.py` | truncation behaviour of `/score` and `/rerank` |
| `bench.py` | latency/throughput benchmark, sentence-transformers vs vLLM |

zerank-2 is © ZeroEntropy, released under Apache 2.0 (relicensed on 2026-07-24; earlier
revisions had a different license). This repository is not affiliated with
ZeroEntropy. Code licensed under Apache 2.0 ([LICENSE](LICENSE)).
