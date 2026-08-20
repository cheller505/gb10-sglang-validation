# GB10 sglang Model Validation & OCR Pipeline

Scripts for validating and benchmarking sglang-served models on DGX Spark GB10 nodes,
wired into the Shadow LiteLLM API (`keys.shadow.ncsa.illinois.edu`).

## Models tested

| Model | Node | Port | Purpose |
|---|---|---|---|
| DeepSeek-OCR-2 | moe (192.168.200.135) | 30000 | Vision-specialized OCR (images → text) |
| Qwen3-Embedding-0.6B | moe | 30001 | Text → 1024-dim vectors |
| Qwen3-Reranker-0.6B | moe | 30002 | Relevance judgment (yes/no scoring) |
| Inkling-Small-NVFP4 | larry+curly (2-node TP=2) | 30000 | Large reasoning/chat model (170B) |

## Scripts

### Validation tests
- `test1_ocr_final.py` — DeepSeek-OCR-2 functional validation (image OCR + text-only degradation)
- `test2_embedding.py` — Qwen3-Embedding semantic similarity test (cat/feline vs unrelated)
- `test3_definitive.py` — Qwen3-Reranker yes/no relevance judgment + logprob scoring
- `test3_fix_verify.py` — Verifies the chat-template fix for the reranker

### Benchmarks
- `bench_moe.py` — Benchmarks all 3 moe models (OCR latency, embedding throughput, reranker latency)
- `bench_pipeline.py` — End-to-end RAG pipeline benchmark (OCR → embed → retrieve → rerank)
- `bench_inkling_direct.py` — Inkling-Small performance benchmark (TTFT, tok/s, inter-token latency)

### Shakespeare OCR pipeline
- `pipeline.py` — Processes the complete works of Shakespeare (1,374 scanned pages from
  Internet Archive) through the full pipeline: OCR each page → embed → index → query.
  Source PDF: `archive.org/details/completeworksofw00shakrich`

## Key findings

- **OCR**: ~0.5s/page on clean synthetic images, ~6.5s/page on real 1900-era scans
- **Embedding**: 9ms latency, 1,200+ embed/s in batch — essentially instant
- **Reranker**: 17ms per judgment, sharp discrimination (P(yes)=80% relevant vs <0.1% irrelevant)
- **Inkling-Small**: 22-33 tok/s sustained, 50-80 tok/s burst (speculative decoding)
- **Arabic OCR**: works correctly, handles RTL script and ligatures
- **sglang JPEG bug**: DeepSeek-OCR-2 crashes on JPEG input (`image.size` not subscriptable),
  must convert to PNG first
- **Reranker chat-template fix**: sglang needs `--chat-template` with empty think block
  (`think\n\n/think\n\n`) or the Qwen3-Reranker produces degenerate "no" for all inputs
