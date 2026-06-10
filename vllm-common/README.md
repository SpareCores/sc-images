# vLLM shared assets

Not a published image — shared sources for the `benchmark-vllm-*` images under `images/`.

| File | Purpose |
|------|---------|
| `VLLM_VERSION` | Pinned upstream vLLM tag (e.g. `0.22.1` → `v0.22.1`) |
| `GUIDELLM_VERSION` | Pinned [GuideLLM](https://github.com/vllm-project/guidellm) for serving load tests |
| `benchmark.py` | Start `vllm serve`, run `guidellm benchmark run`, emit JSONL |

## Harness

1. **Probe** (`--probe-only`): load smallest model (SmolLM2-135M), wait for `/health`.
2. **Benchmark**: model ladder × workloads (chat / rag / long) × GuideLLM profile.

   Default ladder (small → large): SmolLM2-135M, Qwen2.5-0.5B, Gemma-2-2B, Llama-3.1-8B, Phi-4
   (GPU only), Llama-3.3-70B bnb-4bit (GPU only, ~48 GiB VRAM). `google/gemma-2-2b-it` and
   `meta-llama/Llama-3.1-8B-Instruct` require Hugging Face license acceptance plus `HF_TOKEN`.
   - **CPU + GPU**: `sweep` (default **3** steps: sync → saturated throughput → one constant rate). Override size with `GUIDELLM_SWEEP_SIZE`, `GUIDELLM_CPU_SWEEP_SIZE`, or `GUIDELLM_GPU_SWEEP_SIZE` (`2` = sync+throughput only; `6` = fuller curve).
   - **Legacy fast path**: `GUIDELLM_PROFILES=legacy` (or `GUIDELLM_CPU_PROFILES=legacy`) runs `synchronous` + capped `throughput` (`GUIDELLM_THROUGHPUT_RATE`, default 8 on CPU).
3. **Multi-GPU**: `--tensor-parallel-size` = visible GPU count; bnb-4bit 70B uses pipeline parallel when needed.

## JSONL fields

`benchmark=vllm_serving`, `measurement` (ttft, tpot, itl, e2el, output_throughput, …), `percentile` (p50/p95/p99/mean), plus `workload`, `profile`, `strategy`, `mode`, `arch`, `tensor_parallel`, etc.

## Images

| Image | Arch | Mode |
|-------|------|------|
| `benchmark-vllm-gpu` | amd64 + arm64 | GPU (`vllm/vllm-openai` from Docker Hub) |
| `benchmark-vllm-cpu` | amd64 (AVX-512) + arm64 | CPU |
| `benchmark-vllm-cpu-avx2` | amd64 AVX2 only | CPU (`vllm-cpu-base-avx2`) |

Inspector tries GPU → Hub CPU → AVX2 CPU; first successful probe runs the full benchmark.

Bump versions: edit `VLLM_VERSION` / `GUIDELLM_VERSION`, push to `main`; CI rebuilds `vllm-cpu-base-avx2` (when pinned) and `benchmark-vllm-*`. For GPU, confirm `vllm/vllm-openai:v{VLLM_VERSION}` is multi-arch on [Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) before merging.
