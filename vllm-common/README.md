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
   - **GPU**: `sweep` (default 6 steps) — sync → throughput → constant rates.
   - **CPU**: `synchronous` then `throughput` (set `GUIDELLM_CPU_PROFILES=sweep` for sweep).
3. **Multi-GPU**: `--tensor-parallel-size` = visible GPU count; bnb-4bit 70B uses pipeline parallel when needed.

## JSONL fields

`benchmark=vllm_serving`, `measurement` (ttft, tpot, itl, e2el, output_throughput, …), `percentile` (p50/p95/p99/mean), plus `workload`, `profile`, `strategy`, `mode`, `arch`, `tensor_parallel`, etc.

## Images

| Image | Arch | Mode |
|-------|------|------|
| `benchmark-vllm-gpu` | amd64 + arm64 | GPU (Hub or `vllm-gpu-base` on arm64) |
| `benchmark-vllm-cpu` | amd64 (AVX-512) + arm64 | CPU |
| `benchmark-vllm-cpu-avx2` | amd64 AVX2 only | CPU (`vllm-cpu-base-avx2`) |

Inspector tries GPU → Hub CPU → AVX2 CPU; first successful probe runs the full benchmark.

Bump versions: edit `VLLM_VERSION` / `GUIDELLM_VERSION`, push to `main`; CI rebuilds bases and `benchmark-vllm-*` (see [`.github/workflows/push.yml`](../.github/workflows/push.yml)).
