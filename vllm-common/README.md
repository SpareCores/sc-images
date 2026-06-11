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

   Default ladder (small → large): SmolLM2-135M, Qwen2.5-0.5B, Gemma-2-2B, Llama-3.1-8B, Phi-4,
   Llama-3.3-70B bnb-4bit (~48 GiB). On CPU, each model runs only when `model_fits` (RAM); 70B is
   skipped on CPU because bitsandbytes quant is GPU-only. `google/gemma-2-2b-it` and
   `meta-llama/Llama-3.1-8B-Instruct` require Hugging Face license acceptance plus `HF_TOKEN`.
   - **CPU + GPU**: `sweep` (default autoconfig: budget-limited steps — sync → throughput → constant interpolations). Override with `GUIDELLM_SWEEP_SIZE`, `GUIDELLM_CPU_SWEEP_SIZE`, or `GUIDELLM_GPU_SWEEP_SIZE`. Set `BENCHMARK_VLLM_AUTOCONFIG=0` for legacy static settings (CPU `max_requests=25`, `sweep_size=3`).
   - **Legacy fast path**: `GUIDELLM_PROFILES=legacy` (or `GUIDELLM_CPU_PROFILES=legacy`) runs `synchronous` + capped `throughput` (`GUIDELLM_THROUGHPUT_RATE`, default 8 on CPU).
3. **Multi-GPU**: see [Tensor parallelism](#tensor-parallelism) below.

## Autoconfig (budget-first)

When `BENCHMARK_VLLM_AUTOCONFIG=1` (default), the harness derives GuideLLM load and vLLM server knobs from **vCPU count and RAM**, then **fits them into the 2h overall time budget** (see `OVERALL_TIMEOUT_SEC` in `benchmark.py`). Load scales **sub-linearly** with vCPU (open-ended — no hard 500/512 caps), but **wall time per run is capped** so a 896 vCPU box does not run for days.

| Knob | Scales with vCPU | Bounded by |
|------|------------------|------------|
| `max_concurrency`, `max_requests` | sub-linear (~vCPU^0.55–0.65) | budget + GuideLLM env overrides |
| `sweep_size` | log(vCPU), then **shrunk** to fit `per_run_budget` | `per_run_budget_sec` (45–240 s) |
| `max_seconds` per strategy | model size | `per_run_budget / sweep_size` |
| `max_num_seqs`, KV fraction | sub-linear + model RAM | available memory |
| dtype (CPU) | model + arch | gemma / arm64 → bfloat16 |

Example CPU load at different sizes (8 model×workload runs, 2h budget):

| vCPU | max_conc | max_requests | sweep | sec/strategy |
|-----:|---------:|-------------:|------:|-------------:|
| 2 | 32 | 64 | 3 | ~80 |
| 192 | 182 | 364 | 6 | ~40 |
| 896 | 497 | 994 | 7 | ~34 |

Per workload (chat / rag / long), autoconfig restarts `vllm serve` with that workload's `max_model_len` (2048 / 4096 / 8192) so small-RAM hosts do not reserve KV for unused long-context headroom. Budget planning includes the extra startup time.

JSONL rows include `max_model_len`, `tuning_version`, and a `tuning` object (`tuning_version=2` adds per-workload server restarts and explicit KV cache sizing). Host vCPU/RAM come from the `server` table when querying the DB. Disable autoconfig for A/B against older data: `BENCHMARK_VLLM_AUTOCONFIG=0`. Disable per-workload server restarts: `BENCHMARK_VLLM_PER_WORKLOAD_SERVER=0`.

## Tensor parallelism

vLLM does **not** always use every visible GPU. The harness sets the largest
`--tensor-parallel-size` (TP) that is ≤ GPU count and divides the model's
attention head count (`tensor_parallel_size()` in `benchmark.py`). vLLM rejects
invalid TP with e.g. `attention heads (9) must be divisible by tensor parallel size (2)`.

On a **2-GPU** host, default ladder TP:

| Model | Attention heads | TP | GPUs used |
|-------|-----------------|----|-----------|
| SmolLM2-135M | 9 | 1 | GPU 0 only (9 % 2 ≠ 0) |
| Qwen2.5-0.5B | 14 | 2 | both |
| Gemma-2-2B | 8 | 2 | both |
| Llama-3.1-8B | 32 | 2 | both |
| Phi-4 | 40 | 2 | both |
| Llama-3.3-70B bnb-4bit | — | pipeline-parallel-size 2 | both |

SmolLM on 2×GPU is still a real GPU run (`mode=gpu` in JSONL); `nvidia-smi pmon`
showing ~95% SM on one GPU and idle on the other is expected for TP=1.

Emitted JSONL includes `tensor_parallel` (TP used) and `gpu_count` (visible GPUs)
so results are comparable across single- and multi-GPU instances.

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
