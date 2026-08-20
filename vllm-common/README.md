# vLLM shared assets

Not a published image — shared sources for the `benchmark-vllm-*` images under `images/`.

| File | Purpose |
|------|---------|
| `VLLM_VERSION` | Pinned upstream vLLM tag (e.g. `0.27.1` → `v0.27.1`) |
| `GUIDELLM_VERSION` | Pinned [GuideLLM](https://github.com/vllm-project/guidellm) for serving load tests |
| `benchmark.py` | Start `vllm serve`, run `guidellm run`, emit JSONL |

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
| `max_concurrency` | sub-linear (~vCPU^0.55–0.65), floored by DP replicas | `GUIDELLM__MAX_CONCURRENCY` env override |
| `max_seconds` per strategy | workload ctx (2048 vs 4096) + budget / sweep | rag gets ≥2× chat floor before sweep shrink; **only** per-stage limit in autoconfig |
| `--rampup` | **not** vCPU — 15% of `max_seconds`, cap 8s | must leave ≥25s of the throughput stage at peak concurrency |
| `sweep_size` | log(vCPU) capped at 4 on CPU, then **shrunk** to fit `per_run_budget` | `per_run_budget_sec` (45–240 s); fewer steps for rag/long ctx |
| CPU TP / DP | NUMA + model size (small models get more DP replicas) | `MAX_CPU_DP`, RAM for weight copies, `BENCHMARK_VLLM_CPU_TP` / `_DP` |
| `max_num_seqs`, KV fraction | sub-linear + model RAM | available memory |
| dtype (CPU) | model + arch | gemma / arm64 → bfloat16 |

Example CPU load at different sizes (8 model×workload runs, 2h budget, SmolLM2-135M):

| vCPU (NUMA) | DP | max_conc | sweep | rampup | sec/strategy | measure window |
|------------:|---:|---------:|------:|-------:|-------------:|---------------:|
| 1 (1) | 1 | 6 | 2 | 8s | 120 | 112s |
| 96 (2) | 6 | 116 | 4 | 8s | 60 | 52s |
| 192 (4) | 12 | 182 | 4 | 8s | 60 | 52s |
| 896 (8) | 32 | 497 | 4 | 8s | 60 | 52s |

`measure window` is `--max-seconds` minus throughput-stage `--rampup`. GuideLLM counts rampup **inside** the stage budget; vCPU-scaled rampup (old `v/4`, 24s at 96 cores / 30s at ≥128 with ~40s stages) left almost no time at peak concurrency — that was the sharp drop after 96 CPUs. Extra DP keeps small models at ~16 threads/rank instead of one giant OpenMP team.

### GuideLLM sweep limits

Autoconfig runs `guidellm run --profile kind=sweep,...` with a
`--constraint kind=max_duration` (no max-requests by default). Each stage stops
when its time budget is exhausted. Load scales via **`max_concurrency`** on the
profile (and env `GUIDELLM__MAX_CONCURRENCY`, sub-linear in vCPU). Sweep step
count and throughput ramp-up are profile attributes (`sweep_size`,
`rampup_duration`) — GuideLLM 0.7+ CLI shape.

**Sweep stages** (see [GuideLLM sweep profile](https://github.com/vllm-project/guidellm/blob/main/docs/getting-started/benchmark.md)):

1. **Synchronous** — one request at a time (baseline latency / RPS)
2. **Throughput** — as many concurrent requests as allowed (peak capacity)
3. **Constant-rate** — several stages at rates interpolated between (1) and (2); count = `sweep_size − 2`

**`max_concurrency`** caps parallelism in the throughput and constant-rate stages.
A 192-vCPU metal run can schedule ~182 concurrent streams; a 1-vCPU box caps at 6.

**`max_seconds` per strategy** comes from `per_run_budget / sweep_size`, with a higher
floor for rag @ 4096 ctx. Subprocess wall time adds warmup, rampup, one extra stage
for in-flight drain, and margin so heavy rag sweeps are not killed early.

Optional override: set `GUIDELLM_MAX_REQUESTS` or `GUIDELLM_MAX_REQUESTS_CPU` to pass
a `--constraint kind=max_requests` (legacy path and manual experiments). Legacy
autoconfig-off mode still uses fixed request caps (CPU 25 / GPU 120).

Per workload (chat / rag / long), autoconfig restarts `vllm serve` with that workload's `max_model_len` (2048 / 4096 / 8192) so small-RAM hosts do not reserve KV for unused long-context headroom. Before starting a workload, `workload_kv_fits()` skips combos that would KV-OOM on CPU (weights-only `model_fits` is not enough for long ctx). Budget planning includes the extra startup time.

JSONL rows include `max_model_len`, `tuning_version`, and a `tuning`
object (`tuning_version=7`: v6 CPU/GuideLLM behavior plus authoritative
model-footprint planning and per-node/per-GPU feasibility). `tuning.max_requests`
is `null` unless env override. Host vCPU/RAM come from the `server` table when
querying the DB. Disable autoconfig for A/B against older data:
`BENCHMARK_VLLM_AUTOCONFIG=0`. Disable per-workload server restarts:
`BENCHMARK_VLLM_PER_WORKLOAD_SERVER=0`.

## Tensor / data parallelism

vLLM does **not** always use every visible GPU or CPU socket. The harness picks
parallelism as follows:

**GPU:** largest `--tensor-parallel-size` (TP) ≤ GPU count that divides the
model's attention head count (`tensor_parallel_size()` in `benchmark.py`).
vLLM rejects invalid TP with e.g. `attention heads (9) must be divisible by tensor parallel size (2)`.

**CPU (multi-NUMA):** vLLM's `VLLM_CPU_OMP_THREADS_BIND=auto` drops SMT on x86
(one logical CPU per physical core) and, with a single rank, binds only to NUMA
node 0. The harness therefore writes an **explicit** bind string covering every
allowed logical CPU, partitioned NUMA-locally across TP×DP ranks (one CPU is
reserved for the frontend when vCPU ≥ 2). Dockerfile `ENV …=auto` means "let
the harness decide"; a non-`auto` value is passed through.

Autoconfig then:

1. Sets `--tensor-parallel-size = NUMA node count` when heads are divisible by
   that count (official CPU guidance).
2. Otherwise sets `--data-parallel-size ≥ NUMA node count` (TP=1) so every
   socket serves as an independent replica — needed for models like SmolLM2
   (9 heads) on 4-node boxes.
3. Adds extra DP replicas for small models so each rank stays around 16–64
   OpenMP threads instead of one 896-wide OMP team. Cap: `MAX_CPU_DP` and RAM
   (each replica loads a full copy of the weights).
4. Divides `--gpu-memory-utilization` by the number of ranks sharing a NUMA
   node. On CPU that flag is a fraction of the rank's **own memory node**, and
   every worker reserves it independently, so an undivided 0.50 with DP=2 asks
   for 100% of the node and kills a rank with `Available memory on node 0 … is
   less than requested memory for kv`. Their sum stays under
   `CPU_NODE_MEMORY_BUDGET` (with an extra multi-rank derate) so sequential KV
   allocation still fits; if that leaves too little KV cache the workload is
   skipped by `workload_kv_fits()` rather than crashed.

Resource sizing no longer relies on parameter-count rules of thumb when the
Hub is reachable. `--plan-only` resolves:

- exact selected weight-file bytes from Hugging Face file metadata (using the
  safetensors/bin index to avoid counting duplicate formats);
- layers, attention/KV heads, hidden size, and dtype from `config.json`;
- KV bytes/token as `2 × layers × KV-heads-per-TP-rank × head_dim × dtype_bytes`;
- actual per-NUMA-node memory from sysfs and per-device VRAM from CUDA.

Every CPU layout is checked on the node where each worker's first bound CPU
causes vLLM to allocate. Thus `nobind` correctly places every worker on one
node and is rejected when that node cannot hold all weight copies plus minimum
KV cache, even if aggregate host RAM is ample. GPU checks are per device, with
TP weight sharding and DP replication. If authoritative Hub metadata is
unavailable, preflight skips the model unless
`BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA=1` is explicitly set.

This preflight is deliberately conservative but is not a replacement for
vLLM initialization: GPU non-KV memory (CUDA graphs, NCCL, peak activations)
is profiled only after the worker starts. The harness therefore also extracts
vLLM's runtime `Available KV cache memory`, KV token capacity, and suggested
`--kv-cache-memory-bytes` values into each JSONL row's `runtime_memory`.
Context requirements are rounded to vLLM cache blocks (CPU 128, GPU 16), and
the 70B bitsandbytes entry is planned with the same pipeline parallelism used
by the server.

Overrides: `BENCHMARK_VLLM_CPU_TP` / `BENCHMARK_VLLM_CPU_DP`. Docker runs need
`--shm-size` ≥ 4g (see `VLLM_DOCKER_OPTS` in inspector) or TP/DP workers fail
during gloo/shm init.

**GPU overrides** (for A/B; default remains TP-fill, DP=1): `BENCHMARK_VLLM_GPU_TP`,
`BENCHMARK_VLLM_GPU_DP` (leftover GPUs as replicas when heads are not divisible
by GPU count), `VLLM_GPU_MEMORY_UTILIZATION`, `BENCHMARK_VLLM_MAX_NUM_SEQS`.

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

Emitted JSONL includes `tensor_parallel`, `data_parallel` (CPU), `omp_threads_bind`
(CPU), and `gpu_count` so results are comparable across single- and multi-device
instances.

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
