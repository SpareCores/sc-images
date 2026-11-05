# Changelog

## 2026-11-05: Family manual

`README.md` became the manual of the vLLM benchmarks, with
`docs/workloads.md`, `docs/usage.md`, and `docs/contributors.md`, which
replace the image docs' copies of the shared methodology. The notes below are
from the earlier `README.md`, the maintainers' notes on the harness. They are
kept as written, for the history of the design. Some of their numbers predate
the current harness (`tuning_version` 9); the manual describes the current
rules.

### Autoconfig (budget-first)

When `BENCHMARK_VLLM_AUTOCONFIG=1` (default), the harness derives GuideLLM
load and vLLM server knobs from **vCPU count and RAM**, then **fits them into
the 2h overall time budget** (see `OVERALL_TIMEOUT_SEC` in `benchmark.py`).
Load scales **sub-linearly** with vCPU (open-ended — no hard 500/512 caps),
but **wall time per run is capped** so a 896 vCPU box does not run for days.

| Knob | Scales with vCPU | Bounded by |
|------|------------------|------------|
| `max_concurrency` | sub-linear (~vCPU^0.55–0.65), floored by DP replicas on CPU, at 64 on GPU | `GUIDELLM__MAX_CONCURRENCY` env override |
| `max_seconds` per strategy | workload ctx (2048 vs 4096) + budget / sweep | rag gets ≥2× chat floor before sweep shrink; **only** per-stage limit in autoconfig |
| `--rampup` | **not** vCPU — 15% of `max_seconds`, cap 8s on CPU, 2–10s on GPU | CPU: must leave ≥25s of the throughput stage at peak concurrency |
| `sweep_size` | log(vCPU) capped at 4 on CPU, then **shrunk** to fit `per_run_budget` | `per_run_budget_sec` (45–240s); fewer steps for rag/long ctx |
| CPU TP / DP | NUMA + model size (small models get more DP replicas) | `MAX_CPU_DP`, RAM for weight copies, `BENCHMARK_VLLM_CPU_TP` / `_DP` |
| `max_num_seqs`, KV fraction | sub-linear + model RAM | available memory |
| dtype (CPU) | CPU flags, not model or arch | bfloat16 with `bf16` / `avx512_bf16`, else float32 |

Example CPU load at different sizes (8 model×workload runs, 2h budget,
SmolLM2-135M):

| vCPU (NUMA) | DP | max_conc | sweep | rampup | sec/strategy | measure window |
|------------:|---:|---------:|------:|-------:|-------------:|---------------:|
| 1 (1) | 1 | 6 | 2 | 8s | 120 | 112s |
| 96 (2) | 6 | 116 | 4 | 8s | 60 | 52s |
| 192 (4) | 12 | 182 | 4 | 8s | 60 | 52s |
| 896 (8) | 32 | 497 | 4 | 8s | 60 | 52s |

`measure window` is `--max-seconds` minus throughput-stage `--rampup`.
GuideLLM counts rampup **inside** the stage budget; vCPU-scaled rampup (old
`v/4`, 24s at 96 cores / 30s at ≥128 with ~40s stages) left almost no time at
peak concurrency — that was the sharp drop after 96 CPUs. Extra DP keeps small
models at ~16 threads/rank instead of one giant OpenMP team.

Autoconfig runs `guidellm run --profile kind=sweep,...` with a
`--constraint kind=max_duration` (no max-requests by default). Sweep step
count and throughput ramp-up are profile attributes (`sweep_size`,
`rampup_duration`) — GuideLLM 0.7+ CLI shape. A 192-vCPU metal run can
schedule ~182 concurrent streams; a 1-vCPU box caps at 6. Subprocess wall time
adds warmup, rampup, one extra stage for in-flight drain, and margin so heavy
rag sweeps are not killed early.

Per workload (chat / rag / long), autoconfig restarts `vllm serve` with that
workload's `max_model_len` (2048 / 4096 / 8192) so small-RAM hosts do not
reserve KV for unused long-context headroom. Before starting a workload,
`workload_kv_fits()` skips combos that would KV-OOM on CPU (weights-only
`model_fits` is not enough for long ctx). Budget planning includes the extra
startup time. Host vCPU/RAM come from the `server` table when querying the DB.

### Tensor / data parallelism

**CPU (multi-NUMA):** vLLM's `VLLM_CPU_OMP_THREADS_BIND=auto` drops SMT on x86
(one logical CPU per physical core) and, with a single rank, binds only to NUMA
node 0. The harness therefore writes an **explicit** bind string with one
logical CPU per physical core, partitioned NUMA-locally across TP×DP ranks.
Dockerfile `ENV …=auto` means "let the harness decide".

Divides `--gpu-memory-utilization` by the number of ranks sharing a NUMA node.
On CPU that flag is a fraction of the rank's **own memory node**, and every
worker reserves it independently, so an undivided 0.50 with DP=2 asks for 100%
of the node and kills a rank with `Available memory on node 0 … is less than
requested memory for kv`. Their sum stays under `CPU_NODE_MEMORY_BUDGET` (with
an extra multi-rank derate) so sequential KV allocation still fits; if that
leaves too little KV cache the workload is skipped by `workload_kv_fits()`
rather than crashed.

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
KV cache, even if aggregate host RAM is ample.

This preflight is deliberately conservative but is not a replacement for
vLLM initialization: GPU non-KV memory (CUDA graphs, NCCL, peak activations)
is profiled only after the worker starts. The harness therefore also extracts
vLLM's runtime `Available KV cache memory`, KV token capacity, and suggested
`--kv-cache-memory-bytes` values into each JSONL row's `runtime_memory`.
Context requirements are rounded to vLLM cache blocks (CPU 128, GPU 16), and
the 70B bitsandbytes entry is planned with the same pipeline parallelism used
by the server.

On a **2-GPU** host, default ladder TP:

| Model | Attention heads | TP | GPUs used |
|-------|-----------------|----|-----------|
| SmolLM2-135M | 9 | 1 | GPU 0 only (9 % 2 ≠ 0) |
| Qwen2.5-0.5B | 14 | 2 | both |
| Gemma-2-2B | 8 | 2 | both |
| Llama-3.1-8B | 32 | 2 | both |
| Phi-4 | 40 | 2 | both |
| Llama-3.3-70B bnb-4bit | — | pipeline-parallel-size 2 | both |

SmolLM on 2×GPU is still a real GPU run (`mode=gpu` in JSONL); `nvidia-smi
pmon` showing ~95% SM on one GPU and idle on the other is expected for TP=1.

### Images and version bumps

Inspector tries GPU → Hub CPU → AVX2 CPU; first successful probe runs the full
benchmark.

Bump versions: edit `VLLM_VERSION` / `GUIDELLM_VERSION`, push to `main`; CI
rebuilds `vllm-cpu-base-avx2` (when pinned) and `benchmark-vllm-*`. For GPU,
confirm `vllm/vllm-openai:v{VLLM_VERSION}` is multi-arch on
[Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) before merging.
