# Usage

## CPU Requirements

The image is built on the official vLLM CPU server images from Docker Hub,
`vllm/vllm-openai-cpu:v0.27.1-x86_64` for AMD64 and
`vllm/vllm-openai-cpu:v0.27.1-arm64` for Arm64, and its `:main` tag has
variants for both. The [`Dockerfile`](../Dockerfile) sets
`BENCHMARK_VLLM_MODE=cpu`, so the
[harness](../../../vllm-common/benchmark.py) always runs in CPU mode, even on
a server with GPUs.

At startup on AMD64, this benchmark reads the CPU flags from `/proc/cpuinfo`.
The vLLM CPU image for AMD64 needs AVX-512 (Advanced Vector Extensions 512), so
without the `avx512f` flag, this benchmark logs an error that names
`ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main` and exits with code 2.
Setting `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` to `1`, `true`, or `yes` skips this
check. The
[`benchmark-vllm-cpu-avx2`](../../benchmark-vllm-cpu-avx2/README.md) image
sets it; this image does not.

Arm64 CPUs skip the check, and this benchmark does not check the Arm64 CPU
generation either. On Arm64 CPUs without the `bf16` flag, it runs the models
in float32; see [Data Type](./workloads.md#data-type).

When it runs in Docker, this benchmark logs a warning if `/dev/shm` is smaller
than 1 GiB. The vLLM benchmarks fail with Docker's default 64 MiB `/dev/shm`,
so pass `--shm-size=4g`. When a vLLM server fails to start, the error in the
log suggests `--privileged --shm-size=4g`. Only `--shm-size=4g` is required;
`--privileged` matches production.

The `--version` option prints the pinned vLLM and GuideLLM versions and exits:

```bash
docker run --rm ghcr.io/sparecores/benchmark-vllm-cpu:main --version
```

It prints `vllm=0.27.1 guidellm=0.7.3`, read from the version files built into
the image.

## Hugging Face Access and Model Cache

The container needs network access to Hugging Face: this benchmark reads
each model's metadata from it, and the model weights are downloaded from it.
Pass an [access token](https://huggingface.co/docs/hub/security-tokens) in
`HF_TOKEN`, which this benchmark uses for both; production passes the host's
token too. The default models `gemma-2b` and `llama-8b` are gated, so they need
a token from an account that has accepted their licenses. Without such a
token, this benchmark cannot read a gated model's metadata: it logs
`metadata lookup failed for <model_id>`, skips the model unless
`BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` is set, and continues with the next
model.

The image sets `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `/models`, a Docker
volume, so a plain run downloads the models again each time. To reuse them
across runs, mount a host directory on it with `-v`, and replace `<limit>` as
[Usage](../README.md#usage) describes:

```bash
docker run --rm --privileged --network host --shm-size=4g --memory=<limit> --memory-swap=<limit> --memory-swappiness=0 -e HF_TOKEN -v "$PWD/models:/models" ghcr.io/sparecores/benchmark-vllm-cpu:main > results.jsonl 2> results.log
```

If the `--models-dir` directory (default: `/models`) has less than 1 GB of
free space at startup, this benchmark exits with code 1.

## Run Duration

A run has a 2-hour budget, which it splits across the runnable models and
workloads; see [Time Budget](./workloads.md#time-budget). Sample production
runs on AWS (Amazon Web Services) server types took the following times:

- `c8a.large`: 28 minutes
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.large/0/vllm/meta.json))
- `c8a.16xlarge`: 1 hour 5 minutes
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.16xlarge/0/vllm/meta.json))

The `--benchmark-timeout-scale` option takes a whole number of 1 or more, and
mainly extends the time limit of each GuideLLM process. It does not let a run
go past 2 hours; see [Time Budget](./workloads.md#time-budget).

## Plan and Probe Modes

The following options run part of this benchmark only:

- `--plan-only` prints one JSON object with the host's resources, such as its
  vCPUs, memory, and NUMA (Non-Uniform Memory Access) nodes. For each model,
  the object shows whether it fits in memory and, if not, why; see
  [Memory Fit](./workloads.md#memory-fit). It reads the model metadata from
  Hugging Face but starts no vLLM server. It reads the host's memory, not the
  container's `--memory` limit.
- `--probe-only` starts the smallest model, set by `VLLM_PROBE_MODEL`
  (default: `smol-135m`), and waits for the server's health check for up to 15
  minutes, set by `VLLM_PROBE_HEALTH_TIMEOUT_CPU_SEC`. It exits with code 0 if the
  server started. Production uses it to choose the vLLM image; see
  [Production Setup](#production-setup).

## Output Fields

Each JSON line has the main fields listed in
[Results](../README.md#results). The following fields are specific to CPU runs
or read differently on CPUs:

- `mode` is `cpu`, and `arch` is `amd64` or `arm64`.
- `avx512` tells whether an AMD64 CPU has AVX-512, and is `null` on Arm64.
  The `avx2_only_image` field is `true` only when
  `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` is set, so it is `false` in this image by
  default.
- `tensor_parallel` and `data_parallel` give the model's split across the NUMA
  nodes, and `pipeline_parallel` is always 1 on CPUs. The `omp_threads_bind`
  field lists the CPUs of each vLLM worker, separated by vertical bars, such as
  `0-15|16-31|32-46` for three workers. See
  [CPU Parallelism](./workloads.md#cpu-parallelism).
- `tuning` holds the settings of the run: the GuideLLM load, such as
  `max_concurrency` and `sweep_size`, and the vLLM server settings
  `max_num_seqs`, `max_num_batched_tokens`, `dtype`, and `kv_memory_util`. On
  CPUs, this benchmark passes the server settings to `vllm serve`; see
  [Load Settings](./workloads.md#load-settings). Its `gpu_memory_util` value
  applies to GPU runs only.
- `tuning_version` is the version of the rules that compute these settings, 9
  in the current harness.
- `model_weight_gib` is the size of the model weights in GiB,
  `model_torch_dtype` the data type in the model's configuration, and
  `model_metadata_source` where this benchmark read the metadata from. The
  `kv_bytes_per_token_per_tp_rank` field is the key-value cache size of one
  token on each tensor-parallel rank, in bytes.
- `runtime_memory` holds the key-value cache memory and token capacity that
  vLLM reports in its log. It is empty when the log has none of them.
- `gpu_count` is 0. The `gpu_model` and `total_vram_gb` fields describe GPU
  runs; ignore them in CPU results.

## Exit Codes

This benchmark exits with code 1 in the following cases:

- The `--models-dir` directory has less than 1 GB of free space.
- The `guidellm` command is missing.
- No model could run, for example because none fits in memory.
- No JSON line was written.

It exits with code 2 on an AMD64 CPU without AVX-512, unless
`BENCHMARK_VLLM_ALLOW_AVX2_ONLY` is set, and with code 0 otherwise. A run that
skipped models or workloads, or stopped early, still exits with code 0 if it
wrote at least one JSON line. Check which lines are present, and look for the
following messages in stderr:

- `metadata lookup failed for <model_id>`: the model's Hugging Face metadata
  could not be read; see
  [Hugging Face Access and Model Cache](#hugging-face-access-and-model-cache).
- `Skipping <model> workload=<name> — insufficient KV cache headroom`: the
  workload's key-value cache did not fit in memory; see
  [Memory Fit](./workloads.md#memory-fit).
- `Overall timeout reached`: the run reached its 2-hour stop, and the
  remaining models did not run; see [Time Budget](./workloads.md#time-budget).

## Production Setup

The [Inspector](https://docs.sparecores.com/overview#how-it-works) of
Spare Cores Navigator runs one task, `vllm`, for the three vLLM images, on
every server type except those on the
[`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
blocklist. The blocklist also excludes server types with less than 2 GB of
memory from this task. The task uses the following settings:

| Setting | Production | Plain `docker run` |
| --- | --- | --- |
| Image | [`benchmark-vllm-gpu`](../../benchmark-vllm-gpu/README.md) first, then this image, then [`benchmark-vllm-cpu-avx2`](../../benchmark-vllm-cpu-avx2/README.md). The GPU image is skipped on servers without a capable GPU, and `benchmark-vllm-cpu-avx2` on Arm64. Each image first runs `--probe-only`, and the first that exits with code 0 runs the full benchmark, with no fallback after that. | Only the image you run, with no `--probe-only` step first and no fallback to another image |
| Shared memory | 4 GiB | 64 MiB, unless `--shm-size` is passed |
| Privileges | Privileged | Not privileged, unless `--privileged` is passed |
| Networking | Host networking | Docker's default bridge network, unless `--network host` is passed |
| Memory limit | 85% of the host's available memory, with swap disabled and a swappiness of 0 | None, unless `--memory` is passed, and swap as the host allows |
| Environment | The host's `HF_TOKEN` | As passed with `-e`, such as `-e HF_TOKEN` |
| Overall timeout | 3 hours | None, only this benchmark's own 2-hour budget |

The command in [Usage](../README.md#usage) reproduces these container
settings.

Production stores the results of all three images under the task name `vllm`,
in the `data/<vendor>/<server>/<variant>/vllm/` folder of
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data), where
`<variant>` is a numbered folder, `0` for most server types. The
task tries the fastest image first, so the result of a server type comes from
the fastest image that runs on it.

We investigate failed production runs, manually or semi-automatically, and
then run them again. Until then, a server type's latest result in
`sc-inspector-data` can be a failed or partial run.

## Options

Pass the following options after the image name. See
[Environment Variables](#environment-variables) for the settings passed with
`-e`.

| Option | Meaning | Default |
| --- | --- | --- |
| `--models` | Hugging Face model IDs, or short names of the [default models](./workloads.md#models), to run instead of the default models, separated by spaces, such as `qwen-0.5b <owner>/<model>` | The five default models that run on CPUs |
| `--models-dir` | Directory of the 1 GB free-space check and of the model metadata that this benchmark downloads. The model weights still go to `HF_HOME` and `HUGGINGFACE_HUB_CACHE` (`/models`), so mount a disk at `/models` to move them | `/models` |
| `--benchmark-timeout-scale` | Whole number of 1 or more that multiplies the time limits of GuideLLM, but not the 2-hour stop; see [Time Budget](./workloads.md#time-budget) | `1` |
| `--probe-only` | Start the probe model, wait for its health check, and exit | Off |
| `--plan-only` | Print the host's resources and whether each model fits in memory, as JSON, without starting vLLM | Off |
| `--version` | Print the pinned vLLM and GuideLLM versions and exit | Off |

## Environment Variables

The following variables apply to CPU runs. A default marked "legacy" applies
with `BENCHMARK_VLLM_AUTOCONFIG=0`; see
[Legacy Settings](./workloads.md#legacy-settings). If a row lists two
variables and both are set, the first wins, except in the `HF_HOME` and
`HUGGINGFACE_HUB_CACHE` row: the Hugging Face libraries read those, not this
benchmark.

| Variable | Meaning | Default |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face access token for the model metadata and downloads | Unset |
| `BENCHMARK_VLLM_MODE` | `gpu` or `cpu` mode | `cpu`, set by the image |
| `BENCHMARK_VLLM_WORKLOADS` | Comma-separated workload names; a value that matches no CPU workload, such as `long`, runs all of them | `chat,rag` |
| `BENCHMARK_VLLM_AUTOCONFIG` | `0`, `false`, or `no` uses the legacy static settings | `1` |
| `BENCHMARK_VLLM_PER_WORKLOAD_SERVER` | Restart vLLM for each workload with its context length; `0`, `false`, or `no` turns this off, so one server with the context length of the longest workload runs all workloads | Same as `BENCHMARK_VLLM_AUTOCONFIG` |
| `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` | `1`, `true`, or `yes` skips the AVX-512 check on AMD64 | Unset |
| `BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` | Any non-empty value, even `0`, runs a model with an estimated size when its Hugging Face metadata cannot be read | Unset, so the model is skipped |
| `BENCHMARK_VLLM_CPU_TP`, `VLLM_CPU_TENSOR_PARALLEL_SIZE` | Tensor-parallel size | The NUMA node count if it divides the model's attention heads, otherwise 1 |
| `BENCHMARK_VLLM_CPU_DP`, `VLLM_CPU_DATA_PARALLEL_SIZE` | Data-parallel size | Computed; see [CPU Parallelism](./workloads.md#cpu-parallelism) |
| `VLLM_CPU_OMP_THREADS_BIND` | CPUs of each vLLM worker; any value other than `auto` is passed to vLLM as is | `auto`, set by the image, so this benchmark computes it |
| `BENCHMARK_VLLM_VCPUS` | Host vCPU count used for the load settings | The CPUs the process may run on |
| `BENCHMARK_VLLM_NUMA_NODES` | NUMA node count | Read from the host |
| `BENCHMARK_VLLM_CPU_TOPOLOGY` | Logical CPUs of each NUMA node, as CPU lists separated by vertical bars | Read from the host |
| `BENCHMARK_VLLM_CPU_PHYSICAL_TOPOLOGY` | One logical CPU per physical core of each NUMA node, in the same format | Read from the host |
| `BENCHMARK_VLLM_MAX_NUM_SEQS` | Passed to vLLM as `--max-num-seqs` | Computed |
| `BENCHMARK_VLLM_MAX_NUM_BATCHED_TOKENS` | Passed to vLLM as `--max-num-batched-tokens` | Computed |
| `BENCHMARK_VLLM_RAMPUP` | Ramp-up of the throughput stage in seconds | Computed |
| `VLLM_CPU_GPU_MEMORY_UTILIZATION` | Value of `--gpu-memory-utilization` for each vLLM worker | Computed |
| `VLLM_CPU_KVCACHE_SPACE` | Passed to vLLM: the key-value cache size of its CPU backend in GiB | Unset |
| `VLLM_CPU_DTYPE` | Value of `--dtype` | `bfloat16` or `float32`, depending on the CPU |
| `VLLM_PROBE_MODEL` | Model of `--probe-only` | `smol-135m` |
| `VLLM_PROBE_HEALTH_TIMEOUT_CPU_SEC` | Health check wait of `--probe-only` in seconds | `900` |
| `GUIDELLM_CPU_SWEEP_SIZE`, `GUIDELLM_SWEEP_SIZE` | Number of sweep stages, at least 2 | Computed; legacy: 3 |
| `GUIDELLM__MAX_CONCURRENCY` | Maximum concurrent requests | Computed |
| `GUIDELLM__MAX_WORKER_PROCESSES` | GuideLLM worker processes | Computed |
| `GUIDELLM_MAX_REQUESTS`, `GUIDELLM_MAX_REQUESTS_CPU` | Adds a request cap (`max_requests` constraint) to each GuideLLM stage; with `BENCHMARK_VLLM_AUTOCONFIG=0`, only `GUIDELLM_MAX_REQUESTS_CPU` applies | Unset; legacy: 25 |
| `GUIDELLM_PROFILES`, `GUIDELLM_CPU_PROFILES` | Any of `legacy`, `sync-throughput`, or `sync` (all three do the same) runs one synchronous and one throughput stage instead of a sweep, with autoconfig on or off | Unset |
| `GUIDELLM_THROUGHPUT_RATE` | Concurrency of the throughput stage; applies only with `GUIDELLM_PROFILES` or `GUIDELLM_CPU_PROFILES` | `8` |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | Hugging Face cache directory | `/models`, set by the image |
| `TRACKER_QUIET` | Quiet output of `resource-tracker` | `true`, set by the image |
