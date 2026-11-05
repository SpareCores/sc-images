# Usage

## Shared Memory and Versions

The vLLM benchmarks fail with Docker's default 64 MiB `/dev/shm`, so pass
`--shm-size=4g`. On CPUs, when it runs in Docker, the harness logs a warning
if `/dev/shm` is smaller than 1 GiB; on GPUs, it does not warn before it
starts. When a vLLM server fails to start on CPUs, the error in the log
suggests `--privileged` and `--shm-size=4g`. Only `--shm-size=4g` is
required; `--privileged` matches production.

Each image's `Dockerfile` sets `BENCHMARK_VLLM_MODE`, so the
[harness](../benchmark.py) always runs in that image's mode: `gpu` in
[`benchmark-vllm-gpu`](../../images/benchmark-vllm-gpu/README.md), even
without GPUs, and `cpu` in the CPU images, even on a server with GPUs. Each
image checks its hardware at startup; see its page.

The `--version` option prints the pinned vLLM and GuideLLM versions and exits,
such as:

```bash
docker run --rm ghcr.io/sparecores/benchmark-vllm-cpu:main --version
```

It prints `vllm=0.27.1 guidellm=0.7.3`, read from the version files built into
the image. All three images take their versions from the same files in
`vllm-common/`.

## Hugging Face Access and Model Cache

The container needs network access to Hugging Face: the harness reads each
model's metadata from it, and the model weights are downloaded from it. Pass an
[access token](https://huggingface.co/docs/hub/security-tokens) in
`HF_TOKEN`, which the harness uses for both; production passes the host's
access token too. The [default models](./workloads.md#models) `gemma-2b` and
`llama-8b` are gated models, so they need an access token from an account that
has accepted their licenses. Without one, the harness cannot read a gated
model's metadata: it logs `metadata lookup failed for <model_id>`, skips the
model unless `BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` is set, and continues
with the next model.

The images set `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `/models`, a Docker
volume, so a plain run downloads the models again each time. To reuse them
across runs, mount a host directory on it with `-v`, such as:

```bash
docker run --rm --shm-size=4g -e HF_TOKEN -v "$PWD/models:/models" ghcr.io/sparecores/benchmark-vllm-cpu:main > results.jsonl 2> results.log
```

If the `--models-dir` directory (default: `/models`) has less than 1 GB of
free space at startup, the harness exits with code 1. This is its only disk
check: it never deletes the models it downloads, so every model of a run, up
to the largest one that fits, stays in that directory. Make sure it has room
for all of them. It also caches the memory footprint it computes for each
model in `.vllm-footprints/` under that directory.

## Run Duration

A run has a 2-hour budget, which it splits across the runnable models and
workloads; see [Time Budget](./workloads.md#time-budget). Sample production
runs took the following times:

| Server type | Image | Duration |
| --- | --- | --- |
| AWS (Amazon Web Services) [`c8a.large`](https://sparecores.com/server/aws/c8a.large) | `benchmark-vllm-cpu` | 28 minutes ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.large/0/vllm/meta.json)) |
| AWS [`c8a.16xlarge`](https://sparecores.com/server/aws/c8a.16xlarge) | `benchmark-vllm-cpu` | 1 hour 5 minutes ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.16xlarge/0/vllm/meta.json)) |
| AWS `g5.16xlarge`, one NVIDIA A10G GPU | `benchmark-vllm-gpu` | 1 hour 14 minutes ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/meta.json)) |
| UpCloud `GPU-32xCPU-384GB-3xL40S`, three NVIDIA L40S GPUs | `benchmark-vllm-gpu` | 1 hour 40 minutes ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/upcloud/GPU-32xCPU-384GB-3xL40S/0/vllm/meta.json)) |

The `--benchmark-timeout-scale` option takes a whole number of 1 or more, and
mainly extends the time limit of each GuideLLM process. It does not let a run
go past 2 hours; see [Time Budget](./workloads.md#time-budget).

## Plan and Probe Modes

The following options run part of the harness only:

- `--plan-only` prints one JSON object with the host's resources, such as its
  vCPUs, memory, NUMA (Non-Uniform Memory Access) nodes, and GPUs. For each
  model, the object shows whether it fits in memory at the shortest workload
  context and, if not, why; see [CPU Memory Fit](./workloads.md#cpu-memory-fit)
  and [GPU Memory Fit](./workloads.md#gpu-memory-fit). It reads the model
  metadata from Hugging Face but starts no vLLM server. On CPUs, it reads the
  host's memory, not the container's `--memory` limit. For example:

  ```bash
  docker run --rm -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu:main --plan-only
  ```

- `--probe-only` starts the smallest model, set by `VLLM_PROBE_MODEL`
  (default: `smol-135m`), and waits for the server's health check, for up to
  15 minutes on CPUs and 8 minutes on GPUs. It exits with code 0 if the server
  started, and 1 otherwise. Production uses it to choose the vLLM image; see
  [Production Setup](#production-setup).

## Output Fields

Each JSON line has the main fields listed in
[Results](../README.md#results). The following fields are specific to CPU or
GPU runs, or read differently on each:

| Field | On CPUs | On GPUs |
| --- | --- | --- |
| `mode` | `cpu` | `gpu` |
| `avx512` | Whether an AMD64 CPU has AVX-512 (Advanced Vector Extensions 512); `null` on Arm64 | Describes CPU runs; ignore it |
| `avx2_only_image` | `true` only when `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` is set, as in [`benchmark-vllm-cpu-avx2`](../../images/benchmark-vllm-cpu-avx2/README.md) | Describes CPU runs; ignore it |
| `tensor_parallel`, `data_parallel` | The model's split across the NUMA nodes; see [CPU Parallelism](./workloads.md#cpu-parallelism) | The model's split across the GPUs; see [GPU Parallelism](./workloads.md#gpu-parallelism) |
| `pipeline_parallel` | Always 1 | The GPU count for `llama-70b` on more than one GPU, while `tensor_parallel` still holds the size that the head-count rule gives |
| `omp_threads_bind` | The CPUs of each vLLM worker, separated by vertical bars, such as `0-15\|16-31\|32-46` for three workers | `null` |
| `gpu_count` | 0 | The number of GPUs |
| `gpu_model`, `total_vram_gb` | Describe GPU runs; ignore them | The name of the first GPU, and the memory of all GPUs together in GiB |
| `tuning` | The GuideLLM load and the vLLM server settings, passed to `vllm serve`; see [Load Settings](./workloads.md#load-settings) | The GuideLLM load; the vLLM server settings are computed but not passed to `vllm serve` |

The following fields read the same in both modes:

- `tuning` holds the GuideLLM load, such as `max_concurrency` and
  `sweep_size`, and the vLLM server settings `max_num_seqs`,
  `max_num_batched_tokens`, `dtype`, and `kv_memory_util`. Its
  `gpu_memory_util` value applies to GPU runs only.
- `tuning_version` is the version of the rules that compute these settings, 9
  in the current harness, and 0 with `BENCHMARK_VLLM_AUTOCONFIG=0`.
- `model_weight_gib` is the size of the model weights in GiB,
  `model_torch_dtype` the data type in the model's configuration, and
  `model_metadata_source` where the harness read the metadata from. The
  `kv_bytes_per_token_per_tp_rank` field is the key-value cache size of one
  token on each vLLM worker of a tensor-parallel group in bytes.
- `runtime_memory` holds the key-value cache size per vLLM worker in GiB and
  in tokens, and the `--kv-cache-memory-bytes` value that vLLM suggests, as
  read from the server's log. It is empty when the log has none of them.

The scores cover only the requests that succeeded. For a metric that GuideLLM
gives no separate statistics of successful requests for, they cover all
requests, failed ones included. The output does not record how many requests
failed.

## Exit Codes

The harness exits with one of the following codes:

- **0:** it wrote at least one JSON line, even if it skipped models or
  workloads, or stopped early.
- **1:** one of the following cases applies:
  - The `--models-dir` directory has less than 1 GB of free space.
  - The `guidellm` command is missing.
  - No model could run, for example because none fits in memory.
  - No JSON line was written.
  - On GPUs, every visible GPU has a compute capability below 8.0; see
    [`benchmark-vllm-gpu`](../../images/benchmark-vllm-gpu/README.md).
- **2:** on CPUs, the CPU is an AMD64 CPU without AVX-512, unless
  `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` is set; see
  [`benchmark-vllm-cpu`](../../images/benchmark-vllm-cpu/README.md).

Check which lines are present in stdout, and look for the following messages
in stderr:

- `metadata lookup failed for <model_id>`: the model's Hugging Face metadata
  could not be read; see
  [Hugging Face Access and Model Cache](#hugging-face-access-and-model-cache).
- `Skipping <model> workload=<name> — insufficient KV cache headroom`: on
  CPUs, the workload's key-value cache did not fit in memory; see
  [CPU Memory Fit](./workloads.md#cpu-memory-fit).
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
| Image | [`benchmark-vllm-gpu`](../../images/benchmark-vllm-gpu/README.md) first, then [`benchmark-vllm-cpu`](../../images/benchmark-vllm-cpu/README.md), then [`benchmark-vllm-cpu-avx2`](../../images/benchmark-vllm-cpu-avx2/README.md). The GPU image is skipped on servers without GPUs or with every GPU below compute capability 8.0, and `benchmark-vllm-cpu-avx2` on Arm64. Each image first runs `--probe-only`, for at most 18 minutes, and the first that exits with code 0 runs the full benchmark, with no fallback after that. | Only the image passed to `docker run`, with no `--probe-only` step first and no fallback to another image |
| GPUs | All GPUs | As passed with `--gpus` |
| Shared memory | 4 GiB | 64 MiB, unless `--shm-size` is passed |
| Privileges | Privileged | Not privileged, unless `--privileged` is passed |
| Networking | Host networking | Docker's default bridge network, unless `--network host` is passed |
| Memory limit | 85% of the host's available memory, with swap disabled and a swappiness of 0 | None, unless `--memory` is passed, and swap as the host allows |
| Environment | The host's `HF_TOKEN` and the `resource-tracker` settings | As passed with `-e`, such as `-e HF_TOKEN`, without the `resource-tracker` settings |
| Overall timeout | 3 hours | None, only the harness's own 2-hour budget |
| Other tasks | Not run in parallel with other benchmarks | Whatever else runs on the host at the same time |

To reproduce the memory limit, pass `--memory=<limit> --memory-swap=<limit>
--memory-swappiness=0`, with `<limit>` set to 85% of the host's available
memory; the same value for `--memory-swap` disables swap. Production reads
the memory that Linux reports as available, the `available` column of
`free -g` in GiB. Write the limit with a unit: for example, with 40 GiB
available, pass `--memory=34g --memory-swap=34g`.

Production stores the results of all three images under the task name `vllm`,
in the `data/<vendor>/<server>/<variant>/vllm/` folder of
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data), where
`<variant>` is a numbered folder, `0` for most server types. The task tries the
fastest image first, so the result of a server type comes from the fastest
image that runs on it.

We investigate failed production runs, manually or semi-automatically, and
then run them again. Until then, a server type's latest result in
`sc-inspector-data` can be a failed or partial run.

## Options

Pass the following options after the image name. See
[Environment Variables](#environment-variables) for the settings passed with
`-e`.

| Option | Meaning | Default |
| --- | --- | --- |
| `--models` | Hugging Face model IDs, or short names of the [default models](./workloads.md#models), to run instead of the default models, separated by spaces, such as `qwen-0.5b <owner>/<model>` | The six default models on GPUs, and the five that run on CPUs |
| `--models-dir` | Directory of the 1 GB free-space check and of the model metadata that the harness downloads. The model weights still go to `HF_HOME` and `HUGGINGFACE_HUB_CACHE` (`/models`), so mount a disk at `/models` to move them | `/models` |
| `--benchmark-timeout-scale` | Whole number of 1 or more that multiplies the time limits of GuideLLM, but not the 2-hour stop; see [Time Budget](./workloads.md#time-budget) | `1` |
| `--probe-only` | Start the probe model, wait for its health check, and exit | Off |
| `--plan-only` | Print the host's resources and whether each model fits in memory, as JSON, without starting vLLM | Off |
| `--version` | Print the pinned vLLM and GuideLLM versions and exit | Off |

## Environment Variables

A default marked "legacy" applies with `BENCHMARK_VLLM_AUTOCONFIG=0`; see
[Legacy Settings](./workloads.md#legacy-settings). If a row lists two
variables and both are set, the first wins, except in the `HF_HOME` and
`HUGGINGFACE_HUB_CACHE` row: the Hugging Face libraries read those, not the
harness. The harness does not check the GPU tensor-parallel override against
the GPU count or the head count of the model.

| Variable | Meaning | Default |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face access token for the model metadata and downloads | Unset |
| `BENCHMARK_VLLM_MODE` | `gpu` or `cpu` mode | Set by each image |
| `BENCHMARK_VLLM_WORKLOADS` | Comma-separated workload names; a value that matches no workload of the mode, such as `long` on CPUs, runs all of them | `chat,rag,long` on GPUs, `chat,rag` on CPUs |
| `BENCHMARK_VLLM_AUTOCONFIG` | `0`, `false`, or `no` uses the legacy static settings | `1` |
| `BENCHMARK_VLLM_PER_WORKLOAD_SERVER` | Restart vLLM for each workload with its context length; `0`, `false`, or `no` turns this off, so one server with the context length of the longest workload runs all workloads | Same as `BENCHMARK_VLLM_AUTOCONFIG` |
| `BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` | Any non-empty value, even `0`, runs a model with the size estimated from the parameter count in the default model table when its Hugging Face metadata cannot be read | Unset, so the model is skipped |
| `BENCHMARK_VLLM_VCPUS` | Host vCPU count used for the load settings | The CPUs the process may run on |
| `BENCHMARK_VLLM_MAX_NUM_SEQS` | Passed to vLLM as `--max-num-seqs` | Computed on CPUs; unset on GPUs, so vLLM's default |
| `BENCHMARK_VLLM_MAX_NUM_BATCHED_TOKENS` | Passed to vLLM as `--max-num-batched-tokens` | Computed on CPUs; unset on GPUs, so vLLM's default |
| `BENCHMARK_VLLM_RAMPUP` | Ramp-up of the throughput stage in seconds | Computed |
| `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` | On CPUs: `1`, `true`, or `yes` skips the AVX-512 check on AMD64 | Unset; `true` in `benchmark-vllm-cpu-avx2` |
| `BENCHMARK_VLLM_CPU_TP`, `VLLM_CPU_TENSOR_PARALLEL_SIZE` | On CPUs: tensor-parallel size | The NUMA node count if it divides the model's attention heads, otherwise 1 |
| `BENCHMARK_VLLM_CPU_DP`, `VLLM_CPU_DATA_PARALLEL_SIZE` | On CPUs: data-parallel size | Computed; see [CPU Parallelism](./workloads.md#cpu-parallelism) |
| `VLLM_CPU_OMP_THREADS_BIND` | On CPUs: CPUs of each vLLM worker; any value other than `auto` is passed to vLLM as is | `auto`, set by the CPU images, so the harness computes it |
| `BENCHMARK_VLLM_NUMA_NODES` | On CPUs: NUMA node count | Read from the host |
| `BENCHMARK_VLLM_CPU_TOPOLOGY` | On CPUs: logical CPUs of each NUMA node, as CPU lists separated by vertical bars | Read from the host |
| `BENCHMARK_VLLM_CPU_PHYSICAL_TOPOLOGY` | On CPUs: one logical CPU per physical core of each NUMA node, in the same format | Read from the host |
| `VLLM_CPU_GPU_MEMORY_UTILIZATION` | On CPUs: value of `--gpu-memory-utilization` for each vLLM worker | Computed |
| `VLLM_CPU_KVCACHE_SPACE` | On CPUs: passed to vLLM, the key-value cache size of its CPU backend in GiB | Unset |
| `VLLM_CPU_DTYPE` | On CPUs: value of `--dtype` | `bfloat16` or `float32`, depending on the CPU |
| `BENCHMARK_VLLM_GPU_TP`, `VLLM_GPU_TENSOR_PARALLEL_SIZE` | On GPUs: tensor-parallel size | From the GPU count and the model's attention heads |
| `BENCHMARK_VLLM_GPU_DP`, `VLLM_GPU_DATA_PARALLEL_SIZE` | On GPUs: data-parallel size | `1` |
| `VLLM_GPU_MEMORY_UTILIZATION` | On GPUs: value of `--gpu-memory-utilization`, also used by the memory fit check | `0.92` |
| `VLLM_PROBE_MODEL` | Model of `--probe-only` | `smol-135m` |
| `VLLM_PROBE_HEALTH_TIMEOUT_CPU_SEC` | On CPUs: health check wait of `--probe-only` in seconds | `900` |
| `VLLM_PROBE_HEALTH_TIMEOUT_GPU_SEC` | On GPUs: health check wait of `--probe-only` in seconds | `480` |
| `GUIDELLM_CPU_SWEEP_SIZE`, `GUIDELLM_GPU_SWEEP_SIZE`, `GUIDELLM_SWEEP_SIZE` | Number of sweep stages, at least 2; the CPU or GPU variable applies in its mode | Computed; legacy: 3 |
| `GUIDELLM__MAX_CONCURRENCY` | Maximum concurrent requests | Computed |
| `GUIDELLM__MAX_WORKER_PROCESSES` | GuideLLM worker processes | Computed |
| `GUIDELLM_MAX_REQUESTS`, `GUIDELLM_MAX_REQUESTS_CPU` | Adds a request cap (`max_requests` constraint) to each GuideLLM stage; with `BENCHMARK_VLLM_AUTOCONFIG=0`, only `GUIDELLM_MAX_REQUESTS_CPU` applies on CPUs | Unset; legacy: 25 on CPUs, 120 on GPUs |
| `GUIDELLM_PROFILES`, `GUIDELLM_CPU_PROFILES` | Any of `legacy`, `sync-throughput`, or `sync` (all three do the same) runs one synchronous and one throughput stage instead of a sweep, with autoconfig on or off; the CPU variable applies on CPUs only | Unset |
| `GUIDELLM_THROUGHPUT_RATE` | Concurrency of the throughput stage; applies only with `GUIDELLM_PROFILES` or `GUIDELLM_CPU_PROFILES` | `8` on CPUs, `16` on GPUs |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | Hugging Face cache directory | `/models`, set by the images |
| `TRACKER_QUIET` | Quiet output of `resource-tracker` | `true`, set by the images |
