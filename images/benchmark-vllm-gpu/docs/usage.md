# Usage

## GPU Run

Export your Hugging Face access token as `HF_TOKEN` in the shell first;
`-e HF_TOKEN` passes it into the container. Then run the image via Docker with
the GPUs passed and a 4 GiB shared memory:

```bash
docker run --rm --gpus all --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-gpu:main > results.jsonl
```

The host needs the NVIDIA driver and the NVIDIA Container Toolkit for
`--gpus`. The vLLM benchmarks fail with Docker's default 64 MiB `/dev/shm`, so
pass `--shm-size=4g`. On GPUs, this benchmark does not warn about a small
`/dev/shm` before it starts.

The image is built on the official vLLM server image,
`vllm/vllm-openai:v0.27.1` from Docker Hub, and the
[`Dockerfile`](../Dockerfile) sets `BENCHMARK_VLLM_MODE=gpu`, so the
[harness](../../../vllm-common/benchmark.py) always runs in GPU mode. The
`:main` tag has variants for AMD64 and Arm64.

At startup, this benchmark reads the CUDA (Compute Unified Device Architecture)
compute capability of each visible GPU. It needs 8.0 or newer, the Ampere
generation or later. If every visible GPU is below 8.0, it logs an error and
exits with code 1. A GPU whose compute capability cannot be read is allowed.

If no GPU is visible, for example without `--gpus`, the check passes, but no
model fits the [GPU memory check](./workloads.md#gpu-memory-fit). This
benchmark then logs each model as skipped for insufficient memory and exits
with code 1.

The `--version` option prints the pinned vLLM and GuideLLM versions and exits:

```bash
docker run --rm ghcr.io/sparecores/benchmark-vllm-gpu:main --version
```

It prints `vllm=0.27.1 guidellm=0.7.3`, read from the version files built into
the image.

## Hugging Face Access and Model Cache

This benchmark reads the model metadata from Hugging Face before it starts
vLLM, and vLLM downloads the model weights while the run goes on, so the
container needs network access to Hugging Face. Pass an access token in
`HF_TOKEN`; this benchmark passes it to the Hugging Face Hub, and production
sets it too. A gated model needs a token from an account that has accepted
the model's license.

The image sets `HF_HOME` and `HUGGINGFACE_HUB_CACHE` to `/models`, a Docker
volume, so a plain run downloads the models again each time. To reuse them
across runs, mount a host directory on it:

```bash
docker run --rm --gpus all --shm-size=4g -e HF_TOKEN -v "$PWD/models:/models" ghcr.io/sparecores/benchmark-vllm-gpu:main > results.jsonl
```

If the `--models-dir` directory (default: `/models`) has less than 1 GB of
free space at startup, this benchmark exits with code 1. This is its only disk
check: it never deletes the models it downloads, so every model of a run, up
to the 4-bit 70B model, stays in that directory. Make sure it has room for all
of them. It also caches the memory footprint it computes for each model in `.vllm-footprints/` under that
directory.

## Run Duration

A run has a 2-hour budget, which it splits across the runnable models and
workloads; see [Time Budget](./workloads.md#time-budget). Sample production
runs took the following times:

- AWS (Amazon Web Services) `g5.16xlarge` with 64 vCPUs and one NVIDIA A10G
  GPU: 1 hour 14 minutes
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/meta.json))
- UpCloud `GPU-32xCPU-384GB-3xL40S` with three NVIDIA L40S GPUs: 1 hour
  40 minutes
  ([run](https://github.com/SpareCores/sc-inspector-data/blob/main/data/upcloud/GPU-32xCPU-384GB-3xL40S/0/vllm/meta.json))

The `--benchmark-timeout-scale` option multiplies the overall budget and the
GuideLLM time limits. The checks that stop the run still use the unscaled
2 hours: no new model or workload starts after 2 hours, even when the scaled
budget plans for more.

## Plan and Probe Modes

The following options run part of this benchmark only:

- `--plan-only` prints the host and, for each model, its
  [memory fit](./workloads.md#gpu-memory-fit), as JSON. It reads the model
  metadata from Hugging Face but starts no vLLM server.
- `--probe-only` starts the smallest model, `smol-135m` by default
  (`VLLM_PROBE_MODEL`), and waits up to 8 minutes for the server's health check
  (`VLLM_PROBE_HEALTH_TIMEOUT_GPU_SEC`). It exits with code 0 if the server
  started, and 1 otherwise. Production uses it to choose the vLLM image; see
  [Production Setup](#production-setup).

## Output Fields

Each JSON line has the main fields listed in
[Results](../README.md#results). The following fields are specific to GPU runs
or read differently on GPUs:

- `mode` is `gpu`.
- `gpu_count` is the number of GPUs, `gpu_model` the name of the first GPU, and
  `total_vram_gb` the memory of all GPUs together in GiB.
- `tensor_parallel`, `pipeline_parallel`, and `data_parallel` give the model's
  split across the GPUs. For `llama-70b` on more than one GPU,
  `pipeline_parallel` is the GPU count, while `tensor_parallel` still holds the
  size that the head-count rule gives, although the server runs without
  tensor parallelism. See [GPU Parallelism](./workloads.md#gpu-parallelism).
- `tuning` holds the load settings of the run. On GPUs, its `max_num_seqs`,
  `max_num_batched_tokens`, `dtype`, and `kv_memory_util` values are computed
  but not passed to `vllm serve`, which uses its own defaults for them.
- `runtime_memory` holds the key-value cache size per vLLM worker process in
  GiB and in tokens, and the `--kv-cache-memory-bytes` value that vLLM suggests, as read
  from the server's log. It is empty when the log has none of them.
- `avx512`, `avx2_only_image`, and `omp_threads_bind` describe CPU runs; ignore
  them in GPU results. On GPUs, `omp_threads_bind` is `null`.

Where GuideLLM reports the successful requests separately, the scores use only
those; otherwise they use all requests. The output does not record how many
requests failed.

## Exit Codes

This benchmark exits with code 1 in the following cases, and with code 0
otherwise:

- Every visible GPU has a compute capability below 8.0.
- The `--models-dir` directory has less than 1 GB of free space.
- The `guidellm` command is missing.
- No model could run, for example because none fits in GPU memory.
- No JSON line was written.

A run that skipped models or workloads, or stopped early, still exits with
code 0 if it wrote at least one JSON line. Check which lines are present, and
the reasons logged to stderr.

## Production Setup

The Navigator's Inspector
([`sc-inspector`](https://github.com/SpareCores/sc-inspector)) runs one task,
`vllm`, for this image and the two CPU images, on every server type except
those on the
[`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
blocklist, which also excludes server types with less than 2 GB of memory
from this task. The task uses the following settings:

| Setting | Production | Plain `docker run` |
| --- | --- | --- |
| Image | This image first, then [`benchmark-vllm-cpu`](../../benchmark-vllm-cpu/README.md), then [`benchmark-vllm-cpu-avx2`](../../benchmark-vllm-cpu-avx2/README.md). This image is skipped on servers without GPUs or with every GPU below compute capability 8.0. Each image first runs `--probe-only`, for at most 18 min, and the first that passes runs the full benchmark, with no fallback after that. | This image only |
| GPUs | All GPUs | As passed with `--gpus` |
| Shared memory | 4 GiB | 64 MiB, unless `--shm-size` is passed |
| Privileges | Privileged | Not privileged |
| Networking | Host networking | Docker's default bridge network |
| Memory limit | 85% of the host's available memory, with swap disabled | None |
| Environment | `HF_TOKEN` and the `resource-tracker` settings | As passed with `-e` |
| Overall timeout | 3 hours | None, only this benchmark's own 2-hour budget |
| Other tasks | Not run in parallel with other benchmarks | Not applicable |

To reproduce the container settings above, except the memory limit, run the
following command:

```bash
docker run --rm --privileged --network host --gpus all --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-gpu:main > results.jsonl
```

Production stores the results of all three images under the task name `vllm`,
in `data/<vendor>/<server>/<variant>/vllm/` (the variant is a numbered folder,
`0` for most server types) of
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data). The
results do not mark which image produced them: the task tries the fastest
method first, so the result of a server type is the best one measured.

We investigate failed production runs, manually or semi-automatically, and
then run them again. Until then, a server type's latest result in
`sc-inspector-data` can be a failed or partial run.

## Options

Pass the following options after the image name. See
[Environment Variables](#environment-variables) for the settings passed with
`-e`.

| Option | Meaning | Default |
| --- | --- | --- |
| `--models` | Hugging Face model IDs, or short names of the [default models](./workloads.md#models), to run instead of the default models | The six default models |
| `--models-dir` | Hugging Face cache directory, which needs at least 1 GB of free space | `/models` |
| `--benchmark-timeout-scale` | Multiplier of the overall budget and the GuideLLM time limits | `1` |
| `--probe-only` | Start the probe model, wait for its health check, and exit | Off |
| `--plan-only` | Print the host and the memory plan of each model as JSON, without starting vLLM | Off |
| `--version` | Print the pinned vLLM and GuideLLM versions and exit | Off |

## Environment Variables

The following variables apply to GPU runs. A default marked "legacy" applies
with `BENCHMARK_VLLM_AUTOCONFIG=0`; see
[Legacy Settings](./workloads.md#legacy-settings). If a row lists two
variables and both are set, the first wins. The image does not check the
tensor-parallel override against the GPU count or the head count of the model.

| Variable | Meaning | Default |
| --- | --- | --- |
| `HF_TOKEN` | Hugging Face access token for the model metadata and downloads | Unset |
| `BENCHMARK_VLLM_MODE` | `gpu` or `cpu` mode | `gpu`, set by the image |
| `BENCHMARK_VLLM_WORKLOADS` | Comma-separated workload names; a value that matches none runs all | `chat,rag,long` |
| `BENCHMARK_VLLM_AUTOCONFIG` | `0`, `false`, or `no` uses the legacy static settings | `1` |
| `BENCHMARK_VLLM_PER_WORKLOAD_SERVER` | Restart vLLM for each workload with its context length; when off, one server with an 8,192-token context runs all workloads | Same as `BENCHMARK_VLLM_AUTOCONFIG` |
| `BENCHMARK_VLLM_GPU_TP`, `VLLM_GPU_TENSOR_PARALLEL_SIZE` | Tensor-parallel size | From the GPU count and the model's attention heads |
| `BENCHMARK_VLLM_GPU_DP`, `VLLM_GPU_DATA_PARALLEL_SIZE` | Data-parallel size | `1` |
| `VLLM_GPU_MEMORY_UTILIZATION` | Value of `--gpu-memory-utilization`, also used by the memory fit check | `0.92` |
| `BENCHMARK_VLLM_MAX_NUM_SEQS` | Passed to vLLM as `--max-num-seqs` | Unset, so vLLM's default |
| `BENCHMARK_VLLM_MAX_NUM_BATCHED_TOKENS` | Passed to vLLM as `--max-num-batched-tokens` | Unset, so vLLM's default |
| `BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` | Run a model with the size estimated from the parameter count in the default model table when the Hugging Face metadata cannot be read | Unset, so the model is skipped |
| `BENCHMARK_VLLM_VCPUS` | Host vCPU count used for the load settings | The CPUs the process may run on |
| `BENCHMARK_VLLM_RAMPUP` | Ramp-up of the throughput stage in seconds | Computed |
| `GUIDELLM_GPU_SWEEP_SIZE`, `GUIDELLM_SWEEP_SIZE` | Number of sweep stages, at least 2 | Computed; legacy: 3 |
| `GUIDELLM__MAX_CONCURRENCY` | Maximum concurrent requests | Computed |
| `GUIDELLM__MAX_WORKER_PROCESSES` | GuideLLM worker processes | Computed |
| `GUIDELLM_MAX_REQUESTS` | Adds a request cap (`max_requests` constraint) to GuideLLM | Unset; legacy: 120 |
| `GUIDELLM_PROFILES` | `legacy`, `sync-throughput`, or `sync` runs one synchronous and one throughput run instead of a sweep | Unset |
| `GUIDELLM_THROUGHPUT_RATE` | Concurrency of the throughput run; applies only with `GUIDELLM_PROFILES` | `16` |
| `VLLM_PROBE_MODEL` | Model of `--probe-only` | `smol-135m` |
| `VLLM_PROBE_HEALTH_TIMEOUT_GPU_SEC` | Health check wait of `--probe-only` in seconds | `480` |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | Hugging Face cache directory | `/models`, set by the image |
| `TRACKER_QUIET` | Quiet output of `resource-tracker` | `true`, set by the image |
