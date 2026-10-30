# benchmark-vllm-gpu

This benchmark measures how fast NVIDIA GPUs serve LLMs to many concurrent
requests, with vLLM and GuideLLM. It starts each default model with
`vllm serve`, from the smallest to the largest, and sends it synthetic requests
over HTTP with GuideLLM, from one request at a time up to peak load. It is the
GPU variant of the vLLM benchmarks: the
[vLLM manual](../../vllm-common/README.md) describes the harness they share,
and these docs cover how it runs on GPUs.

This benchmark reports the following metrics for each model, workload, and
load stage, as one JSON line per metric and statistic:

- **Latency:** TTFT (Time to First Token), TPOT (Time per Output Token), ITL
  (Inter-Token Latency), and end-to-end request latency in milliseconds, each
  as p50, p95, p99, and mean.
- **Throughput:** output tokens and total tokens per second, and requests per
  second, each as a mean.

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements for over 5,000 cloud server types, among
them [LLM inference speed](../benchmark-llm/README.md): how fast `llama.cpp`
processes prompts and generates text for one request at a time. This
benchmark measures LLM serving instead: the latency and throughput of a vLLM
server under concurrent load, from one request at a time up to peak load.

## Limitations

These results compare LLM serving under load on NVIDIA GPUs. Note the
following limitations of GPU runs:

- **[NVIDIA GPUs Only](./docs/usage.md#gpu-run):** This benchmark needs
  NVIDIA GPUs with CUDA (Compute Unified Device Architecture) compute
  capability 8.0 or newer, the Ampere generation or later. If no GPU is
  visible or every visible GPU is older, it exits with code 1.
- **[Partial GPU Use](./docs/workloads.md#gpu-parallelism):** A model can run
  on fewer GPUs than the server has, because its tensor-parallel size must
  divide its attention head count. The 70B model uses pipeline parallelism
  instead.
- **[Load Follows vCPU Count](./docs/workloads.md#load-settings):**
  Concurrency, GuideLLM workers, and the number of load stages scale with the
  host's vCPU count, not with the GPU count, GPU model, or GPU memory.
- **[vLLM Server Defaults](./docs/workloads.md#load-settings):** On GPUs, vLLM
  runs with its own batch limits and data type. The vLLM server settings
  recorded in each row's `tuning` object are not passed to `vllm serve`.
- **[Skipped Models and Workloads](./docs/workloads.md#gpu-memory-fit):** A
  model or workload that does not fit in the memory of each GPU it would use is
  skipped. The run still exits with code 0 as long as another model or
  workload wrote results.
- **[One Host, Synthetic Prompts](./docs/workloads.md#guidellm-sweep):** This
  benchmark measures serving latency and throughput on one host, with
  synthetic prompts of fixed lengths. It does not measure the network latency
  between a client and the server, or the quality of the model output.

## Usage

Export your Hugging Face access token as `HF_TOKEN` in the shell first;
`-e HF_TOKEN` passes it into the container. Then run the image via Docker with
the GPUs passed and a larger shared memory:

```bash
docker run --rm --gpus all --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-gpu:main > results.jsonl
```

The host needs the NVIDIA driver and the NVIDIA Container Toolkit for
`--gpus`. Without `--shm-size=4g`, the vLLM benchmarks fail with Docker's
default 64 MiB `/dev/shm`. See [GPU Run](./docs/usage.md#gpu-run) for the GPU
check and the supported architectures.

This benchmark downloads the models from Hugging Face while it runs, and uses
`HF_TOKEN` for the downloads and the model metadata. A gated model needs a
token from an account that has accepted the model's license. The
models are cached in `/models`, a Docker volume; see
[Hugging Face Access and Model Cache](./docs/usage.md#hugging-face-access-and-model-cache)
to reuse them across runs.

A run fits its models and workloads into a 2-hour time budget. Sample
production runs took 1 hour 14 minutes and 1 hour 40 minutes. See
[Run Duration](./docs/usage.md#run-duration) for how the budget is split.

Production runs the container privileged, with host networking, and with
every GPU passed. A plain `docker run` uses neither privileged mode nor host
networking. See [Production Setup](./docs/usage.md#production-setup) before you
compare your results with production results.

### Settings

Pass options after the image name and environment variables with `-e`. The
following settings are the most common on GPUs; see the
[full option and variable reference](./docs/usage.md#options) for all of them
and their defaults:

- `--models` replaces the default models with Hugging Face model IDs or the
  short names of the default models.
- `BENCHMARK_VLLM_WORKLOADS` selects workloads by name, such as `chat,rag`.
- `BENCHMARK_VLLM_GPU_TP` overrides the tensor-parallel size, and
  `BENCHMARK_VLLM_GPU_DP` the data-parallel size (default: 1).
- `VLLM_GPU_MEMORY_UTILIZATION` sets the fraction of each GPU's memory that
  vLLM may use (default: 0.92).
- `--plan-only` prints the host and the memory plan of each model as JSON
  without starting vLLM.
- `--version` prints the pinned versions, `vllm=0.27.1 guidellm=0.7.3`, and
  exits.

### Results

The process prints JSON Lines to stdout: one line per model, workload,
GuideLLM load stage, metric, and statistic. Logs go to stderr, including the
load settings, each `vllm serve` command, and the cause when a server fails to
start. The output of the vLLM server goes
to a temporary file inside the container, not to stdout or stderr. The main
fields of each line are the following:

- `measurement` names the metric: `ttft`, `tpot`, `itl`, and `e2el` for
  latency, and `output_throughput`, `total_throughput`, and
  `request_throughput` for throughput.
- `percentile` is `p50`, `p95`, `p99`, or `mean` for latency, and `null` for
  throughput. The `score` field holds the value, and `unit` is `ms`,
  `tokens/sec`, or `requests/sec`.
- `model` (the short name, such as `llama-8b`; `model_id` holds the Hugging
  Face ID), `workload`, and `strategy` identify the run. The `strategy` field
  is the GuideLLM load stage, `synchronous`, `throughput`, or `constant`, and
  `concurrency` is its mean number of concurrent requests. The `target_rate`
  field is the request rate of a `constant` stage, which tells those stages
  apart.
- `mode` is `gpu`. The `gpu_count`, `gpu_model` (the name of the first GPU),
  and `total_vram_gb` (the memory of all GPUs in GiB) fields describe the GPUs.
- `tensor_parallel`, `pipeline_parallel`, and `data_parallel` show how the
  model was split across the GPUs; see
  [GPU Parallelism](./docs/workloads.md#gpu-parallelism).
- `vllm_version` and `guidellm_version` give the pinned versions.

See [Output Fields](./docs/usage.md#output-fields) for the GPU-specific
fields. For example, see the production
[output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/stdout)
of an AWS (Amazon Web Services) `g5.16xlarge` with 64 vCPUs and one NVIDIA
A10G GPU.
[Spare Cores Navigator](https://docs.sparecores.com/overview) does not
publish the results of the vLLM benchmarks yet.

The process exits with code 0 even when it skipped models or workloads, so
check which lines are present, and the reasons logged to stderr. It exits with
code 1 when no model could run or no line was written; see
[Exit Codes](./docs/usage.md#exit-codes). Resource usage is recorded with
`resource-tracker`, which
[streams it](https://sparecores.com/article/resource-tracker-rs#streaming-to-sentinel)
to the [Spare Cores Sentinel](https://sentinel.sparecores.com/) platform for
the team's own monitoring. It is not published.

## Workloads

Each model runs the following three workloads of fixed synthetic prompt and
output lengths. This benchmark restarts `vllm serve` for each workload, with a
context length that fits it:

- **[`chat`](./docs/workloads.md#prompt-and-output-lengths):** 256 prompt
  tokens and 128 output tokens, with a 2,048-token context.
- **[`rag`](./docs/workloads.md#prompt-and-output-lengths):** 1,024 prompt
  tokens and 256 output tokens, with a 4,096-token context.
- **[`long`](./docs/workloads.md#prompt-and-output-lengths):** 4,096 prompt
  tokens and 512 output tokens, with an 8,192-token context. Only GPU runs
  include it.

Each workload runs one GuideLLM sweep: a synchronous stage of one request at a
time, a throughput stage at the highest concurrency allowed, and, when the time
budget allows, constant-rate stages between the two. See [Models](./docs/workloads.md#models) for the six
default models, from 135 million to 70 billion parameters, and
[Run Sequence](./docs/workloads.md#run-sequence) for when a model is skipped
or stops the run.

On GPUs, this benchmark picks how to split each model across the GPUs, checks
that it fits in each GPU's memory before starting it, and derives the load from
the host's vCPU count. See
[GPU Parallelism](./docs/workloads.md#gpu-parallelism),
[GPU Memory Fit](./docs/workloads.md#gpu-memory-fit), and
[Load Settings](./docs/workloads.md#load-settings) for the rules.

## References

See the full list of [references](./docs/references.md) used in this
documentation. To change or rebuild this image, see the
[contributor guide](./docs/contributors.md).
