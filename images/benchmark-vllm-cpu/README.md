# benchmark-vllm-cpu

This benchmark measures how fast CPUs serve LLMs to many concurrent requests.
It starts each default model with vLLM (`vllm serve`) on the CPU, from the
smallest to the largest, and sends synthetic requests to it over HTTP with
GuideLLM, from one request at a time up to peak load. It is the CPU variant of
the vLLM benchmarks for AMD64 CPUs with AVX-512 (Advanced Vector Extensions
512) and for Arm64 CPUs. The [vLLM manual](../../vllm-common/README.md)
describes the harness that all vLLM benchmarks share, and these docs cover how
this image runs it on CPUs.

This benchmark reports the following metrics for each model, workload, and
load stage, as one JSON line per metric and statistic:

- **Latency:** in milliseconds, each as the 50th, 95th, and 99th percentile
  (p50, p95, p99) and the mean:
  - TTFT (Time to First Token)
  - TPOT (Time per Output Token)
  - ITL (Inter-Token Latency)
  - End-to-end request latency
- **Throughput:** each as a mean:
  - Output tokens per second
  - Total tokens per second
  - Requests per second

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements for over 5,000 cloud server types, among
them [LLM inference speed](../benchmark-llm/README.md), measured with
`llama.cpp`. This benchmark measures LLM *serving* on CPUs instead: the
latency and throughput of a vLLM server under concurrent load, from one
request at a time up to peak load. In its note on the LLM inference speed
scores, the Navigator points readers who need high concurrency to the vLLM
benchmarks, but the Navigator does not show their results yet.

The vLLM benchmarks run the same harness in the following images:

- `benchmark-vllm-cpu`, this image: AMD64 CPUs with AVX-512, and Arm64 CPUs
- [`benchmark-vllm-cpu-avx2`](../benchmark-vllm-cpu-avx2/README.md): AMD64
  CPUs with AVX2 (Advanced Vector Extensions 2) but without AVX-512
- [`benchmark-vllm-gpu`](../benchmark-vllm-gpu/README.md): NVIDIA GPUs

## Limitations

These results compare LLM serving under load on CPUs. Note the following
limitations of CPU runs:

- **[AVX-512 on AMD64](./docs/usage.md#requirements-and-versions):** On AMD64 CPUs,
  this benchmark needs AVX-512. Without it, it exits with code 2, and its
  error message points to `benchmark-vllm-cpu-avx2`. Arm64 CPUs skip this
  check.
- **[Data Type Follows the CPU](./docs/workloads.md#data-type):** Models run
  in bfloat16 on CPUs that support it, and in float32 on the others, so two
  server types can serve the same model in different data types.
- **[Load Follows vCPU Count](./docs/workloads.md#load-settings):**
  Concurrency, GuideLLM workers, the number of load stages, and the vLLM batch
  limits scale with the host's vCPU count.
- **[One Thread per Physical Core](./docs/workloads.md#cpu-parallelism):**
  This benchmark runs vLLM with one thread on each physical core, so vLLM
  does not use the second SMT (Simultaneous Multithreading) thread of a core.
  The threads of each vLLM worker stay on one NUMA (Non-Uniform Memory Access)
  node.
- **[Skipped Models and Workloads](./docs/workloads.md#memory-fit):** A model
  or workload that does not fit in the memory of the NUMA nodes it would use
  is skipped. The 4-bit `llama-70b` model and the `long` workload never run on
  CPUs. The run still exits with code 0 as long as another model or workload
  wrote results. See also [Run Sequence](./docs/workloads.md#run-sequence).
- **[One Host, Synthetic Prompts](./docs/workloads.md#guidellm-sweep):** This
  benchmark measures serving latency and throughput on one host, with
  synthetic prompts of fixed lengths, sent by GuideLLM from the same container
  as the server. It does not measure the network latency between a client and
  the server, or the quality of the model output.

## Usage

Export your Hugging Face
[access token](https://huggingface.co/docs/hub/security-tokens) as `HF_TOKEN`
in the shell first;
`-e HF_TOKEN` passes it into the container. Then run the image via Docker with
the container settings of production:

```bash
docker run --rm --privileged --network host --shm-size=4g --memory=<limit> --memory-swap=<limit> --memory-swappiness=0 -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu:main > results.jsonl 2> results.log
```

Replace `<limit>` in both options with 85% of the host's available memory, as
production does; the same value for `--memory-swap` disables swap. Production
reads the memory that Linux reports as available, the `available` column of
`free -g` in GiB. Write the limit with a unit: for example, with 40 GiB
available, pass `--memory=34g --memory-swap=34g`.

The vLLM benchmarks fail with Docker's default 64 MiB `/dev/shm`, so a minimal
run needs only `--shm-size=4g` and `-e HF_TOKEN`; the other options match
production:

```bash
docker run --rm --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu:main > results.jsonl 2> results.log
```

See [Production Setup](./docs/usage.md#production-setup) before you compare your
results with production results.

This benchmark downloads the models from Hugging Face while it runs, and uses
`HF_TOKEN` for the downloads and the model metadata. Two default models,
`gemma-2b` and `llama-8b`, are gated models: they need a token from an account that
has accepted their licenses. Without such a token, this benchmark skips
them and runs the other models. The models are cached in `/models`, a Docker
volume; see
[Hugging Face Access and Model Cache](./docs/usage.md#hugging-face-access-and-model-cache)
to reuse them across runs.

A run fits its models and workloads into a 2-hour time budget. Sample
production runs took 28 minutes on an AWS (Amazon Web Services) `c8a.large`
and 1 hour 5 minutes on a `c8a.16xlarge`. See
[Time Budget](./docs/workloads.md#time-budget) for how the budget is split.

### Settings

Pass options after the image name, and environment variables with `-e` before
it. The following settings are the most common on CPUs; see the
[full option and variable reference](./docs/usage.md#options) for all of them
and their defaults:

- `--models` replaces the default models with Hugging Face model IDs or the
  short names of the default models, separated by spaces, such as
  `--models qwen-0.5b <owner>/<model>`.
- `BENCHMARK_VLLM_WORKLOADS` selects workloads by name, such as `chat`.
- `BENCHMARK_VLLM_CPU_TP` overrides the tensor-parallel size, and
  `BENCHMARK_VLLM_CPU_DP` the data-parallel size.
- `VLLM_CPU_DTYPE` overrides the data type, which is `bfloat16` or `float32`
  by default, depending on the CPU.
- `--plan-only` prints the host's resources and whether each model fits in
  memory, as JSON, without starting vLLM.
- `--version` prints the pinned versions, `vllm=0.27.1 guidellm=0.7.3`, and
  exits.

### Results

The process prints JSON Lines to stdout: one line per model, workload,
GuideLLM load stage, metric, and statistic. Logs go to stderr. The vLLM
server's own log goes to a temporary file inside the container, not to stdout
or stderr. The main fields of each line are the following:

- `measurement` names the metric: `ttft`, `tpot`, `itl`, and `e2el` for
  latency, and `output_throughput`, `total_throughput`, and
  `request_throughput` for throughput.
- `percentile` is `p50`, `p95`, `p99`, or `mean` for latency, and `null` for
  throughput. The `score` field holds the value, and `unit` is `ms`,
  `tokens/sec`, or `requests/sec`.
- `model` (the short name, such as `llama-8b`), `workload`, and `strategy`
  identify the run, and `model_id` holds the Hugging Face model ID. The
  `strategy` field is the GuideLLM load stage, `synchronous`, `throughput`, or
  `constant`, and `concurrency` is its mean number of concurrent requests. The
  `target_rate` field is the request rate of a `constant` stage, and `null`
  when GuideLLM gives no rate, as in production `synchronous` stages.
- `mode` is `cpu`, and `arch` is `amd64` or `arm64`.
- `tensor_parallel`, `data_parallel`, and `omp_threads_bind` show how the
  model was split across the NUMA nodes and CPU cores; see
  [CPU Parallelism](./docs/workloads.md#cpu-parallelism).
- `tuning` holds the load and vLLM server settings of the run, such as the
  data type in `dtype`.
- `vllm_version` and `guidellm_version` give the pinned versions.

See [Output Fields](./docs/usage.md#output-fields) for the CPU-specific
fields. For a full example, see the production
[output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.12xlarge/0/vllm/stdout)
of an AWS `c8a.12xlarge`.

The process exits with code 0 even when it skipped models or workloads, so
check which lines are present, and the reasons logged to stderr, which the
commands above save to `results.log`. It exits with code 1 when no model could
run or no line was written, and with code 2 on an AMD64 CPU without AVX-512.
See [Exit Codes](./docs/usage.md#exit-codes) for the codes and the log
messages of skipped models.
Resource usage is recorded with `resource-tracker`, which
[streams it](https://sparecores.com/article/resource-tracker-rs#streaming-to-sentinel)
to the [Spare Cores Sentinel](https://sentinel.sparecores.com/) platform for
the Spare Cores team's own monitoring. The resource usage is not published.

## Workloads

Each model runs the following two workloads of fixed synthetic prompt and
output lengths. This benchmark restarts `vllm serve` for each workload, with a
context length that fits it:

- **[`chat`](./docs/workloads.md#prompt-and-output-lengths):** 256 prompt
  tokens and 128 output tokens, with a context of 2,048 tokens.
- **[`rag`](./docs/workloads.md#prompt-and-output-lengths):** 1,024 prompt
  tokens and 256 output tokens, with a context of 4,096 tokens.

Each workload runs one GuideLLM sweep: a synchronous stage of one request at a
time, a throughput stage at the highest concurrency allowed, and, on hosts with
8 or more vCPUs and when the time budget allows, constant-rate stages between
the two. See
[Models](./docs/workloads.md#models) for the five default models that run on
CPUs, from 135 million to 14 billion parameters, and
[Run Sequence](./docs/workloads.md#run-sequence) for when a model is skipped
or stops the run.

On CPUs, this benchmark does the following for each model before it starts the
server:

- It splits the model across the NUMA nodes and binds its threads to physical
  cores; see [CPU Parallelism](./docs/workloads.md#cpu-parallelism).
- It checks that the model fits in the memory of each NUMA node; see
  [Memory Fit](./docs/workloads.md#memory-fit).
- It picks a data type that the CPU supports; see
  [Data Type](./docs/workloads.md#data-type).
- It derives the load and the vLLM batch limits from the host's vCPU count;
  see [Load Settings](./docs/workloads.md#load-settings).

## References

See the full list of [references](./docs/references.md) used in this
documentation. To change or rebuild this image, see the
[contributor guide](./docs/contributors.md).
