# vLLM benchmarks

The vLLM benchmarks measure how fast CPUs and NVIDIA GPUs serve LLMs to many
concurrent requests. Each run starts the default models with vLLM
(`vllm serve`), from the smallest to the largest, and sends synthetic
requests to each over HTTP with GuideLLM, from one request at a time up to
peak load. Three images run the same [harness](./benchmark.py), each on its
own hardware.

The vLLM benchmarks write one JSON line per model, workload, load stage,
metric, and statistic. The lines report the following metrics:

- **Latency:** in milliseconds, each as the 50th, 95th, and 99th percentile
  (p50, p95, p99) and the mean: TTFT (Time to First Token), TPOT (Time per
  Output Token), ITL (Inter-Token Latency), and E2EL (End-to-End Request
  Latency)
- **Throughput:** each as a mean: output and total TPS (Tokens per Second),
  and RPS (Requests per Second)

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements for over 5,000 cloud server types, among
them [LLM inference speed](../images/benchmark-llm/README.md): how fast
`llama.cpp` processes prompts and generates text for one request at a time.
The vLLM benchmarks measure LLM *serving* instead: the latency and throughput
of a vLLM server under concurrent load, from one request at a time up to peak
load. In its note on the LLM inference speed scores, the Navigator points
readers who need high concurrency to the vLLM benchmarks, but the Navigator
does not show their results yet.

The harness runs in the following images, in the order that production tries
them:

- [`benchmark-vllm-gpu`](../images/benchmark-vllm-gpu/README.md): NVIDIA GPUs
  with CUDA (Compute Unified Device Architecture) compute capability 8.0 or
  newer
- [`benchmark-vllm-cpu`](../images/benchmark-vllm-cpu/README.md): AMD64 CPUs
  with AVX-512 (Advanced Vector Extensions 512), and Arm64 CPUs
- [`benchmark-vllm-cpu-avx2`](../images/benchmark-vllm-cpu-avx2/README.md):
  AMD64 CPUs with AVX2 (Advanced Vector Extensions 2) but without AVX-512

Each image's README covers what only that image does: its base image, its
hardware check, and its run command. This manual covers the rest, with CPU
and GPU sections where the two differ.

## Limitations

These results compare LLM serving under load. Note the following limitations
of all vLLM benchmarks; the image READMEs add the hardware limits of each
image:

- **[Load Follows vCPU Count](./docs/workloads.md#load-settings):**
  Concurrency, GuideLLM workers, and the number of load stages scale with the
  host's vCPU count, on GPUs too: not with the GPU count, GPU model, or GPU
  memory. On CPUs, the vLLM batch limits scale with it as well.
- **[Data Type Follows the CPU](./docs/workloads.md#data-type):** On CPUs,
  models run in bfloat16 on CPUs that support it, and in float32 on the
  others, so two server types can serve the same model in different data
  types.
- **[vLLM Server Defaults on GPUs](./docs/workloads.md#load-settings):** On
  GPUs, vLLM runs with its own batch limits and data type. The vLLM server
  settings recorded in each line's `tuning` object are not passed to
  `vllm serve`.
- **[One Thread per Physical Core](./docs/workloads.md#cpu-parallelism):** On
  CPUs, vLLM runs with one thread on each physical core, so it does not use
  the second SMT (Simultaneous Multithreading) thread of a core. The threads
  of each vLLM worker stay on one NUMA (Non-Uniform Memory Access) node.
- **[Partial GPU Use](./docs/workloads.md#gpu-parallelism):** A model can run
  on fewer GPUs than the server has, because its tensor-parallel size must
  divide its attention head count. On more than one GPU, the 70B model,
  `llama-70b`, uses pipeline parallelism instead.
- **[Skipped Models and Workloads](./docs/workloads.md#cpu-memory-fit):** A
  model or workload that does not fit in the memory it would use, of each NUMA
  node on CPUs or of each GPU on GPUs, is skipped. The 4-bit
  [`llama-70b`](./docs/workloads.md#models) model and the
  [`long`](./docs/workloads.md#prompt-and-output-lengths) workload never run
  on CPUs. The run still exits with code 0 as long as another model or
  workload wrote results. See also
  [GPU Memory Fit](./docs/workloads.md#gpu-memory-fit).
- **[One Host, Synthetic Prompts](./docs/workloads.md#guidellm-sweep):** The
  vLLM benchmarks measure serving latency and throughput on one host, with
  synthetic prompts of fixed lengths, sent by GuideLLM from the same container
  as the server. They do not measure the network latency between a client and
  the server, or the quality of the model output.

## Usage

Each image's README gives its `docker run` command. Every image needs the
following:

- **Shared memory:** `--shm-size=4g`, because the vLLM benchmarks fail with
  Docker's default 64 MiB `/dev/shm`.
- **Hugging Face token:** export your Hugging Face
  [access token](https://huggingface.co/docs/hub/security-tokens) as
  `HF_TOKEN` in the shell, and pass it with `-e HF_TOKEN`.
- **GPUs, for `benchmark-vllm-gpu` only:** `--gpus all`, with the NVIDIA
  driver and the NVIDIA Container Toolkit on the host.

Production also runs the container in privileged mode, with host networking
and a memory limit. See [Production Setup](./docs/usage.md#production-setup)
before you compare your results with production results.

The vLLM benchmarks download the models from Hugging Face while they run, and
use `HF_TOKEN` for the downloads and the model metadata. Two
[default models](./docs/workloads.md#models), `gemma-2b` and `llama-8b`, are
gated models: they need an access token from an account that has accepted
their licenses. Without one, the vLLM benchmarks skip them and run the other
models. The models are cached in `/models`, a Docker volume; see
[Hugging Face Access and Model Cache](./docs/usage.md#hugging-face-access-and-model-cache)
to reuse them across runs.

A run fits its models and workloads into a 2-hour time budget. Sample
production runs took from 28 minutes to 1 hour 40 minutes; see
[Run Duration](./docs/usage.md#run-duration).

### Settings

Pass options after the image name, and environment variables with `-e` before
it. The following settings are the most common; see the
[full option and variable reference](./docs/usage.md#options) for all of them
and their defaults:

- `--models` replaces the default models with Hugging Face model IDs or the
  short names of the default models, separated by spaces, such as
  `--models qwen-0.5b <owner>/<model>`.
- `BENCHMARK_VLLM_WORKLOADS` selects workloads by name, such as `chat`.
- `BENCHMARK_VLLM_CPU_TP` and `BENCHMARK_VLLM_GPU_TP` override the
  tensor-parallel size, and `BENCHMARK_VLLM_CPU_DP` and
  `BENCHMARK_VLLM_GPU_DP` the data-parallel size.
- `VLLM_CPU_DTYPE` overrides the data type on CPUs, which is `bfloat16` or
  `float32` by default, depending on the CPU.
- `VLLM_GPU_MEMORY_UTILIZATION` sets the fraction of each GPU's memory that
  vLLM may use (default: 0.92).
- `--plan-only` prints the host's resources and whether each model fits in
  memory, as JSON, without starting vLLM.
- `--version` prints the pinned versions, `vllm=0.27.1 guidellm=0.7.3`, and
  exits.

### Results

The process prints one JSON line to stdout per model, workload, GuideLLM load
stage, metric, and statistic. Logs go to stderr, including the load settings,
each `vllm serve` command, and the cause when a server fails to start. The
vLLM server's own log goes to a temporary file inside the container, not to
stdout or stderr. The main fields of each line are the following:

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
- `mode` is `cpu` or `gpu`, and `arch` is `amd64` or `arm64`.
- `tensor_parallel`, `pipeline_parallel`, and `data_parallel` show how the
  model was split across the NUMA nodes or GPUs.
- `tuning` holds the load and vLLM server settings of the run, such as the
  data type in `dtype`.
- `vllm_version` and `guidellm_version` give the pinned versions.

See [Output Fields](./docs/usage.md#output-fields) for the CPU-only and
GPU-only fields. For full examples, see the production output of an AWS
(Amazon Web Services)
[`c8a.12xlarge`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.12xlarge/0/vllm/stdout)
on CPUs and of an AWS
[`g5.16xlarge`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/stdout)
with one NVIDIA A10G GPU.

The process exits with code 0 even when it skipped models or workloads, so
check which lines are present in stdout and which reasons are logged to
stderr. It exits with code 1 when no model could run or no line was written;
see [Exit Codes](./docs/usage.md#exit-codes) for the other codes and the log
messages of skipped models. Resource usage is recorded with
`resource-tracker`, which
[streams it](https://sparecores.com/article/resource-tracker-rs#streaming-to-sentinel)
to the [Spare Cores Sentinel](https://sentinel.sparecores.com/) platform for
the Spare Cores team's own monitoring. The resource usage is not published.

## Workloads

Each model runs the following workloads of fixed synthetic prompt and output
lengths. The vLLM benchmarks restart `vllm serve` for each workload, with a
context length that fits it:

- **[`chat`](./docs/workloads.md#prompt-and-output-lengths):** 256 prompt
  tokens and 128 output tokens, with a context of 2,048 tokens.
- **[`rag`](./docs/workloads.md#prompt-and-output-lengths):** 1,024 prompt
  tokens and 256 output tokens, with a context of 4,096 tokens.
- **[`long`](./docs/workloads.md#prompt-and-output-lengths):** 4,096 prompt
  tokens and 512 output tokens, with a context of 8,192 tokens. Only GPU runs
  include it.

Each workload runs one GuideLLM sweep: a synchronous stage of one request at a
time, a throughput stage at the highest concurrency allowed, and, when the
time budget allows, constant-rate stages between the two. On CPUs, hosts with
fewer than 8 vCPUs run no constant-rate stages. See
[Models](./docs/workloads.md#models) for the six default models, from 135
million to 70 billion parameters, five of which run on CPUs, and
[Run Sequence](./docs/workloads.md#run-sequence) for when a model is skipped
or stops the run.

Before it starts the server for each model, the harness does the following:

- It splits the model across the NUMA nodes and CPU cores, or across the
  GPUs; see [CPU Parallelism](./docs/workloads.md#cpu-parallelism) and
  [GPU Parallelism](./docs/workloads.md#gpu-parallelism).
- It checks that the model fits in the memory of each NUMA node or each GPU;
  see [CPU Memory Fit](./docs/workloads.md#cpu-memory-fit) and
  [GPU Memory Fit](./docs/workloads.md#gpu-memory-fit).
- On CPUs, it picks a data type that the CPU supports; see
  [Data Type](./docs/workloads.md#data-type).
- It derives the load from the host's vCPU count, and on CPUs also the vLLM
  batch limits; see [Load Settings](./docs/workloads.md#load-settings).

## Design History

The harness first ran fixed load settings on every host. It now derives the
load from the host's vCPU count and fits all runs into a 2-hour budget, so
that a host with hundreds of vCPUs gets more load without running for days.
An early version also scaled the throughput stage's ramp-up with the vCPU
count, which left almost no time at peak concurrency on hosts with more than
96 vCPUs; the ramp-up is now a share of the stage, capped at a few seconds.
The fixed settings remain as [Legacy Settings](./docs/workloads.md#legacy-settings)
for comparisons with older results. See the [changelog](./CHANGELOG.md) for
the details.

## References

Common terms, such as GuideLLM, TTFT, and tensor parallelism, are defined in
the [glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs.
To change or rebuild the vLLM images, see the
[contributor guide](./docs/contributors.md).
