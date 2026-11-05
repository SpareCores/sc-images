# Workloads

## Prompt and Output Lengths

Each workload sends synthetic prompts of a fixed length and asks for a fixed
number of output tokens. The harness starts `vllm serve` with a context
length (`--max-model-len`) that fits the workload:

| Workload | Prompt tokens | Output tokens | Context length | Runs on |
| --- | --- | --- | --- | --- |
| `chat` | 256 | 128 | 2,048 | CPUs and GPUs |
| `rag` | 1,024 | 256 | 4,096 | CPUs and GPUs |
| `long` | 4,096 | 512 | 8,192 | GPUs only |

Set `BENCHMARK_VLLM_WORKLOADS` to run fewer workloads, such as
`BENCHMARK_VLLM_WORKLOADS=chat`.

## Models

The six default models run from smallest to largest:

| Short name | Hugging Face model | Parameters | Attention heads |
| --- | --- | --- | --- |
| `smol-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135 million | 9 |
| `qwen-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 billion | 14 |
| `gemma-2b` | `google/gemma-2-2b-it` | 2 billion | 8 |
| `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8 billion | 32 |
| `phi-4` | `microsoft/phi-4` | 14 billion | 40 |
| `llama-70b` | `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` | 70 billion | From its `config.json` |

The `gemma-2b` and `llama-8b` models are gated models; see
[Hugging Face Access and Model Cache](./usage.md#hugging-face-access-and-model-cache).
The head counts above are fallbacks: the harness reads each model's head count
from its `config.json` on Hugging Face first.

The `llama-70b` model is 4-bit quantized with `bitsandbytes`, and vLLM serves
it with `--quantization bitsandbytes --load-format bitsandbytes`. The harness
treats these serve options as GPU-only, so it never runs that model on CPUs,
where five default models run.

The `--models` option replaces this list with Hugging Face model IDs or the
short names above.

## Run Sequence

For each model, the harness first checks that the model fits in memory; see
[CPU Memory Fit](#cpu-memory-fit) and [GPU Memory Fit](#gpu-memory-fit). Then
it runs the following steps for each workload:

1. Computes the [load settings](#load-settings) for the workload's context
   length.
2. Skips the workload if the model's memory plan for that context length does
   not fit.
3. Starts `vllm serve` and waits for its health check, up to 10 minutes on
   CPUs and 5 minutes on GPUs.
4. Runs one [GuideLLM sweep](#guidellm-sweep), and converts its results into
   JSON lines.
5. Stops the server: it sends `SIGTERM` to the server's process group, and
   `SIGKILL` after 30 seconds.

The following rules skip models or stop the run early:

- If a model does not fit in memory, it is skipped, and the run continues with
  the next model.
- If a model's server does not pass its health check, the run stops, and no
  larger model is tried.
- If a model's highest mean output throughput is above 0 but below 1 token per
  second, the run stops after that model.
- If a GuideLLM run fails or times out, the lines of its partial results are
  still written, if it saved any.

In all of these cases, the run exits with code 0 as long as it wrote at least
one JSON line; see [Exit Codes](./usage.md#exit-codes).

## GuideLLM Sweep

GuideLLM runs in the same container as the vLLM server and sends its requests
to `http://127.0.0.1:8000` through vLLM's OpenAI-compatible
`/v1/completions` endpoint, so the results include no network latency between
hosts. The prompts are synthetic text of the workload's token counts. GuideLLM
generates them from a fixed random seed (42), the starting value of its random
number generator, so every run sends the same prompts.

The sweep size, N, is the number of load stages a GuideLLM sweep runs. Each
stage is recorded as a `strategy` in the output:

1. `synchronous`: one request at a time, for the baseline latency.
2. `throughput`: as many concurrent requests as the maximum concurrency
   allows, for the peak throughput.
3. `constant`: N - 2 stages at constant request rates between the two.

Each stage stops at its time limit, and the throughput stage ramps up its
concurrency first. No stage has a request cap unless `GUIDELLM_MAX_REQUESTS`
or, on CPUs, `GUIDELLM_MAX_REQUESTS_CPU` sets one. The GuideLLM command and
its options are in the [harness](../benchmark.py).

For example, on an AWS (Amazon Web Services)
[`c8a.12xlarge`](https://sparecores.com/server/aws/c8a.12xlarge) CPU server,
each sweep had a size of 3, with 80 seconds per stage
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.12xlarge/0/vllm/stdout)).
On an AWS `g5.16xlarge` with one NVIDIA A10G GPU, the `chat` sweep had a size
of 6: one synchronous, one throughput, and four constant-rate stages
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/stdout)).

## Load Settings

By default (`BENCHMARK_VLLM_AUTOCONFIG=1`), the harness derives the GuideLLM
load from the host's vCPU count `v`, on GPUs too: not from the GPU count, the
GPU model, or GPU memory. Set `BENCHMARK_VLLM_VCPUS` to use another count.
Each line's `tuning` object records the settings below. See
[Environment Variables](./usage.md#environment-variables) to override them.

### On CPUs

On CPUs, the harness uses the following formulas for the load, where
`workers`, the number of vLLM workers, is the tensor-parallel size times the
data-parallel size:

- **Maximum concurrency:** `max(min_concurrency, int(6 * v ** 0.65))`, where
  `min_concurrency` is `max(1, min(32, 4 * v), 4 * workers)`
- **GuideLLM worker processes:** `max(1, min(32, v, max_concurrency // 8))`
- **Sweep size:** `max(2, min(4, 2 + floor(log2(v) / 3)))`, that is, 2 below
  8 vCPUs, 3 from 8 to 63 vCPUs, and 4 from 64 vCPUs, then reduced, down to
  2, while the stages would be shorter than their minimum
- **Seconds per stage:** the [time limit](#time-budget) of the sweep divided
  by the sweep size, at most 240 seconds, and at least 25 seconds for `chat`
  and 44 seconds for `rag`, a minimum that grows with the context length
- **Throughput-stage ramp-up:** 15% of the seconds per stage, between 2s and
  8s, and short enough to leave at least 25s of the stage at peak concurrency
- **Warmup:** `10`, passed to GuideLLM as `warmup`

On CPUs, the harness also computes the following vLLM server settings and
passes them to `vllm serve`:

- **`--max-num-seqs`:** grows sub-linearly with `v`, and is capped by the CPU
  cores of each vLLM worker and by the key-value cache that fits in memory
- **`--max-num-batched-tokens`:**
  `min(max_model_len, max(2048, int(64 * v ** 0.5)))`, where `max_model_len`
  is the workload's context length
- **`--dtype`:** the data type; see [Data Type](#data-type)
- **`--gpu-memory-utilization`:** the memory fraction of each vLLM worker; see
  [CPU Memory Fit](#cpu-memory-fit)
- **`--tensor-parallel-size` and `--data-parallel-size`:** see
  [CPU Parallelism](#cpu-parallelism)

### On GPUs

On GPUs, the harness uses the following formulas for the load:

- **Maximum concurrency:** `max(64, int(8 * v ** 0.60))`
- **GuideLLM worker processes:** `min(max(8, int(3 * v ** 0.50)), max(4, v))`
- **Sweep size:** `max(2, min(3 + floor(log2(v) / 2), 6 + floor(log2(v) / 3)))`,
  then reduced until the sweep fits the [time budget](#time-budget) of the run
- **Seconds per stage:** the run budget divided by the sweep size, at least
  `max(25, 22 * max_model_len / 2048)`, and at most 240
- **Throughput-stage ramp-up in seconds:**
  `min(10, max(2, 0.15 * seconds per stage))`
- **Warmup:** `0.05`, so GuideLLM leaves the first 5% of each stage out of
  the results, to exclude startup effects
  ([GuideLLM](https://github.com/vllm-project/guidellm/blob/v0.7.3/README.md))

Production runs used the following settings. The workload columns give the
sweep size times the seconds per stage:

| Server type | vCPUs | Maximum concurrency | Workers | `chat` | `rag` | `long` |
| --- | --- | --- | --- | --- | --- | --- |
| AWS `g5.16xlarge` | 64 | 97 | 24 | 6 x 40s | 5 x 48s | 2 x 120s |
| AWS `g5.8xlarge` | 32 | 64 | 16 | 5 x 48s | 5 x 48s | 2 x 120s |

The `long` workload also ran two stages of 120 seconds on the UpCloud
`GPU-32xCPU-384GB-3xL40S`
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/upcloud/GPU-32xCPU-384GB-3xL40S/0/vllm/stdout)).
Its minimum of 88 seconds per stage leaves room for only two stages in a run
budget of 240 seconds.

On GPUs, the harness also computes the vLLM server settings `max_num_seqs`,
`max_num_batched_tokens`, `dtype`, and `kv_memory_util`, and records them in
the `tuning` object, but does not pass them to `vllm serve`, which uses its
own defaults. Set `BENCHMARK_VLLM_MAX_NUM_SEQS` or
`BENCHMARK_VLLM_MAX_NUM_BATCHED_TOKENS` to pass `--max-num-seqs` or
`--max-num-batched-tokens`.

## Time Budget

A run has an overall budget of 2 hours. The harness splits it as follows:

1. It reserves 600 seconds, plus time for each server start, one per runnable
   model and workload: 120 seconds on CPUs and 60 seconds on GPUs.
2. It divides the rest by the number of sweeps, one per model and workload.
3. It limits each sweep to 45 to 240 seconds.

For example, on the `g5.16xlarge`, the log shows
`overall=7200s reserve=1320s runs=12 per_run=240s`
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/stderr)).

Before it starts each model, workload, and GuideLLM process, the harness
checks the elapsed time. After 2 hours, it starts nothing new: it logs
"Overall timeout reached" before the next model, and the models it has not
reached do not run. This can happen, for example, when the 45-second minimum
per sweep pushes the plan past 2 hours.

The `--benchmark-timeout-scale` option takes a whole number, and values below
1 act as 1. It multiplies the following limits:

- The overall budget, before the split above. Each sweep stays limited to 240
  seconds, so its stages get longer only when its budget was below 240
  seconds without the scale.
- The time limit of each GuideLLM process.
- With `BENCHMARK_VLLM_AUTOCONFIG=0`, the time limit of each stage.

The option does not scale the 2-hour stop, so a value above 1 does not let a
run go on longer.

## CPU Parallelism

On CPUs, the harness does not run one vLLM worker across the whole host. It
splits each model across the NUMA (Non-Uniform Memory Access) nodes and CPU
cores as follows:

- **Tensor parallelism:** The tensor-parallel size is the NUMA node count when
  that count divides the model's attention head count, and 1 otherwise.
- **Data parallelism:** The data-parallel size is the number of model
  replicas, each with a full copy of the weights. The harness sets it as
  follows:
  - At least one replica per NUMA node when the tensor-parallel size is below
    the NUMA node count.
  - On every host, more replicas if the physical core count, divided by the
    tensor-parallel size and by the cores per worker for the model size (see
    the table below), gives more.
  - At most 32 replicas, and only as many as fit in the memory of each NUMA
    node.
- **Thread binding:** The harness sets `VLLM_CPU_OMP_THREADS_BIND` to an
  explicit CPU list for each worker, with one logical CPU per physical core,
  all from one NUMA node. It finds the SMT (Simultaneous Multithreading)
  siblings of each core in sysfs (`thread_siblings_list`) and leaves them out.
  On hosts with at least 2 vCPUs, it keeps one CPU free for the server's
  frontend process. A `VLLM_CPU_OMP_THREADS_BIND` value other than `auto`
  replaces these lists.

The cores per worker depend on the model size:

| Model size | Physical cores per worker |
| --- | --- |
| Up to 0.5 billion parameters | 16 |
| Up to 2 billion parameters | 32 |
| Up to 8 billion parameters | 64 |
| Larger | 128 |

For example, on the `c8a.12xlarge`, `smol-135m` ran as three replicas: 48
physical cores divided by 16 cores per worker. They were bound to
`0-15|16-31|32-46`, that is, 16, 16, and 15 cores, with one CPU left for the
frontend
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.12xlarge/0/vllm/stdout)).

## GPU Parallelism

On GPUs, the harness does not always use every GPU of the server. It splits
each model across the GPUs as follows:

- **Tensor parallelism:** The tensor-parallel size is the largest value, up to
  the GPU count, that divides the model's attention head count. The server
  starts with `--tensor-parallel-size <size> --gpu-memory-utilization 0.92`.
- **Data parallelism:** One replica, unless `BENCHMARK_VLLM_GPU_DP` sets more.
- **Pipeline parallelism:** On more than one GPU, the `bitsandbytes`-quantized
  `llama-70b` starts with `--pipeline-parallel-size <GPU count>` instead, and
  without `--gpu-memory-utilization`. `VLLM_GPU_MEMORY_UTILIZATION` then
  affects only its memory fit check.

For example, on the UpCloud server with three NVIDIA L40S GPUs, `smol-135m`
(9 attention heads) ran on all three GPUs, and `qwen-0.5b` (14 heads) and the
models up to `phi-4` on two
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/upcloud/GPU-32xCPU-384GB-3xL40S/0/vllm/stdout)).
The `llama-70b` model ran with `--pipeline-parallel-size 3`
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/upcloud/GPU-32xCPU-384GB-3xL40S/0/vllm/stderr)).

For `llama-70b` on more than one GPU, read `pipeline_parallel` in the output:
`tensor_parallel` still shows the size that the head-count rule gives, and the
log line of the parallel settings shows it too, although the server runs
without tensor parallelism.

## CPU Memory Fit

Before it starts any server, the harness checks each model against the memory
of the host, using the model's metadata from Hugging Face. The check reads the
host's memory, not the container's `--memory` limit, so the plan can count on
memory that the container cannot use. The check works as follows:

- A model runs when at least one tensor- and data-parallel layout fits: the
  weights and a minimum key-value cache of each vLLM worker must fit in the
  free memory of the NUMA node that the worker runs on.
- When the serving data type is wider than the data type in the model's
  configuration, the check scales the size of the weights by the same ratio.
  For example, it doubles the weights of a bfloat16 model that runs in
  float32; see [Data Type](#data-type).
- Before each workload, the same check runs with the workload's context
  length, and a workload whose key-value cache does not fit is skipped.
- If the Hugging Face metadata cannot be read, the model is skipped, unless
  `BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` is set.

On CPUs, vLLM's `--gpu-memory-utilization` option sets the fraction of the
NUMA node's total memory that each worker reserves for itself. The harness
splits the fraction among the workers that share a NUMA node, so that together
they reserve at most 70% of the node's free memory, and caps each worker's
fraction at 0.50. It leaves 0.8 GiB of each node for startup, and when several
workers share a node, it lowers the 70% budget by a further 10%. Set
`VLLM_CPU_GPU_MEMORY_UTILIZATION` to override the fraction. Run the image with
`--plan-only` to see the plan of each model on a host.

## GPU Memory Fit

Before it starts any server, the harness estimates how much memory each model
needs on each GPU, without downloading the weights:

- The weight size comes from the file metadata on Hugging Face, and the layer
  count, attention and key-value heads, hidden size, and data type from the
  model's `config.json`.
- Each GPU needs the following, added together:
  - the weight size plus 10%, divided by the tensor-parallel size times the
    pipeline-parallel size
  - the key-value cache of one sequence of the full context length, rounded
    up to blocks of 16 tokens
  - 0.5 GiB
- A model fits when every GPU it uses has that much memory, both within 92% of
  its total memory and within 95% of its free memory. Each GPU is checked on its
  own, not the total memory of all GPUs.

A model that does not fit is skipped, and the run continues with the next one.
Before each workload, the same check runs with the workload's context length,
and a workload that does not fit is skipped. If the Hugging Face metadata
cannot be read, the model is skipped, unless
`BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` is set.

For example, on the `g5.16xlarge`, whose A10G GPU has 22 GiB of memory,
`phi-4` and `llama-70b` were skipped for insufficient memory
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/stderr)).
Run the image with `--plan-only` to see the plan of each model on a host.

## Data Type

On CPUs, the harness runs the models in bfloat16 when `/proc/cpuinfo` lists
the `bf16` flag on Arm64 or `avx512_bf16` on AMD64, and in float32 otherwise.
It passes the data type to `vllm serve` as `--dtype`, and records it in the
`dtype` field of each line's `tuning` object. Set `VLLM_CPU_DTYPE`, such as
`-e VLLM_CPU_DTYPE=bfloat16`, to use another data type; the
[memory check](#cpu-memory-fit) then uses the data type set.

Two server types can therefore serve the same model in different data types.
An AMD64 CPU without AVX-512 has no `avx512_bf16` flag, so it runs the models
in float32, which takes 4 bytes per weight, twice as many as bfloat16. Such a
host can fit fewer models than a host with the same memory that supports
bfloat16. The harness does not fall back to float16: on AWS Graviton2 CPUs,
which lack `bf16`, the vLLM server hangs with float16, as recorded in the
[harness](../benchmark.py).

On GPUs, the harness does not pass `--dtype`, so vLLM picks the data type.

## Legacy Settings

With `BENCHMARK_VLLM_AUTOCONFIG=0`, the harness uses static settings instead
of the formulas in [Load Settings](#load-settings):

| Setting | On CPUs | On GPUs |
| --- | --- | --- |
| Sweep size | 3 | 3 |
| Maximum concurrency | 512 | 512 |
| GuideLLM worker processes | 10 | 10 |
| Throughput-stage ramp-up | None | None |
| Seconds per stage | `45 + 12 * <parameters in billions>` | `40 + 8 * <parameters in billions>` |
| Request cap per stage | 25, set with `GUIDELLM_MAX_REQUESTS_CPU` | 120, set with `GUIDELLM_MAX_REQUESTS` |

Legacy runs also use one vLLM server for all workloads, unless
`BENCHMARK_VLLM_PER_WORKLOAD_SERVER` is set.

With autoconfig on or off, set `GUIDELLM_PROFILES`, or on CPUs
`GUIDELLM_CPU_PROFILES`, to any of `legacy`, `sync-throughput`, or `sync` (all
three do the same) to run one synchronous and one throughput stage instead of
a sweep, with the throughput stage at a concurrency of
`GUIDELLM_THROUGHPUT_RATE` (default: 8 on CPUs, 16 on GPUs).
