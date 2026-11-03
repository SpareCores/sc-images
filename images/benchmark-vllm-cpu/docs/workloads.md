# Workloads

## Prompt and Output Lengths

Each workload sends synthetic prompts of a fixed length and asks for a fixed
number of output tokens. This benchmark starts `vllm serve` with a context
length (`--max-model-len`) that fits the workload:

| Workload | Prompt tokens | Output tokens | Context length |
| --- | --- | --- | --- |
| `chat` | 256 | 128 | 2,048 |
| `rag` | 1,024 | 256 | 4,096 |

The `long` workload, with 4,096 prompt tokens, 512 output tokens, and an
8,192-token context, runs only on GPUs, in
[`benchmark-vllm-gpu`](../../benchmark-vllm-gpu/README.md). Set
`BENCHMARK_VLLM_WORKLOADS` to run fewer workloads, such as
`BENCHMARK_VLLM_WORKLOADS=chat`.

## Models

The following five default models run on CPUs, from smallest to largest:

| Short name | Hugging Face model | Parameters |
| --- | --- | --- |
| `smol-135m` | `HuggingFaceTB/SmolLM2-135M-Instruct` | 135 million |
| `qwen-0.5b` | `Qwen/Qwen2.5-0.5B-Instruct` | 0.5 billion |
| `gemma-2b` | `google/gemma-2-2b-it` | 2 billion |
| `llama-8b` | `meta-llama/Llama-3.1-8B-Instruct` | 8 billion |
| `phi-4` | `microsoft/phi-4` | 14 billion |

The `gemma-2b` and `llama-8b` models are gated; see
[Hugging Face Access and Model Cache](./usage.md#hugging-face-access-and-model-cache).
The sixth default model, `llama-70b`
(`unsloth/Llama-3.3-70B-Instruct-bnb-4bit`), is 4-bit quantized with
`bitsandbytes`. This benchmark treats its `bitsandbytes` serve options as
GPU-only, so it never runs that model on CPUs.

The `--models` option replaces this list with Hugging Face model IDs or the
short names above.

## Run Sequence

For each model, this benchmark first checks that the model fits in memory;
see [Memory Fit](#memory-fit). Then it runs the following steps for each
workload:

1. Computes the [load settings](#load-settings) for the workload's context
   length.
2. Skips the workload if its key-value cache does not fit in memory.
3. Starts `vllm serve` and waits up to 10 minutes for its health check.
4. Runs one [GuideLLM sweep](#guidellm-sweep), and converts its results into
   JSON lines.
5. Stops the server.

The following rules skip models or stop the run early:

- If a model's server does not pass its health check, the run stops, and no
  larger model is tried.
- If a model's highest mean output throughput is below 1 token per second, the
  run stops after that model.
- If a model does not fit in memory, it is skipped, and the run continues with
  the next model.

All of these paths exit with code 0 as long as the run wrote at least one JSON
line; see [Exit Codes](./usage.md#exit-codes).

## GuideLLM Sweep

GuideLLM runs in the same container as the vLLM server and sends its requests
through vLLM's OpenAI-compatible `/v1/completions` endpoint, so the results
include no network between hosts. The prompts are synthetic text of the
workload's token counts. GuideLLM generates them from a fixed random seed
(42), the starting value of its random number generator, so every run sends
the same prompts.

The sweep size, N, is the number of load stages a GuideLLM sweep runs. Each
stage is recorded as a `strategy` in the output:

1. `synchronous`: one request at a time, for the baseline latency.
2. `throughput`: as many concurrent requests as the maximum concurrency
   allows, for the peak throughput.
3. `constant`: N - 2 stages at constant request rates between the two.

Each stage stops at its time limit, and the throughput stage ramps up its
concurrency first. No stage has a request cap unless `GUIDELLM_MAX_REQUESTS`
or `GUIDELLM_MAX_REQUESTS_CPU` sets one. For example, on an AWS (Amazon Web
Services) `c8a.12xlarge`, each sweep had a size of 3, with 80 seconds per
stage
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.12xlarge/0/vllm/stdout)).
The GuideLLM command and its options are in the
[harness](../../../vllm-common/benchmark.py).

## Load Settings

By default (`BENCHMARK_VLLM_AUTOCONFIG=1`), this benchmark derives the GuideLLM
load and the vLLM server settings from the host's vCPU count `v`. Set
`BENCHMARK_VLLM_VCPUS` to use another count. On CPUs, it uses the following
formulas for the load, where `workers` is the number of vLLM workers, the
tensor-parallel size times the data-parallel size:

- **Maximum concurrency:** `max(floor, int(6 * v ** 0.65))`, where `floor` is
  `max(1, min(32, 4 * v), 4 * workers)`
- **GuideLLM worker processes:** `max(1, min(32, v, max_concurrency // 8))`
- **Sweep size:** `max(2, min(4, 2 + floor(log2(v) / 3)))`, that is, 2 below
  8 vCPUs, 3 from 8 to 63 vCPUs, and 4 from 64 vCPUs, then reduced until the
  sweep fits the [time budget](#time-budget) of the run
- **Seconds per stage:** the run budget divided by the sweep size
- **Throughput-stage ramp-up:** 15% of the seconds per stage, between 2s and
  8s, and short enough to leave at least 25s of the stage at peak concurrency

On CPUs, this benchmark also computes the following vLLM server settings and
passes them to `vllm serve`:

- **`--max-num-seqs`:** grows sub-linearly with `v`, and is capped by the CPU
  cores of each vLLM worker and by the key-value cache that fits in memory
- **`--max-num-batched-tokens`:**
  `min(max_model_len, max(2048, int(64 * v ** 0.5)))`, where `max_model_len`
  is the workload's context length
- **`--dtype`:** the data type; see [Data Type](#data-type)
- **`--gpu-memory-utilization`:** the memory fraction of each vLLM worker; see
  [Memory Fit](#memory-fit)
- **`--tensor-parallel-size` and `--data-parallel-size`:** see
  [CPU Parallelism](#cpu-parallelism)

Each line's `tuning` object records these settings. See
[Environment Variables](./usage.md#environment-variables) to override them.

## Time Budget

By default, a run has an overall budget of 2 hours, multiplied by
`--benchmark-timeout-scale`. This benchmark splits the budget as follows:

1. It reserves 600 seconds, plus 120 seconds for each CPU server start, one
   per runnable model and workload.
2. It divides the rest by the number of model and workload runs.
3. It limits each run of one model and workload, that is, one sweep, to 45 to
   240 seconds.

## CPU Parallelism

This benchmark does not run one vLLM worker across the whole host. It splits
each model across the NUMA (Non-Uniform Memory Access) nodes and CPU cores as
follows:

- **Tensor parallelism:** The tensor-parallel size is the NUMA node count when
  that count divides the model's attention head count, and 1 otherwise.
- **Data parallelism:** When the tensor-parallel size is below the NUMA node
  count, this benchmark runs at least one replica per NUMA node. It adds
  replicas until each vLLM worker has about 16 physical cores for models up to
  0.5 billion parameters, 32 up to 2 billion, 64 up to 8 billion, and 128 for
  larger models. It runs at most 32 replicas, and only as many as fit in the
  memory of each NUMA node, because each replica loads a full copy of the
  weights.
- **Thread binding:** This benchmark sets `VLLM_CPU_OMP_THREADS_BIND` to an
  explicit CPU list for each worker, with one logical CPU per physical core,
  all from one NUMA node. It finds the SMT (Simultaneous Multithreading)
  siblings of each core in sysfs (`thread_siblings_list`) and leaves them out.
  On hosts with at least 2 vCPUs, it keeps one CPU free for the server's
  frontend process. A `VLLM_CPU_OMP_THREADS_BIND` value other than `auto`
  replaces these lists.

For example, on the `c8a.12xlarge`, `smol-135m` ran as three replicas bound to
`0-15|16-31|32-46`, that is, 16, 16, and 15 cores
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c8a.12xlarge/0/vllm/stdout)).

## Memory Fit

Before it starts any server, this benchmark checks each model against the
memory of the host, using the model's metadata from Hugging Face:

- A model runs when at least one tensor- and data-parallel layout fits: the
  weights and a minimum key-value cache of each vLLM worker must fit in the
  free memory of the NUMA node that the worker runs on.
- Before each workload, the same check runs with the workload's context
  length, and a workload whose key-value cache does not fit is skipped.
- If the Hugging Face metadata cannot be read, the model is skipped, unless
  `BENCHMARK_VLLM_ALLOW_ESTIMATED_METADATA` is set.

On CPUs, vLLM's `--gpu-memory-utilization` option is a fraction of the total
memory of the worker's NUMA node, which each worker reserves on its own. This
benchmark divides it among the workers that share a NUMA node, so that their
sum stays within 70% of the node's free memory, and caps it at 0.50. When
several workers share a node, it lowers that budget by a further 10% and
leaves 0.8 GiB for startup. Set `VLLM_CPU_GPU_MEMORY_UTILIZATION` to override
the fraction. Run the image with `--plan-only` to see the plan of each model on
your server.

## Data Type

This benchmark runs the models in bfloat16 when `/proc/cpuinfo` lists the
`bf16` flag on Arm64 or `avx512_bf16` on AMD64, and in float32 otherwise. It
passes the data type to `vllm serve` as `--dtype`, and records it in the
`dtype` field of each line's `tuning` object. Set `VLLM_CPU_DTYPE` to use
another data type.

Two server types can therefore serve the same model in different data types.
This benchmark does not fall back to float16: on AWS Graviton2 CPUs, which lack
`bf16`, the vLLM server hangs with float16, as recorded in the
[harness](../../../vllm-common/benchmark.py).

## Legacy Settings

With `BENCHMARK_VLLM_AUTOCONFIG=0`, this benchmark uses static settings
instead of the formulas above. On CPUs, they include the following:

- A sweep size of 3
- A cap of 25 requests per stage
- One vLLM server for all workloads, unless
  `BENCHMARK_VLLM_PER_WORKLOAD_SERVER` is set

Set `GUIDELLM_PROFILES` or `GUIDELLM_CPU_PROFILES` to `legacy`,
`sync-throughput`, or `sync` to run one synchronous and one throughput run
instead of a sweep, with the throughput run at a concurrency of
`GUIDELLM_THROUGHPUT_RATE` (default: 8).
