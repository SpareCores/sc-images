# Workloads

## Prompt and Output Lengths

Each workload sends synthetic prompts of a fixed length and asks for a fixed
number of output tokens. This benchmark starts `vllm serve` with a context
length (`--max-model-len`) that fits the workload:

| Workload | Prompt tokens | Output tokens | Context length |
| --- | --- | --- | --- |
| `chat` | 256 | 128 | 2,048 |
| `rag` | 1,024 | 256 | 4,096 |
| `long` | 4,096 | 512 | 8,192 |

The `long` workload runs only on GPUs; the CPU images run `chat` and `rag`.
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

The `llama-70b` model is 4-bit quantized with `bitsandbytes`, and vLLM serves
it with `--quantization bitsandbytes --load-format bitsandbytes`. The head
counts above are fallbacks: this benchmark reads each model's head count from
its `config.json` on Hugging Face first.

The `--models` option replaces this list with Hugging Face model IDs or the
short names above.

## Run Sequence

For each model, this benchmark first checks that the model fits in GPU memory;
see [GPU Memory Fit](#gpu-memory-fit). Then it runs the following steps for
each workload:

1. Computes the [load settings](#load-settings) for the workload's context
   length.
2. Skips the workload if the model's memory plan for that context length does
   not fit.
3. Starts `vllm serve` and waits up to 5 minutes for its health check.
4. Runs one [GuideLLM sweep](#guidellm-sweep), and converts its results into
   JSON lines.
5. Stops the server: it sends `SIGTERM` to the server's process group, and
   `SIGKILL` after 30 seconds.

The following rules skip models or stop the run early:

- If a model's server does not pass its health check, the run stops, and no
  larger model is tried.
- If a model's highest mean output throughput is above 0 but below 1 token per
  second, the run stops after that model.
- If a model does not fit in GPU memory, it is skipped, and the run continues
  with the next model.
- If a GuideLLM run fails or times out, the lines of its partial results are
  still written, if it saved any.

All of these paths exit with code 0 as long as the run wrote at least one JSON
line; see [Exit Codes](./usage.md#exit-codes).

## GuideLLM Sweep

GuideLLM runs in the same container as the vLLM server and sends its requests
to `http://127.0.0.1:8000` through vLLM's OpenAI-compatible
`/v1/completions` endpoint, so the results include no network between
hosts. The prompts are
synthetic text of the workload's token counts, generated with the fixed seed
42.

A GuideLLM sweep of size N runs N load stages, each recorded as a `strategy`
in the output:

1. `synchronous`: one request at a time, for the baseline latency.
2. `throughput`: as many concurrent requests as the maximum concurrency
   allows, for the peak throughput.
3. `constant`: N - 2 stages at constant request rates between the two.

For example, a sweep of size 6 on an AWS (Amazon Web Services) `g5.16xlarge`
ran one synchronous, one throughput, and four constant-rate stages for the
`chat` workload
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/stdout)).
Each stage stops at its time limit, and the throughput stage ramps up its
concurrency first. The GuideLLM command and its options are in the
[harness](../../../vllm-common/benchmark.py).

## Load Settings

By default (`BENCHMARK_VLLM_AUTOCONFIG=1`), this benchmark derives the GuideLLM
load from the host's vCPU count `v`, not from the GPU count, the GPU model, or
GPU memory. Set `BENCHMARK_VLLM_VCPUS` to use another count. On GPUs, it uses
the following formulas:

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

This benchmark also computes vLLM server settings: the batch limits
`max_num_seqs` and `max_num_batched_tokens`, the data type `dtype`, and the
memory fraction `kv_memory_util`. It records them in each line's `tuning`
object. On GPUs, it does not pass them to `vllm serve`,
which uses its own defaults. Set `BENCHMARK_VLLM_MAX_NUM_SEQS` or
`BENCHMARK_VLLM_MAX_NUM_BATCHED_TOKENS` to pass `--max-num-seqs` or
`--max-num-batched-tokens`. See
[Environment Variables](./usage.md#environment-variables) to override the
other settings.

## Time Budget

By default, a run has an overall budget of 2 hours, multiplied by
`--benchmark-timeout-scale`; no new model or workload starts after 2 hours,
whatever the scale (see [Run Duration](./usage.md#run-duration)). This
benchmark splits the budget as follows:

1. It reserves 600 seconds, plus 60 seconds for each GPU server start, one per
   runnable model and workload.
2. It divides the rest by the number of model and workload runs.
3. It limits each run of one model and workload, that is, one sweep, to 45 to
   240 seconds.

For example, on the `g5.16xlarge` with one NVIDIA A10G GPU, the log shows
`overall=7200s reserve=1320s runs=12 per_run=240s`
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/g5.16xlarge/0/vllm/stderr)).

## GPU Parallelism

This benchmark does not always use every GPU of the server. It splits each
model across the GPUs as follows:

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
larger models on two
([output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/upcloud/GPU-32xCPU-384GB-3xL40S/0/vllm/stdout)).
The `llama-70b` model ran with `--pipeline-parallel-size 3`
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/upcloud/GPU-32xCPU-384GB-3xL40S/0/vllm/stderr)).

For `llama-70b` on more than one GPU, read `pipeline_parallel` in the output:
`tensor_parallel` still shows the size that the head-count rule gives, and the
log line of the parallel settings shows it too, although the server runs
without tensor parallelism.

## GPU Memory Fit

Before it starts any server, this benchmark estimates how much memory each
model needs on each GPU, without downloading the weights:

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
Run the image with `--plan-only` to see the plan of each model on your server.

## Legacy Settings

With `BENCHMARK_VLLM_AUTOCONFIG=0`, this benchmark uses static settings
instead of the formulas above. On GPUs, they are the following:

- A sweep size of 3, with a maximum concurrency of 512 and 10 GuideLLM workers
- A cap of 120 requests per run, and no throughput-stage ramp-up
- `40 + 8 * <parameters in billions>` seconds per stage

Set `GUIDELLM_PROFILES` to `legacy`, `sync-throughput`, or `sync` to run one
synchronous and one throughput run instead of a sweep, with the throughput run
at a concurrency of `GUIDELLM_THROUGHPUT_RATE` (default: 16).
