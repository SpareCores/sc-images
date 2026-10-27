# benchmark-llm

This benchmark measures how fast a server runs LLM (Large Language Model)
inference for one request stream at a time, with `llama-bench` from llama.cpp.
It downloads six 4-bit quantized models, from 135 million to 70 billion
parameters, and benchmarks them from smallest to largest. It runs on NVIDIA
GPUs when the CUDA (Compute Unified Device Architecture) build of llama.cpp
can start, and on the CPU otherwise.

This benchmark reports speed in tokens per second for two tasks, prompt
processing and text generation, each at several token counts. It prints one
JSON line per model, task, and token count, with the average and the values of
five repetitions.

## Purpose

[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes
empirical performance measurements for more than 5,000 cloud server types,
including LLM inference speed from this benchmark: one score in tokens per
second for each server type, model, task, and token count. In the
[Navigator's Web UI](https://sparecores.com/servers), you can order the server
list by one of these scores, and each server's details page charts all of
them.

We needed one method that runs on every server type, from servers with less
than 1 GB of RAM to servers with terabytes of memory, hundreds of CPU cores,
and several GPUs. llama.cpp runs inference on both CPUs and GPUs with minimal
tuning, and its project publishes container images for AMD64 and Arm64, a CUDA
build, and a configurable benchmark tool, `llama-bench`.

llama.cpp is often the fastest self-hosted option for a single user, so this
benchmark measures one request stream at a time. For shared serving with many
concurrent requests, vLLM is preferred; the
[vLLM benchmarks](../../vllm-common/README.md) cover that case, but the
Navigator does not publish their results yet.

## Limitations

These results compare single-stream LLM inference speed across server types.
Note the following limitations:

- **[Concurrent Serving](./docs/limitations.md#concurrent-serving):** This
  benchmark runs one `llama-bench` process at a time and sends no concurrent
  requests, so it does not measure serving throughput under load.
- **[Output Quality](./docs/limitations.md#output-quality):** This benchmark
  measures speed in tokens per second, not the quality or accuracy of the
  model output.
- **[Fixed Models and Build](./docs/limitations.md#fixed-models-and-build):**
  Every run uses the same six models, 4-bit quantization, and llama.cpp build
  `b4753`. Other models, quantization levels, and builds are not covered.
- **[CPU or GPU Results](./docs/limitations.md#cpu-or-gpu-results):**
  On a GPU server, a score can come from the GPU or from the CPU fallback, and
  the Navigator does not mark which. Only NVIDIA GPUs are used, and only on AMD64. See also
  [GPU Coverage](./docs/limitations.md#gpu-coverage).
- **[Missing Results](./docs/limitations.md#missing-results):** On slow
  servers, timeouts skip larger token counts and models, and the run still
  exits with code 0. The Navigator shows "no data" for the skipped results.
- **[Downloads and Disk Space](./docs/limitations.md#downloads-and-disk-space):**
  The models are downloaded during the run, so a slow download can skip the
  largest models, even on a fast server. A model is also skipped if the disk has
  too little free space for it.

## Usage

Run the image via Docker. Without GPU access, it runs the CPU build of
llama.cpp:

```bash
docker run --rm ghcr.io/sparecores/benchmark-llm:main > results.jsonl
```

The image downloads the models from Hugging Face while it runs, so it needs
network access and up to \~58 GB of free disk space for all six models. It
caches them in `/models`, a Docker volume. See
[Model Downloads and Disk Space](./docs/usage.md#model-downloads-and-disk-space)
to reuse downloaded models.

To run on NVIDIA GPUs, the host needs the NVIDIA driver and the
NVIDIA Container Toolkit. Pass the GPUs to the container. Only the AMD64 variant of `:main` includes
the CUDA build:

```bash
docker run --rm --gpus all ghcr.io/sparecores/benchmark-llm:main > results.jsonl
```

If the CUDA build cannot start inside the container, for example on a
non-NVIDIA GPU or with NVIDIA driver issues, this benchmark falls back to the
CPU build. The fallback does not replace the host requirements above. See
[GPU Run](./docs/usage.md#gpu-run) for how it picks the build and the GPU
layers.

Production runs the container privileged and with host networking, which a
plain `docker run` does not, and passes the GPUs on GPU servers. See
[Production Setup](./docs/usage.md#production-setup) before you compare your
results with the Navigator's.

Sample production runs took about 18 to 32 minutes. Run time depends on the
server's speed, on the download speed, and on how many models and token counts
fit in their timeouts. See [Run Duration](./docs/usage.md#run-duration) for
the timeouts.

### Settings

Pass options after the image name. The following options are the most common;
see the [full option reference](./docs/usage.md#options) for all options and
defaults:

- `--model-urls` replaces the default model list with URLs of GGUF model
  files, separated by spaces and ordered from smallest to largest.
- `--models-dir` sets the download and cache directory (default `/models`).
- `--benchmark-timeout-scale` multiplies every per-scenario timeout by an
  integer (default 1).
- `--download-timeout` sets the maximum wait for each model download
  (default 900s).
- `--version` prints the commit hash of llama.cpp build `b4753`, `51f311e0`,
  and exits.

### Results

The process prints JSON Lines to stdout: one line per model, task, and token
count, written as each measurement finishes. Logs go to stderr. The main fields
of each line are the following:

- `avg_ts` is the average speed in tokens per second, and `stddev_ts` its
  standard deviation. `samples_ts` lists the five repetitions.
- `n_prompt` and `n_gen` identify the task: a prompt processing line has
  `n_gen: 0` and `n_prompt` tokens, and a text generation line has
  `n_prompt: 0` and `n_gen` tokens.
- `model_filename`, `model_size` (in bytes), and `model_n_params` identify the
  model.
- `backends` is `CUDA` or `CPU`, and `n_gpu_layers` is the number of model
  layers offloaded to the GPUs. `gpu_info`, `cpu_info`, and `n_threads`
  describe the hardware used.
- `avg_ns` and `samples_ns` give the same measurements as time per repetition,
  in nanoseconds.

The process exits with code 0 even when it skipped models or token counts, so
check which lines are present, and the reasons logged to stderr. On `SIGINT` or
`SIGTERM`, it exits with code 128 plus the signal number. Resource usage is
recorded with `resource-tracker`, which streams it to the Spare Cores Sentinel
platform for the team's own monitoring. It is not published.

## Workloads

Each model runs the following two tasks, in this order:

- [Prompt processing](./docs/workloads.md#prompt-processing): processes a
  prompt of 16 to 16,384 tokens without generating any (`-p <n> -n 0`).
- [Text generation](./docs/workloads.md#text-generation): generates 16 to
  4,096 tokens without a prompt (`-n <n> -p 0`).

Each token count runs as a separate `llama-bench` process with its own timeout.
See [Models](./docs/workloads.md#models) for the six default models, and
[Timeouts and Early Stops](./docs/workloads.md#timeouts-and-early-stops) for
how slow servers skip larger token counts and models.

## Design History

We picked a diverse set of widely used LLMs, from 135 million to 70 billion
parameters, and use their 4-bit quantized versions, which need much less memory
and load faster. We chose llama.cpp because it runs on both CPUs and GPUs,
from the smallest to the largest server types, and pinned build `b4753`, the
most recent at the time, which worked in our initial tests. We do not plan to
update it: comparable results would need a rerun on every server type already
measured, potentially costing tens of thousands of US dollars. We would
reconsider if a newer build materially changed the performance difference
between server types, not only the overall speed.

Slow servers cannot spend hours on long prompts, so the expected speed rises
with the token count: 1 token per second can be acceptable for a short
response, but not for batch processing of long inputs. In our experiments on
several cloud server types, speed usually rose with the token count up to
1,024 tokens, so early stops start only from there.

In production, speed and download time limit how many models a server
completes. On an AWS (Amazon Web Services) `c7g.large`, the 7B model timed out
on its first scenario
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/llm/stderr)),
and on a `c9g.48xlarge`, this benchmark gave up the 70B model's download at
25.7 MB/s
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c9g.48xlarge/0/llm/stderr)).
See the [LLM inference speed](https://sparecores.com/article/llm-inference-speed)
article for the methodology and results in more detail.

## References

See the full list of [references](./docs/references.md) used in this
documentation. To change or rebuild this image, see the
[contributor guide](./docs/contributors.md).
