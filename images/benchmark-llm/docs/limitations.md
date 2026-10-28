# Limitations

## Disclaimer: What This Benchmark Does *Not* Deliver

### Concurrent Serving

This benchmark runs one `llama-bench` process at a time, and the
[harness](../benchmark.py) sends no concurrent requests. The results show the
speed a single user gets from a model, *not* the throughput of a server that
answers many requests at once, such as a shared LLM service.

For a single user, llama.cpp is often the fastest self-hosted option, but it
does not scale as well to many concurrent requests, where vLLM is preferred.
The vLLM benchmarks measure that case;
[Spare Cores Navigator](https://docs.sparecores.com/overview) does not show
their results yet.

### Output Quality

This benchmark measures speed in tokens per second. It does *not* check the
quality or accuracy of the model output.

## Scope Decisions

### Fixed Models and Build

Every run uses the same six models, all 4-bit quantized, and the same llama.cpp
build, `b4753` (commit `51f311e0`). The models are a diverse set of widely used
LLMs, from 135 million to 70 billion parameters, and their quantized versions
need much less memory and load faster. See [Models](./workloads.md#models) for
the list.

Other models, quantization levels, and llama.cpp builds are not covered. We do
not plan to update the build or the models, because comparable results would
need a rerun on every server type already measured. A newer llama.cpp build may
be faster in general; the published results show the differences between server
types at build `b4753`.

The `--model-urls` option replaces the model list for your own runs. The
results of such a run are not comparable with the Navigator's.

### CPU or GPU Results

On AMD64, this benchmark first tries the CUDA (Compute Unified Device
Architecture) build of llama.cpp, and falls back to the CPU build if it cannot
start, for example on a non-NVIDIA GPU or with NVIDIA driver issues. Production
also retries a failed GPU run without GPUs. See
[GPU Run](./usage.md#gpu-run) and
[Production Setup](./usage.md#production-setup).

The Navigator publishes the score measured on a server type, without marking
whether it came from the GPU or the CPU. Each output line records it in
`backends` (`CUDA` or `CPU`), but this field is not published.

### GPU Coverage

Only NVIDIA GPUs are used, through CUDA. The Arm64 image has no CUDA build, so
it always runs on the CPU.

When a model does not fit in GPU memory, this benchmark offloads fewer of its
layers to the GPUs, and the remaining layers stay on the CPU. Such a score
depends on both, and `n_gpu_layers` in the output shows how many layers were
offloaded.
See [GPU Layers](./workloads.md#gpu-layers).

## Design Constraints

### Missing Results

A slow server cannot spend hours on long prompts or large models, so every
measurement has a timeout, and the following rules skip the rest of a run:

- If the first, 16-token measurement of either task fails or times out, this
  benchmark stops and skips all larger models.
- If a later token count fails or times out, the remaining token counts of that
  task are skipped for that model.
- If a measurement of 1,024 or more tokens takes too long to meet the next
  token count's expected speed, the remaining token counts of that task are
  skipped.
- If a model would not finish downloading within `--download-timeout`, the
  run stops, so that model and all larger ones are skipped.
- If the disk has too little free space for a model, only that model is
  skipped, and the run continues with the next one. The next default models
  are larger, so they are skipped too.

All of these paths exit with code 0, so a run with missing results looks like a
successful run. See
[Timeouts and Early Stops](./workloads.md#timeouts-and-early-stops) for the
timeouts.

The Navigator shows "no data" for a missing score, because the cause is not
recorded: the server can be too slow, or the run can have failed or not tried.
The Navigator's LLM workload profile includes a penalty when the smaller models
have scores but the larger ones are missing, because then the larger models
could not load or finish in time. For example, the
[AWS (Amazon Web Services) `c7g.large` page](https://sparecores.com/server/aws/c7g.large)
has scores for the smaller models only.

<!-- embed -->
[Workload Profiles of c7g.large](https://sparecores.com/embed/server/aws/c7g.large/workload_profile)

### Downloads and Disk Space

The models are downloaded from Hugging Face during the run, not built into the
image, so download speed affects which models a run reaches. In production, an
AWS `c9g.48xlarge` downloaded at 25.7 MB/s and gave up the 70B model, whose
file is \~42 GB
([log](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c9g.48xlarge/0/llm/stderr)).
Its [LLM scores](https://sparecores.com/server/aws/c9g.48xlarge#llm_prompt_chart)
stop at the 14B model.

<!-- embed -->
[LLM Prompt Processing of c9g.48xlarge](https://sparecores.com/embed/server/aws/c9g.48xlarge/llm_prompt)

Together, the six default models take \~58 GB of disk space. This benchmark
checks the free space of the container's root filesystem, not of `--models-dir`;
see [Model Downloads and Disk Space](./usage.md#model-downloads-and-disk-space).
