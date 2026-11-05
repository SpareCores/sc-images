# benchmark-vllm-cpu-avx2

This benchmark measures how fast AMD64 CPUs with AVX2 (Advanced Vector
Extensions 2) but without AVX-512 (Advanced Vector Extensions 512) serve LLMs
to many concurrent requests. It runs the harness and CPU methodology of
[`benchmark-vllm-cpu`](../benchmark-vllm-cpu/README.md) on a vLLM build for
these CPUs: it starts each default model with `vllm serve` on the CPU and
sends synthetic requests to it with GuideLLM, from one request at a time up to
peak load. The [vLLM manual](../../vllm-common/README.md) describes the harness
that all vLLM benchmarks share, and these docs cover only what this image does
differently.

Like `benchmark-vllm-cpu`, this image reports the following metrics, as one
JSON line per model, workload, load stage, metric, and statistic:

- **Latency:** TTFT (Time to First Token), TPOT (Time per Output Token), ITL
  (Inter-Token Latency), and end-to-end request latency
- **Throughput:** output tokens, total tokens, and requests per second

Each line also records that this image wrote it, and the data type that the
model ran in; see [Results](#results).

## Purpose

The official vLLM CPU server image for AMD64 on Docker Hub, which
`benchmark-vllm-cpu` is built on, needs AVX-512. On an AMD64 CPU without
AVX-512, `benchmark-vllm-cpu` exits with code 2 and names this image in its
error message. This image runs the same measurements on vLLM built from source
for AVX2, so that server types with these CPUs can get vLLM results too.

[Spare Cores Navigator](https://docs.sparecores.com/overview) does not show
the results of the vLLM benchmarks yet; see the
[Purpose](../benchmark-vllm-cpu/README.md#purpose) of `benchmark-vllm-cpu` for
how they relate to the LLM inference speed that the Navigator publishes.

## Limitations

Apart from its AVX-512 requirement, the results of this image share the
[limitations](../benchmark-vllm-cpu/README.md#limitations) of
`benchmark-vllm-cpu`, such as load settings that follow the vCPU count and
models that are skipped when they do not fit in memory. Note the following
limitations of this image in addition:

- **[No CPU Feature Check](#requirements-and-versions):** This image skips the
  harness's AVX-512 check and does not check for AVX2 either. Production tries
  it on every AMD64 server type where the earlier vLLM images fail, without
  checking the CPU flags first. See also [Production Setup](#production-setup).
- **[Data Type and Memory](#data-type):** On CPUs without AVX-512, the models
  run in float32 by default. The memory check counts the weights of a
  bfloat16 model at twice their size, because they run in float32, so a host
  can fit fewer models than a host with the same memory and bfloat16 support.
- **[Separate vLLM Build](#requirements-and-versions):** This image runs vLLM
  built from source for AVX2, not the official Docker Hub image that
  `benchmark-vllm-cpu` runs. Both images use the same vLLM version.

## Usage

Export your Hugging Face
[access token](https://huggingface.co/docs/hub/security-tokens) as `HF_TOKEN`
in the shell first; `-e HF_TOKEN` passes it into the container. Then run the
image via Docker with the container settings of production:

```bash
docker run --rm --privileged --network host --shm-size=4g --memory=<limit> --memory-swap=<limit> --memory-swappiness=0 -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main > results.jsonl 2> results.log
```

Replace `<limit>` in both options with 85% of the host's available memory, as
production does; the same value for `--memory-swap` disables swap. Production
reads the memory that Linux reports as available, the `available` column of
`free -g` in GiB. Write the limit with a unit: for example, with 40 GiB available, pass
`--memory=34g --memory-swap=34g`.

The vLLM benchmarks fail with Docker's default 64 MiB `/dev/shm`, so a minimal
run needs only `--shm-size=4g` and `-e HF_TOKEN`; the other options match
production:

```bash
docker run --rm --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main > results.jsonl 2> results.log
```

This image downloads the models from Hugging Face while it runs. Two
[default models](../benchmark-vllm-cpu/docs/workloads.md#models), `gemma-2b`
and `llama-8b`, are gated models: they need a token from an account that has
accepted their licenses, and without one, they are skipped. The models are
cached in `/models`, a Docker volume; to reuse them across runs, mount a host
directory on it, such as `-v "$PWD/models:/models"`. See
[Hugging Face Access and Model Cache](../benchmark-vllm-cpu/docs/usage.md#hugging-face-access-and-model-cache)
for details.

A run fits its models and workloads into a 2-hour time budget; see
[Time Budget](../benchmark-vllm-cpu/docs/workloads.md#time-budget) for how the
budget is split.

### Requirements and Versions

This image is built on
[`vllm-cpu-base-avx2`](../vllm-cpu-base-avx2/README.md), which builds vLLM
from source with the `VLLM_CPU_AVX2=true` build option. This image and its
base are published for AMD64 only. The `benchmark-vllm-cpu` image runs the
official vLLM CPU server image from Docker Hub instead, whose AMD64 build
needs AVX-512. This image and `benchmark-vllm-cpu` take the vLLM version from
the same pin, 0.27.1.

The [`Dockerfile`](./Dockerfile) sets `BENCHMARK_VLLM_MODE=cpu`, so the
[harness](../../vllm-common/benchmark.py) always runs in CPU mode, and
`BENCHMARK_VLLM_ALLOW_AVX2_ONLY=true`, which skips the harness's AVX-512 check
on AMD64. This image does not check for AVX2 itself. To check a host before a
run, look for the `avx2` flag in `/proc/cpuinfo`; the following command prints
`avx2` only when the CPU has it:

```bash
grep -o -m 1 avx2 /proc/cpuinfo
```

At startup, the log in stderr lists the CPU flags that the harness found, such as
`cpu_flag_avx2=yes`, and `cpu_bf16=False` on a CPU without bfloat16 support.

The `--version` option prints the pinned vLLM and GuideLLM versions and exits:

```bash
docker run --rm ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main --version
```

It prints `vllm=0.27.1 guidellm=0.7.3`, read from the version files built into
the image.

### Settings

This image takes the options and environment variables of
`benchmark-vllm-cpu`; see its
[full option and variable reference](../benchmark-vllm-cpu/docs/usage.md#options).
Its only different default is `BENCHMARK_VLLM_ALLOW_AVX2_ONLY`, which this
image sets to `true`. Set `VLLM_CPU_DTYPE` to run the models in a data type
other than float32; see [Data Type](#data-type).

### Results

The JSON lines in stdout have the fields of `benchmark-vllm-cpu`; see its
[Results](../benchmark-vllm-cpu/README.md#results) and
[Output Fields](../benchmark-vllm-cpu/docs/usage.md#output-fields). The
following fields read differently in this image:

- `avx2_only_image` is `true`, because this image sets
  `BENCHMARK_VLLM_ALLOW_AVX2_ONLY`.
- `avx512` is `false` on CPUs without AVX-512.
- `mode` is `cpu`, and `arch` is `amd64`.
- `tuning` holds the data type in `dtype`, `float32` on CPUs without
  AVX-512 by default.

The process exits with code 0 when it wrote at least one JSON line, even if it
skipped models or workloads. It exits with code 1 in the following cases:

- No model could run, for example because none fits in memory.
- No JSON line was written.
- `/models` has less than 1 GB of free space at startup.

With its default settings, this image does not exit with code 2, the code of
`benchmark-vllm-cpu` on CPUs without AVX-512, because it skips that check. Any
`BENCHMARK_VLLM_ALLOW_AVX2_ONLY` value other than `1`, `true`, or `yes`, such
as `-e BENCHMARK_VLLM_ALLOW_AVX2_ONLY=0`, turns the check back on. Check which lines are present in
stdout and which reasons are logged to stderr; the commands above save them to
`results.jsonl` and `results.log`. See
[Exit Codes](../benchmark-vllm-cpu/docs/usage.md#exit-codes) for the log
messages of skipped models.

### Production Setup

The Navigator's [Inspector](https://docs.sparecores.com/overview#how-it-works)
runs one task, `vllm`, for the three vLLM images, with the container settings
of the first command above and a task timeout of 3 hours. On each server type,
it tries the images in the following order, and the first image whose
[`--probe-only`](../benchmark-vllm-cpu/docs/usage.md#plan-and-probe-modes)
run exits with code 0 runs the full benchmark:

1. [`benchmark-vllm-gpu`](../benchmark-vllm-gpu/README.md)
2. `benchmark-vllm-cpu`
3. This image

The Inspector skips this image on Arm64 server types, but does not check the
CPU flags of an AMD64 server type before it tries this image. It stops each
probe after 18 minutes. A probe starts the smallest default model,
SmolLM2-135M, waits for its health check, and exits with code 0 or 1. To test
this image on a server type before a full run, probe it yourself:

```bash
docker run --rm --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main --probe-only
```

Production stores the results of all three images under the task name
`vllm`. The task tries the fastest image first, so the result of a server
type comes from the fastest image that runs on it. See
[Production Setup](../benchmark-vllm-cpu/docs/usage.md#production-setup) of
`benchmark-vllm-cpu` for how production differs from a plain `docker run`.

## Workloads

This image runs the workloads, models, and GuideLLM sweeps of
`benchmark-vllm-cpu` unchanged: each model runs the
[`chat` and `rag` workloads](../benchmark-vllm-cpu/docs/workloads.md#prompt-and-output-lengths).
See [Models](../benchmark-vllm-cpu/docs/workloads.md#models) for the five
default models that run on CPUs, and
[Run Sequence](../benchmark-vllm-cpu/docs/workloads.md#run-sequence) for when a
model is skipped or stops the run.

### Data Type

The harness runs the models in bfloat16 only when `/proc/cpuinfo` lists the
`avx512_bf16` or `bf16` flag, and in float32 otherwise. An AMD64 CPU without
AVX-512 has no `avx512_bf16` flag, so on such a CPU, this image runs the models
in float32 by default. The harness passes the data type to `vllm serve`.
Setting `VLLM_CPU_DTYPE`, such as `-e VLLM_CPU_DTYPE=bfloat16`, overrides the
harness's choice, and the memory check below then uses the data type set.

The float32 format takes 4 bytes per weight, twice as many as bfloat16. When
the serving data type is wider than the data type in the model's configuration,
the [memory check](../benchmark-vllm-cpu/docs/workloads.md#memory-fit) of the
[harness](../../vllm-common/benchmark.py) scales the size of the weights by the
same ratio, for example by 2 for a bfloat16 model. A host without bfloat16
support can therefore fit fewer models than a host with the same memory that
has it.

To see which models fit on a host before a full run, start this image with
the `--plan-only` option. It prints the host's resources and whether each
model fits in memory, as JSON, without starting vLLM:

```bash
docker run --rm -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main --plan-only
```

See [Plan and Probe Modes](../benchmark-vllm-cpu/docs/usage.md#plan-and-probe-modes)
for details.

## References

See the full list of [references](./docs/references.md) used in this
documentation. The [references](../benchmark-vllm-cpu/docs/references.md) of
`benchmark-vllm-cpu` explain the vLLM terms of its documentation.
