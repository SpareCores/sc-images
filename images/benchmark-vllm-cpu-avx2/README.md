# benchmark-vllm-cpu-avx2

This benchmark measures how fast AMD64 CPUs with AVX2 (Advanced Vector
Extensions 2) but without AVX-512 (Advanced Vector Extensions 512) serve LLMs
to many concurrent requests. It runs the vLLM benchmarks on a vLLM build for
these CPUs: it starts each default model with `vllm serve` on the CPU and
sends synthetic requests to it with GuideLLM, from one request at a time up to
peak load. The [vLLM benchmarks manual](../../vllm-common/README.md) describes
the methodology, settings, and output that all vLLM benchmarks share, and this
README covers only what this image does differently.

Like the other vLLM benchmarks, this benchmark reports the latency and
throughput [metrics](../../vllm-common/README.md) of each model under load;
see the manual's [Results](../../vllm-common/README.md#results). Each line
also marks the AVX2 image and the data type that the model ran in; see
[Results](#results).

## Purpose

The [`benchmark-vllm-cpu`](../benchmark-vllm-cpu/README.md) image is built on
the [official vLLM CPU server image](https://hub.docker.com/r/vllm/vllm-openai-cpu)
from Docker Hub, whose AMD64 build needs AVX-512. On an AMD64 CPU without
AVX-512, `benchmark-vllm-cpu` exits with code 2 and names this image in its
error message. This image runs the same measurements on vLLM built from source
for AVX2, so that server types with these CPUs can get vLLM results too. See
the manual's [Purpose](../../vllm-common/README.md#purpose) for how the vLLM
benchmarks relate to the LLM inference speed that Spare Cores Navigator
publishes.

## Limitations

The results of this image share the
[limitations](../../vllm-common/README.md#limitations) of the vLLM benchmarks
on CPUs. Note the following limitations of this image in addition:

- **[No CPU Feature Check](#requirements-and-versions):** This image skips the
  harness's AVX-512 check and does not check for AVX2 either. Production tries
  it on every AMD64 server type where the two vLLM images before it in the
  probe order fail their probe, without checking the CPU flags first. See also
  [Production Setup](#production-setup).
- **[Data Type and Memory](#data-type):** On CPUs without AVX-512, the models
  run in float32 by default. The memory check counts the weights of a
  bfloat16 model at twice their size, because they run in float32, so a host
  can fit fewer models than a host with the same memory and bfloat16 support.
- **[Separate vLLM Build](#requirements-and-versions):** This image runs vLLM
  built from source for AVX2, not the official Docker Hub image that
  `benchmark-vllm-cpu` runs. Both images use the same vLLM version.

## Usage

Run the image via Docker with the container settings of production, with
`<limit>` set as the manual's
[Production Setup](../../vllm-common/docs/usage.md#production-setup)
describes:

```bash
docker run --rm --privileged --network host --shm-size=4g --memory=<limit> --memory-swap=<limit> --memory-swappiness=0 -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main > results.jsonl 2> results.log
```

A minimal run needs only `--shm-size=4g` and `-e HF_TOKEN`; the other
options of the first command only reproduce production:

```bash
docker run --rm --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main > results.jsonl 2> results.log
```

See the manual's [Usage](../../vllm-common/README.md#usage) for the Hugging
Face token, the model cache, and the 2-hour time budget.

### Requirements and Versions

This image is built on
[`vllm-cpu-base-avx2`](../vllm-cpu-base-avx2/README.md), which builds vLLM
from source with the `VLLM_CPU_AVX2=true` build option. This image and its
base are published for AMD64 only. This image and `benchmark-vllm-cpu` take
the vLLM version from the same pin, 0.27.1.

The [`Dockerfile`](./Dockerfile) sets `BENCHMARK_VLLM_ALLOW_AVX2_ONLY=true`,
which skips the harness's AVX-512 check on AMD64.

This image does not check for AVX2 itself. To check a host before a
run, look for the `avx2` flag in `/proc/cpuinfo`; the following command prints
`avx2` only when the CPU has it:

```bash
grep -o -m 1 avx2 /proc/cpuinfo
```

At startup, the log in stderr lists the CPU flags that the harness found, such
as `cpu_flag_avx2=yes`, and `cpu_bf16=False` on a CPU without bfloat16 support.
On a CPU without AVX-512, the log has no `cpu_flag_avx512f` line.

### Settings

This image takes the
[options and environment variables](../../vllm-common/docs/usage.md#options)
of the vLLM benchmarks. The only default that differs from
`benchmark-vllm-cpu` is `BENCHMARK_VLLM_ALLOW_AVX2_ONLY`, which this image sets
to `true`.

### Results

The JSON lines in stdout have the fields that the manual's
[Results](../../vllm-common/README.md#results) and
[Output Fields](../../vllm-common/docs/usage.md#output-fields) describe for
CPU runs. The following fields read differently in this image:

- `avx2_only_image` is `true`, because this image sets
  `BENCHMARK_VLLM_ALLOW_AVX2_ONLY`.
- `avx512` is `false` on CPUs without AVX-512.
- `mode` is `cpu`, and `arch` is `amd64`.
- `tuning` holds the data type in `dtype`, which is `float32` by default on
  CPUs without AVX-512.

The process exits with the codes of the manual's
[Exit Codes](../../vllm-common/docs/usage.md#exit-codes). With its default
settings, this image does not exit with code 2, the code of
`benchmark-vllm-cpu` on CPUs without AVX-512, because it skips that check. Any
`BENCHMARK_VLLM_ALLOW_AVX2_ONLY` value other than `1`, `true`, or `yes`, such
as `-e BENCHMARK_VLLM_ALLOW_AVX2_ONLY=0`, turns the check back on.

### Production Setup

Production tries this image last of the three vLLM images, without checking
the CPU flags of an AMD64 server type first; see the manual's
[Production Setup](../../vllm-common/docs/usage.md#production-setup). To test
this image on a server type before a full run, run its
[`--probe-only`](../../vllm-common/docs/usage.md#plan-and-probe-modes) check:

```bash
docker run --rm --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main --probe-only
```

## Workloads

This benchmark runs the `chat` and `rag` workloads of the vLLM benchmarks on
the five default models that run on CPUs, unchanged; see the manual's
[Workloads](../../vllm-common/README.md#workloads).

### Data Type

An AMD64 CPU without AVX-512 has no `avx512_bf16` flag, so on such a CPU, this
image runs the models in float32 by default, and the
[memory check](../../vllm-common/docs/workloads.md#cpu-memory-fit) doubles the
size of a bfloat16 model's weights; see
[Data Type](../../vllm-common/docs/workloads.md#data-type). Setting
`VLLM_CPU_DTYPE`, such as `-e VLLM_CPU_DTYPE=bfloat16`, overrides the data
type, and the memory check then uses the data type set.

To see which models fit on a host before a full run, start this image with
the [`--plan-only`](../../vllm-common/docs/usage.md#plan-and-probe-modes)
option. It prints the host's resources and whether each model fits in memory,
as JSON, without starting vLLM:

```bash
docker run --rm -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main --plan-only
```

## References

Common terms, such as AVX2, AVX-512, and float32, are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs. To
change or rebuild this image, see the
[contributor guide](../../vllm-common/docs/contributors.md) of the vLLM
benchmarks.
