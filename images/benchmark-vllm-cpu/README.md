# benchmark-vllm-cpu

This benchmark measures how fast AMD64 CPUs with AVX-512 (Advanced Vector
Extensions 512) and Arm64 CPUs serve LLMs to many concurrent requests. It is
the CPU image of the vLLM benchmarks: it starts each default model with vLLM
(`vllm serve`) on the CPU and sends synthetic requests to it with GuideLLM,
from one request at a time up to peak load.

The [vLLM benchmarks manual](../../vllm-common/README.md) describes the
methodology, settings, and output that all vLLM benchmarks share, and this
page covers only what this image does differently.

Like the other vLLM benchmarks, this benchmark reports the latency and
throughput [metrics](../../vllm-common/README.md) of each model under load;
see the manual's [Results](../../vllm-common/README.md#results).

## Purpose

This image runs the vLLM benchmarks on CPUs, with the
[official vLLM CPU server images](https://hub.docker.com/r/vllm/vllm-openai-cpu)
from Docker Hub. Production tries it on every server type where
[`benchmark-vllm-gpu`](../benchmark-vllm-gpu/README.md) does not run.

On AMD64 CPUs without AVX-512, which the official image does not support,
[`benchmark-vllm-cpu-avx2`](../benchmark-vllm-cpu-avx2/README.md) runs the
same benchmark instead. See the manual's
[Purpose](../../vllm-common/README.md#purpose) for how the vLLM benchmarks
relate to the LLM inference speed that Spare Cores Navigator publishes.

## Limitations

The results of this image share the
[limitations](../../vllm-common/README.md#limitations) of the vLLM benchmarks
on CPUs. Note the following limitation of this image in addition:

- **[AVX-512 on AMD64](#requirements-and-versions):** On AMD64 CPUs, this
  benchmark needs AVX-512. Without it, it exits with code 2, and its error
  message points to `benchmark-vllm-cpu-avx2`. Arm64 CPUs skip this check.

## Usage

Run the image via Docker with the container settings of production, with
`<limit>` set as the manual's
[Production Setup](../../vllm-common/docs/usage.md#production-setup)
describes:

```bash
docker run --rm --privileged --network host --shm-size=4g --memory=<limit> --memory-swap=<limit> --memory-swappiness=0 -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu:main > results.jsonl 2> results.log
```

A minimal run needs only `--shm-size=4g` and `-e HF_TOKEN`; the other
options of the first command only reproduce production:

```bash
docker run --rm --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-cpu:main > results.jsonl 2> results.log
```

See the manual's [Usage](../../vllm-common/README.md#usage) for the Hugging
Face token, the model cache, the 2-hour time budget, and the
[settings](../../vllm-common/docs/usage.md#options) that this image takes.

### Requirements and Versions

The image is built on the official vLLM CPU server images from Docker Hub,
`vllm/vllm-openai-cpu:v0.27.1-x86_64` for AMD64 and
`vllm/vllm-openai-cpu:v0.27.1-arm64` for Arm64, and its `:main` tag has
variants for both.

At startup on AMD64, this benchmark reads the CPU flags from `/proc/cpuinfo`.
The vLLM CPU image for AMD64 needs AVX-512, so without the `avx512f` flag,
this benchmark logs an error that names
`ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main` and exits with code 2.
Setting `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` to `1`, `true`, or `yes` skips this
check; `benchmark-vllm-cpu-avx2` sets it, and this image does not.

Arm64 CPUs skip the check, and this benchmark does not check the Arm64 CPU
generation either. On Arm64 CPUs without the `bf16` flag, it runs the models
in float32; see [Data Type](../../vllm-common/docs/workloads.md#data-type).

### Results

The JSON lines in stdout have the fields that the manual's
[Results](../../vllm-common/README.md#results) and
[Output Fields](../../vllm-common/docs/usage.md#output-fields) describe for
CPU runs. In this image, `arch` is `amd64` or `arm64`, `avx512` is `true` on
AMD64 and `null` on Arm64, and `avx2_only_image` is `false`.

The process exits with the codes of the manual's
[Exit Codes](../../vllm-common/docs/usage.md#exit-codes), and with code 2 on
an AMD64 CPU without AVX-512.

## Workloads

This benchmark runs the `chat` and `rag` workloads of the vLLM benchmarks on
the five default models that run on CPUs; see the manual's
[Workloads](../../vllm-common/README.md#workloads).

## References

Common terms, such as GuideLLM, NUMA (Non-Uniform Memory Access), and
AVX-512, are defined in the [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs. To change or rebuild this image, see the
[contributor guide](../../vllm-common/docs/contributors.md) of the vLLM
benchmarks.
