# benchmark-vllm-gpu

This benchmark measures how fast NVIDIA GPUs serve LLMs to many concurrent
requests. It is the GPU image of the vLLM benchmarks: it starts each default
model with vLLM (`vllm serve`), from the smallest to the largest, and sends
synthetic requests to it with GuideLLM, from one request at a time up to peak
load.

The [vLLM benchmarks manual](../../vllm-common/README.md) describes the
methodology, settings, and output that all vLLM benchmarks share, and this
page covers only what this image does differently.

Like the other vLLM benchmarks, this benchmark reports the latency and
throughput [metrics](../../vllm-common/README.md) of each model under load;
see the manual's [Results](../../vllm-common/README.md#results).

## Purpose

This image runs the vLLM benchmarks on NVIDIA GPUs, with the
[official vLLM server image](https://hub.docker.com/r/vllm/vllm-openai) from
Docker Hub. Production tries it first on every server type, and runs the CPU
images only where it does not run. See the manual's
[Purpose](../../vllm-common/README.md#purpose) for how the vLLM benchmarks
relate to the LLM inference speed that Spare Cores Navigator publishes.

## Limitations

The results of this image share the
[limitations](../../vllm-common/README.md#limitations) of the vLLM benchmarks
on GPUs. Note the following limitation of this image in addition:

- **[NVIDIA GPUs Only](#gpu-run):** This benchmark needs NVIDIA GPUs with CUDA
  (Compute Unified Device Architecture) compute capability 8.0 or newer, the
  Ampere generation or later. If no GPU is visible or every visible GPU is
  older, it exits with code 1.

## Usage

Run the image via Docker with the GPUs passed and 4 GiB of shared memory:

```bash
docker run --rm --gpus all --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-gpu:main > results.jsonl
```

To reproduce the container settings of production, except the memory limit,
run the following command:

```bash
docker run --rm --privileged --network host --gpus all --shm-size=4g -e HF_TOKEN ghcr.io/sparecores/benchmark-vllm-gpu:main > results.jsonl
```

See the manual's [Usage](../../vllm-common/README.md#usage) for the Hugging
Face token, the model cache, the 2-hour time budget, and the
[settings](../../vllm-common/docs/usage.md#options) that this image takes.

### GPU Run

The host needs the NVIDIA driver and the NVIDIA Container Toolkit for
`--gpus`. The image is built on `vllm/vllm-openai:v0.27.1` from Docker Hub,
and its `:main` tag has variants for AMD64 and Arm64.

At startup, this benchmark reads the CUDA compute capability of each visible
GPU. It needs compute capability 8.0 or newer, the Ampere generation or
later, such as the NVIDIA A10G or L40S, but not the older T4G. If every
visible GPU is below 8.0, it logs an error and exits with code 1. A GPU whose
compute capability cannot be read is allowed.

If no GPU is visible, for example without `--gpus`, the check passes, but no
model fits the
[GPU memory check](../../vllm-common/docs/workloads.md#gpu-memory-fit). This
benchmark then logs each model as skipped for insufficient memory and exits
with code 1.

### Results

The JSON lines in stdout have the fields that the manual's
[Results](../../vllm-common/README.md#results) and
[Output Fields](../../vllm-common/docs/usage.md#output-fields) describe for
GPU runs.

## Workloads

This benchmark runs all three workloads of the vLLM benchmarks, `chat`, `rag`,
and `long`, on the six default models; see the manual's
[Workloads](../../vllm-common/README.md#workloads). The
[Load Settings](../../vllm-common/docs/workloads.md#load-settings) give the
settings of production runs on GPU server types.

## References

The terms of the vLLM benchmarks, such as `--max-model-len`, are defined in
their [references](../../vllm-common/docs/references.md).
Common terms, such as GuideLLM, CUDA, and tensor parallelism, are defined in
the [glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs.
To change or rebuild this image, see the
[contributor guide](../../vllm-common/docs/contributors.md) of the vLLM
benchmarks.
