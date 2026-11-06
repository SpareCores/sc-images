# Contributor Guide

## Image Layout

The three vLLM images differ only in their `Dockerfile` and base image. Each
`Dockerfile` installs `psutil`, and `guidellm[recommended]` at the pinned
GuideLLM version, with pip, and copies the following files:

- the `resource-tracker` binary, from the image set by `RESOURCE_TRACKER_IMAGE`
- [`benchmark.py`](../benchmark.py), [`VLLM_VERSION`](../VLLM_VERSION), and
  [`GUIDELLM_VERSION`](../GUIDELLM_VERSION) from `vllm-common/`, to `/`

The images share the [harness](../benchmark.py) and the version files, so
their build context is the root of the
[`sc-images` repository](https://github.com/SpareCores/sc-images), as each
image's `CONTEXT` file sets. The container entrypoint of each image is
`resource-tracker -- python3 /benchmark.py`.

| Image | Base image | Architectures |
| --- | --- | --- |
| [`benchmark-vllm-gpu`](../../images/benchmark-vllm-gpu/Dockerfile) | `vllm/vllm-openai:v<VLLM_VERSION>` from Docker Hub | `amd64`, `arm64` |
| [`benchmark-vllm-cpu`](../../images/benchmark-vllm-cpu/Dockerfile) | `vllm/vllm-openai-cpu:v<VLLM_VERSION>-x86_64` for `amd64` and `vllm/vllm-openai-cpu:v<VLLM_VERSION>-arm64` for `arm64` from Docker Hub, selected by the `TARGETARCH` build argument | `amd64`, `arm64` |
| [`benchmark-vllm-cpu-avx2`](../../images/benchmark-vllm-cpu-avx2/Dockerfile) | `ghcr.io/sparecores/vllm-cpu-base-avx2:main-amd64`, vLLM built from source for AVX2 by [`vllm-cpu-base-avx2`](../../images/vllm-cpu-base-avx2/README.md) | `amd64` |

The images set the following environment variables. `/models` is a volume, and
`USER`, `HOME`, and `TORCHINDUCTOR_CACHE_DIR` are for runs as a non-root user
with `--user`; no `Dockerfile` has a `USER` instruction:

| Variable | `benchmark-vllm-gpu` | `benchmark-vllm-cpu` | `benchmark-vllm-cpu-avx2` |
| --- | --- | --- | --- |
| `BENCHMARK_VLLM_MODE` | `gpu` | `cpu` | `cpu` |
| `BENCHMARK_VLLM_ALLOW_AVX2_ONLY` | Unset | Unset | `true` |
| `VLLM_CPU_OMP_THREADS_BIND` | Unset | `auto` | `auto` |
| `TRACKER_QUIET` | `true` | `true` | `true` |
| `HF_HOME`, `HUGGINGFACE_HUB_CACHE` | `/models` | `/models` | `/models` |
| `USER` | `benchmark` | `benchmark` | `benchmark` |
| `HOME` | `/tmp` | `/tmp` | `/tmp` |
| `TORCHINDUCTOR_CACHE_DIR` | `/tmp/torch_inductor` | `/tmp/torch_inductor` | `/tmp/torch_inductor` |

## Build Arguments

| Argument | Meaning | Default |
| --- | --- | --- |
| `VLLM_VERSION` | `benchmark-vllm-gpu` and `benchmark-vllm-cpu`: tag of the base image, without the `v` prefix and the architecture suffix | `0.27.1` |
| `VLLM_CPU_BASE_AMD64` | `benchmark-vllm-cpu-avx2`: the AVX2 base image | `ghcr.io/sparecores/vllm-cpu-base-avx2:main-amd64` |
| `GUIDELLM_VERSION` | GuideLLM version installed with pip | `0.7.3` |
| `RESOURCE_TRACKER_IMAGE` | Image to copy the `resource-tracker` binary from. CI passes the image for the target architecture, `ghcr.io/sparecores/resource-tracker:main-<arch>`. | `ghcr.io/sparecores/resource-tracker:main` |

CI reads `VLLM_VERSION` and `GUIDELLM_VERSION` from the files in
`vllm-common/`, and builds each image for its architectures, after the
`resource-tracker` image it depends on. The `Dockerfile` defaults match the
files. The `benchmark-vllm-cpu-avx2` image takes its vLLM version from its
base image, which `vllm-cpu-base-avx2` builds from the same
`VLLM_VERSION` file.

To build an image locally, run the following command from the root of the
`sc-images` repository, with the image's folder name, such as
`benchmark-vllm-cpu`:

```bash
docker build -f images/<image>/Dockerfile -t <image> .
```

To check the rebuilt image, print the pinned versions:

```bash
docker run --rm <image> --version
```

## Version Changes

The version files in `vllm-common/`, `VLLM_VERSION` and `GUIDELLM_VERSION`,
pin the vLLM and GuideLLM versions for all vLLM images. Before changing
`VLLM_VERSION`, confirm on Docker Hub that the new tags exist for both
architectures, because CI builds the images for both:

- [`vllm/vllm-openai`](https://hub.docker.com/r/vllm/vllm-openai/tags), for
  both `linux/amd64` and `linux/arm64`
- [`vllm/vllm-openai-cpu`](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags),
  both the `-x86_64` and the `-arm64` tag

Changing either version makes new results incomparable with earlier ones, so
every server type would need a rerun. We do not plan to change the pinned
versions.

## Tests

The unit tests of [`benchmark.py`](../benchmark.py) are in
[`test_benchmark.py`](../test_benchmark.py) in `vllm-common/`. They include
GPU cases for the data-parallel override, the memory fit check of each GPU,
and the compute capability check.
