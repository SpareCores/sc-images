# Contributor Guide

## Image Layout

The [`Dockerfile`](../Dockerfile) builds the image from
`vllm/vllm-openai:v<VLLM_VERSION>` on Docker Hub. It installs `psutil` and
`guidellm[recommended]` at the pinned GuideLLM version with pip, and copies the
following files:

- the `resource-tracker` binary, from the image set by `RESOURCE_TRACKER_IMAGE`
- [`benchmark.py`](../../../vllm-common/benchmark.py),
  [`VLLM_VERSION`](../../../vllm-common/VLLM_VERSION), and
  [`GUIDELLM_VERSION`](../../../vllm-common/GUIDELLM_VERSION) from
  `vllm-common/`, to `/`

The [harness](../../../vllm-common/benchmark.py) and the version files are
shared with the CPU images,
[`benchmark-vllm-cpu`](../../benchmark-vllm-cpu/README.md) and
[`benchmark-vllm-cpu-avx2`](../../benchmark-vllm-cpu-avx2/README.md), so
the build context is the root of the
[`sc-images` repository](https://github.com/SpareCores/sc-images), as the
[`CONTEXT`](../CONTEXT) file sets. The image sets `BENCHMARK_VLLM_MODE=gpu`,
`TRACKER_QUIET=true`, and the Hugging Face cache to `/models`, a volume. It
also sets `USER=benchmark`, `HOME=/tmp`, and
`TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor` as environment variables, for
runs with `--user`; the `Dockerfile` has no `USER` instruction. The
[container entrypoint](../Dockerfile#L22) is
`resource-tracker -- python3 /benchmark.py`.

## Build Arguments

| Argument | Meaning | Default |
| --- | --- | --- |
| `VLLM_VERSION` | Tag of the `vllm/vllm-openai` base image, without the `v` prefix | `0.27.1` |
| `GUIDELLM_VERSION` | GuideLLM version installed with pip | `0.7.3` |
| `RESOURCE_TRACKER_IMAGE` | Image to copy the `resource-tracker` binary from. CI passes the image for the target architecture, `ghcr.io/sparecores/resource-tracker:main-<arch>`. | `ghcr.io/sparecores/resource-tracker:main` |

CI reads `VLLM_VERSION` and `GUIDELLM_VERSION` from the files in
`vllm-common/`, and builds the image for `amd64` and `arm64`, after the
`resource-tracker` image it depends on. The `Dockerfile` defaults match the
files.

To build the image locally, run the following command from the root of the
`sc-images` repository:

```bash
docker build -f images/benchmark-vllm-gpu/Dockerfile -t benchmark-vllm-gpu .
```

To check the rebuilt image, print the pinned versions:

```bash
docker run --rm benchmark-vllm-gpu --version
```

## Version Changes

The version files in `vllm-common/`, `VLLM_VERSION` and `GUIDELLM_VERSION`,
pin the vLLM and GuideLLM versions for all vLLM images.
Before changing `VLLM_VERSION`, confirm on
[Docker Hub](https://hub.docker.com/r/vllm/vllm-openai/tags) that the
`vllm/vllm-openai` tag exists for both `linux/amd64` and `linux/arm64`, because
CI builds this image for both.

Changing either version makes new results incomparable with the published
ones, which would need a rerun on every server type. We do not plan to update
them.

## Tests

The unit tests of [`benchmark.py`](../../../vllm-common/benchmark.py) are in
[`test_benchmark.py`](../../../vllm-common/test_benchmark.py) in
`vllm-common/`.
They include GPU cases for the data-parallel override, the memory fit check of
each GPU, and the compute capability check.
