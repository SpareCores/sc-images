# Contributor Guide

## Image Layout

The [`Dockerfile`](../Dockerfile) builds the image from the vLLM CPU server
image on Docker Hub for the target architecture:
`vllm/vllm-openai-cpu:v<VLLM_VERSION>-x86_64` for `amd64` and
`vllm/vllm-openai-cpu:v<VLLM_VERSION>-arm64` for `arm64`. The `TARGETARCH`
build argument selects the tag. With pip, it installs `psutil`, and
`guidellm[recommended]` at the pinned GuideLLM version. It also copies the
following files:

- the `resource-tracker` binary, from the image set by `RESOURCE_TRACKER_IMAGE`
- [`benchmark.py`](../../../vllm-common/benchmark.py),
  [`VLLM_VERSION`](../../../vllm-common/VLLM_VERSION), and
  [`GUIDELLM_VERSION`](../../../vllm-common/GUIDELLM_VERSION) from
  `vllm-common/`, to `/`

The [harness](../../../vllm-common/benchmark.py) and the version files are
shared with [`benchmark-vllm-gpu`](../../benchmark-vllm-gpu/README.md) and
[`benchmark-vllm-cpu-avx2`](../../benchmark-vllm-cpu-avx2/README.md), so the
build context is the root of the
[`sc-images` repository](https://github.com/SpareCores/sc-images), as the
[`CONTEXT`](../CONTEXT) file sets. The image sets `BENCHMARK_VLLM_MODE=cpu`,
`VLLM_CPU_OMP_THREADS_BIND=auto`, `TRACKER_QUIET=true`, and the Hugging Face
cache to `/models`, a volume. It also sets `USER=benchmark`, `HOME=/tmp`, and
`TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor` as environment variables, for
runs as a non-root user. The [container entrypoint](../Dockerfile#L26) is
`resource-tracker -- python3 /benchmark.py`.

## Build Arguments

| Argument | Meaning | Default |
| --- | --- | --- |
| `VLLM_VERSION` | Tag of the `vllm/vllm-openai-cpu` base image, without the `v` prefix and the architecture suffix | `0.27.1` |
| `GUIDELLM_VERSION` | GuideLLM version installed with pip | `0.7.3` |
| `RESOURCE_TRACKER_IMAGE` | Image to copy the `resource-tracker` binary from. CI passes the image for the target architecture, `ghcr.io/sparecores/resource-tracker:main-<arch>`. | `ghcr.io/sparecores/resource-tracker:main` |

CI reads `VLLM_VERSION` and `GUIDELLM_VERSION` from the files in
`vllm-common/`, and builds the image for `amd64` and `arm64`, after the
`resource-tracker` image it depends on. The `Dockerfile` defaults match the
files.

To build the image locally, run the following command from the root of the
`sc-images` repository:

```bash
docker build -f images/benchmark-vllm-cpu/Dockerfile -t benchmark-vllm-cpu .
```

To check the rebuilt image, print the pinned versions:

```bash
docker run --rm benchmark-vllm-cpu --version
```

## Version Changes

The version files in `vllm-common/`, `VLLM_VERSION` and `GUIDELLM_VERSION`,
pin the vLLM and GuideLLM versions for all vLLM images. Before changing
`VLLM_VERSION`, confirm on
[Docker Hub](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags) that both the
`-x86_64` and the `-arm64` tag of the new version exist, because CI builds
this image for both architectures.

Changing either version makes new results incomparable with earlier ones, so
every server type would need a rerun. We do not plan to change the pinned
versions.

## Tests

The unit tests of [`benchmark.py`](../../../vllm-common/benchmark.py) are in
[`test_benchmark.py`](../../../vllm-common/test_benchmark.py) in
`vllm-common/`.
