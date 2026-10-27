# Contributor Guide

## Image Layout

The [`Dockerfile`](../Dockerfile) builds the image from the official llama.cpp
container images, at the build set by `LLAMA_CPP_BUILD`:

- **AMD64:** the final stage is `ghcr.io/ggml-org/llama.cpp:full-cuda-<build>`,
  with its llama.cpp build moved to `/llama_cpp_gpu`. The CPU build from
  `ghcr.io/ggml-org/llama.cpp:full-<build>` is copied to `/llama_cpp_cpu`.
- **Arm64:** the final stage is `ghcr.io/ggml-org/llama.cpp:full-<build>`, with
  its llama.cpp build moved to `/llama_cpp_cpu`. There is no CUDA build.

On AMD64, [`extract-shared-cpu-libs.sh`](../extract-shared-cpu-libs.sh) copies
every shared library that `ldd` reports for the CPU build's `llama-bench` into
`/llama_cpp_cpu/libs`.

The image installs `psutil` with pip, copies the `resource-tracker` binary, and
declares `/models` as a volume. Its entrypoint is
`resource-tracker -- /benchmark.py`, with `TRACKER_QUIET=true`.

## Build Arguments

| Argument | Meaning | Default |
| --- | --- | --- |
| `LLAMA_CPP_BUILD` | Tag of the llama.cpp container images. Both the AMD64 and the Arm64 images must exist for the tag. | `b4753` |
| `RESOURCE_TRACKER_IMAGE` | Image to copy the `resource-tracker` binary from. CI passes the image for the target architecture, `ghcr.io/sparecores/resource-tracker:main-<arch>`. | `ghcr.io/sparecores/resource-tracker:main` |

CI builds the image for `amd64` and `arm64`, after the `resource-tracker` image
it depends on.

Changing `LLAMA_CPP_BUILD` or the default models makes new results
incomparable with the published ones, which would need a rerun on every server
type; see [Fixed Models and Build](./limitations.md#fixed-models-and-build).
