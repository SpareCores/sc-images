# llama.cpp Server

This image runs `llama-server`, the HTTP server of llama.cpp, built from
source with CUDA (Compute Unified Device Architecture) for NVIDIA GPUs. It
contains only the server and the libraries of its build, on the CUDA 12.4.0
runtime image of NVIDIA for Ubuntu 22.04. This image measures nothing itself.

This image reports no results: its container prints the log of `llama-server`
and serves the llama.cpp HTTP API on port 8080. The
[`benchmark-llm`](../benchmark-llm/README.md) benchmark, which measures LLM
inference speed with llama.cpp for
[Spare Cores Navigator](https://docs.sparecores.com/overview), runs the
upstream llama.cpp images, not this one. No other image in the
[`sc-images` repository](https://github.com/SpareCores/sc-images) builds on
this image.

## Usage

To start a server for a GGUF (GGML Universal File) model file on the host via
Docker:

```bash
docker run --rm --gpus all -p 8080:8080 -v <models>:/models ghcr.io/sparecores/llama.cpp:main -m /models/<file>.gguf
```

Replace `<models>` with the absolute path of the host folder that holds the
model, and `<file>.gguf` with the file name of the model in that folder. The
container entrypoint is `llama-server`, so the arguments after the image name
go to `llama-server`. The `-m` option comes from llama.cpp, and `/models` is
only an example mount point: the image defines no volume.

The `--gpus all` option needs a host with an NVIDIA GPU, the NVIDIA driver,
and the NVIDIA Container Toolkit. The [`Dockerfile`](./Dockerfile) notes that
its CUDA version needs to "generally match the container host's environment",
so check that the host's NVIDIA driver supports CUDA 12.4.

To check from the host that the server responds:

```bash
curl -f http://localhost:8080/health
```

Docker runs the same request inside the container as the image's health
check, with Docker's default interval, timeout, and retries, and shows the
result as the health status of the container.

The build enables the llama.cpp support for loading models from a URL
(`LLAMA_CURL=ON`), so `llama-server` can also download a model instead of
reading a mounted file.

### Settings

This image sets one environment variable, `LLAMA_ARG_HOST=0.0.0.0`, so that
`llama-server` listens on all interfaces of the container, and `-p` can
publish its port. The image does not declare the port with `EXPOSE`. All other
settings are the options of `llama-server`, given after the image name or as
`LLAMA_ARG_*` environment variables with `-e`. See the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
for the options and their defaults.

The health check always requests port 8080 on `localhost`. When `--port` or
`LLAMA_ARG_PORT` moves the server to another port, or `LLAMA_ARG_HOST` binds
it to an address other than `0.0.0.0` or the loopback address, Docker marks
the container `unhealthy`, while the server keeps running.

### Output

The container prints the log of `llama-server`. The HTTP API (its endpoints
and their request and response formats) is that of llama.cpp, and this image
does not change it; the llama.cpp server documentation linked in
[Settings](#settings) describes it.

## Build

The [`Dockerfile`](./Dockerfile) clones the
[llama.cpp repository](https://github.com/ggml-org/llama.cpp) and builds it
with CMake on `nvidia/cuda:12.4.0-devel-ubuntu22.04`, with the CUDA backend
(`GGML_CUDA=ON`) and the support for loading models from a URL
(`LLAMA_CURL=ON`). CI builds with the default `CUDA_DOCKER_ARCH`, which a
comment in the `Dockerfile` describes as all supported CUDA architectures.

The published image copies the shared libraries of the build and the
`llama-server` binary onto `nvidia/cuda:12.4.0-runtime-ubuntu22.04`, and adds
`curl` for the health check. The image sets no user, so `llama-server` runs
as root.

The `Dockerfile` also defines a `full` stage with all llama.cpp tools and
Python scripts, and a `light` stage with `llama-cli` only. CI builds the last
stage, `server`, so only the server is published, for AMD64 and Arm64.

### llama.cpp Version

The `Dockerfile` sets the build argument `LLAMA_COMMIT_HASH` to the llama.cpp
release `b4753`, but declares it only before the first `FROM`, where it is not
visible to the `RUN` steps of the build stage. The `git checkout` therefore
gets no ref, and the build compiles the default branch of llama.cpp as it is
at build time. Passing `--build-arg LLAMA_COMMIT_HASH=<ref>` does not change
this.

As a result, the llama.cpp version of the published image depends on when CI
last built it, and the AMD64 and Arm64 builds of one CI run can contain
different llama.cpp commits.

## References

The terms of this image, such as `llama-server`, are defined in its
[Glossary](./docs/references.md). Common terms, such as CUDA, GGUF, and the
NVIDIA Container Toolkit, are defined in the
[Spare Cores Glossary](https://docs.sparecores.com/glossary).
