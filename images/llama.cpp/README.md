# llama.cpp Server

This image runs `llama-server`, the HTTP server of llama.cpp, built from
source with CUDA (Compute Unified Device Architecture) for NVIDIA GPUs. It
includes the server and the libraries of its build on NVIDIA's CUDA
12.4.0 runtime image for Ubuntu 22.04. It is published for AMD64 and Arm64.

This image runs a server, not a benchmark, so it produces no benchmark
results. Its container prints the log of `llama-server` and serves the
llama.cpp HTTP API on port 8080. No other image in the
[`sc-images` repository](https://github.com/SpareCores/sc-images) builds on
this one: the [`benchmark-llm`](../benchmark-llm/README.md) benchmark runs
the upstream llama.cpp images instead.

## Usage

To start a server for a GGUF (GGML Universal File) model file on the host via
Docker:

```bash
docker run --rm --gpus all -p 8080:8080 -v <models>:/models ghcr.io/sparecores/llama.cpp:main -m /models/<file>.gguf
```

The command has the following parts:

- **`--gpus all`:** Needs a host with an NVIDIA GPU, the NVIDIA driver, and
  the NVIDIA Container Toolkit. The [`Dockerfile`](./Dockerfile) notes that
  its CUDA version needs to "generally match the container host's
  environment", so check that the host's NVIDIA driver supports CUDA 12.4.
- **`-v <models>:/models`:** Mounts the host folder that holds the model.
  Replace `<models>` with its absolute path. `/models` is only an example
  mount point: the image defines no volume.
- **`-m /models/<file>.gguf`:** The `llama-server` option that sets the model
  file. Replace `<file>.gguf` with the file name of the model. The container
  entrypoint is `llama-server`, so the arguments after the image name go to
  that server.

The build enables the llama.cpp support for loading models from a URL
(`LLAMA_CURL=ON`), so `llama-server` can also download a model instead of
reading a mounted file, with its `--model-url` or `--hf-repo` option; see the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server#common-params).

To check from the host that the server responds:

```bash
curl -f http://localhost:8080/health
```

The image's health check runs the same request inside the container, with
Docker's default interval, timeout, and retries, and Docker shows the result
as the health status of the container.

### Settings

The `Dockerfile` sets one environment variable, `LLAMA_ARG_HOST=0.0.0.0`, so
that `llama-server` listens on all interfaces of the container, and `-p` can
publish its port. The image does not declare the port with `EXPOSE`. All other
settings are the options of `llama-server`, given after the image name or as
`LLAMA_ARG_*` environment variables with `-e`. See the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
for the options and their defaults.

The image sets no API key, so the API accepts requests from any client that
reaches the published port. To require a key, pass the `--api-key` option of
`llama-server` or set `LLAMA_API_KEY`.

The health check always requests port 8080 on `localhost`. Docker marks the
container `unhealthy`, while the server keeps running, in the following cases:

- `--port` or `LLAMA_ARG_PORT` moves the server to another port.
- `LLAMA_ARG_HOST` binds the server to an address other than `0.0.0.0` or the
  loopback address.

To use another port on the host, change only the host side of `-p`, such as
`-p 9000:8080`: the container port stays 8080, so the health check keeps
working.

### Output

The container prints the log of `llama-server`. The HTTP API (its endpoints
and their request and response formats) is that of llama.cpp, and this image
does not change it; the
[llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
describes it.

## Build

The [`Dockerfile`](./Dockerfile) clones the
[llama.cpp repository](https://github.com/ggml-org/llama.cpp) and builds it
with CMake on `nvidia/cuda:12.4.0-devel-ubuntu22.04`, with the CUDA backend
(`GGML_CUDA=ON`) and `LLAMA_CURL=ON`. CI builds with the default
`CUDA_DOCKER_ARCH`, so the build passes no list of CUDA architectures to
CMake, and the llama.cpp build files choose which NVIDIA GPU generations to
compile for.

The published image has the shared libraries of the build and the
`llama-server` binary on `nvidia/cuda:12.4.0-runtime-ubuntu22.04`, plus
`curl` for the health check. The image sets no user, so `llama-server` runs
as root.

The `Dockerfile` also defines a `full` stage with all llama.cpp tools and
Python scripts, and a `light` stage with `llama-cli` only. CI builds the last
stage, `server`, so only the server is published.

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

The term `llama-server` is defined in its
[Glossary](./docs/references.md). Common terms, such as CUDA, GGUF, and the
NVIDIA Container Toolkit, are defined in the
[Spare Cores Glossary](https://docs.sparecores.com/glossary).
