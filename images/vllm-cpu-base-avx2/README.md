# vLLM CPU Base (AVX2)

This image provides a vLLM 0.27.1 server built from source for AMD64 CPUs. It
is the base image of
[`benchmark-vllm-cpu-avx2`](../benchmark-vllm-cpu-avx2/README.md), the vLLM
benchmark for AMD64 CPUs with AVX2 (Advanced Vector Extensions 2) but without
AVX-512 (Advanced Vector Extensions 512). It builds the `vllm-openai` stage of
the upstream vLLM CPU
[`Dockerfile`](https://github.com/vllm-project/vllm/blob/v0.27.1/docker/Dockerfile.cpu),
so its container starts `vllm serve`.

This image runs a server, not a benchmark, so it produces no benchmark
results. Its container prints only the log of `vllm serve`. The
[vLLM family manual](../../vllm-common/README.md) describes how the benchmark
built on this image measures LLM serving and how to read its results.

## Usage

To start a vLLM server for one model via Docker:

```bash
docker run --rm ghcr.io/sparecores/vllm-cpu-base-avx2:main <owner>/<model>
```

Replace `<owner>/<model>` with the Hugging Face model ID of the model to
serve. The container
[entrypoint](https://github.com/vllm-project/vllm/blob/v0.27.1/docker/Dockerfile.cpu#L307)
is `vllm serve`, so the arguments after the image name go to `vllm serve`.

The image does not check which instruction sets the host CPU supports. The
vLLM benchmarks test an AMD64 host for AVX-512 in their harness, which this
image does not include, and `benchmark-vllm-cpu-avx2` skips that test.

This image sets no Docker options for ports, shared memory, or the model
cache. The vLLM benchmarks need `--shm-size=4g`, because they fail with
Docker's default 64 MiB `/dev/shm`; see the manual's
[Usage](../../vllm-common/README.md#usage).

The image is published for AMD64 only, under two tags: `main-amd64`, which
`benchmark-vllm-cpu-avx2` builds on, and `main`, which CI creates from it.

## Build

CI builds this image from the vLLM source at the version that
[`VLLM_VERSION`](../../vllm-common/VLLM_VERSION) pins.
[`benchmark-vllm-cpu`](../benchmark-vllm-cpu/README.md) builds on the vLLM CPU
image from Docker Hub at the same version.
The image has no `Dockerfile` of its own. Its [`prepare.sh`](./prepare.sh)
hook clones vLLM and patches the upstream
[`docker/Dockerfile.cpu`](https://github.com/vllm-project/vllm/blob/v0.27.1/docker/Dockerfile.cpu),
and CI builds the `vllm-openai` stage of that file. The stage does the
following:

- installs the vLLM CPU build
- installs the `triton-cpu` wheel
- copies the `examples/` folder of vLLM
- sets the container entrypoint to `vllm serve`

The upstream `Dockerfile` builds on `ubuntu:22.04` and uses Python 3.12 by
default. The hook patches only this file and its `.dockerignore`; for
example, it pins `ubuntu:22.04` by digest. With the compiler cache on, as in
CI, the build also points the compiler launchers in vLLM's `setup.py` at a
wrapper. See [Dockerfile Patches](./docs/contributors.md#dockerfile-patches)
and [Compiler Cache](./docs/contributors.md#compiler-cache) for the details.

On AMD64, the vLLM 0.27.1 build compiles three variants of the vLLM CPU
extension: one for AVX-512 with AMX (Advanced Matrix Extensions), one for
AVX-512, and one for AVX2
([`cpu_extension.cmake`](https://github.com/vllm-project/vllm/blob/v0.27.1/cmake/cpu_extension.cmake#L131-L152)).
The [`BUILD_ARGS`](./BUILD_ARGS) file passes `VLLM_CPU_AVX2=true` to the
build, but the upstream `Dockerfile` of vLLM 0.27.1 declares no such build
argument, so the setting does not change the build.

We do not plan to change the pinned version, because results with another
version would not be comparable; see
[Version Changes](../../vllm-common/docs/contributors.md#version-changes).

## References

Common terms, such as AVX2, AVX-512, and `vllm serve`, are defined in the
[Spare Cores Glossary](https://docs.sparecores.com/glossary).

## Contributing

To rebuild this image locally or change how CI builds it, see the
[contributor guide](./docs/contributors.md).
