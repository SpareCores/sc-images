# vllm-cpu-base-avx2

amd64 vLLM `vllm-openai` CPU image built with **`VLLM_CPU_AVX2=true`** for hosts that have AVX2 but not AVX-512.

Hub `vllm/vllm-openai-cpu` amd64 targets AVX-512; [`benchmark-vllm-cpu-avx2`](../benchmark-vllm-cpu-avx2) uses this base.

Published as `ghcr.io/sparecores/vllm-cpu-base-avx2:main` (amd64 only). The folder's [`prepare.sh`](prepare.sh) hook clones the pinned source, sizes compile parallelism on physical RAM, and patches `docker/Dockerfile.cpu`; the [build framework](../../README.md#build-framework) builds it on `linux/amd64`. Version pin: [`VLLM_VERSION`](../../vllm-common/VLLM_VERSION).

CI exports layer cache to **GitHub Actions cache** (`type=gha`, scope `vllm-cpu-base-avx2-amd64-v{VLLM_VERSION}`) and **GHCR** (`buildcache-amd64`). The first successful build populates both; reruns on the same runner class should restore GHA cache quickly. If `buildcache-amd64` does not exist yet, registry import is skipped and the build still runs ([registry cache docs](https://docs.docker.com/build/cache/backends/registry/)).
