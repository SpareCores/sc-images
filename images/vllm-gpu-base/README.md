# vllm-gpu-base

Pinned **arm64 CUDA** build of upstream [vLLM](https://github.com/vllm-project/vllm) `vllm-openai` image. Docker Hub publishes `vllm/vllm-openai` for amd64 only; Grace-Hopper / Grace-Blackwell hosts need a source build.

## Version pin

See [`VLLM_VERSION`](../../vllm-common/VLLM_VERSION).

```bash
curl -sS https://api.github.com/repos/vllm-project/vllm/releases/latest \
  | jq -r '.tag_name' | sed 's/^v//'
```

Verify Hub tags for amd64 GPU and CPU images before merging a bump.

## CI build

The folder's [`prepare.sh`](prepare.sh) hook clones `v${VLLM_VERSION}` into `vllm-common/.vllm-src`, patches the upstream Dockerfile for RAM-aware parallelism, and the framework builds `docker/Dockerfile` (target `vllm-openai`) on **linux/arm64** only (`PLATFORMS`). See the [build framework](../../README.md#build-framework).

### Parallelism and caching

After zram setup, [`.github/scripts/compute-build-parallelism.sh`](../../.github/scripts/compute-build-parallelism.sh) scales runtime compile jobs **linearly from a 64 GiB reference** (the size that completes reliably on `c6g.8xlarge`):

- **Self-hosted:** `effective_gib = (MemTotal + 50% of SwapTotal) / 1024`; `max_jobs ≈ 32 × effective_gib / 64`, capped by CPU count, `VLLM_COMPILE_GIB_PER_SLOT` (3 GiB/slot), and workflow caps; `nvcc_threads = 2` when `max_jobs ≥ 4`, else `1`; BuildKit `max-parallelism` scales the same way (reference 16 at 64 GiB).
- **GitHub-hosted (`ubuntu-24.04-arm`):** conservative profile — physical RAM only, 6 GiB/slot, `max_jobs ≤ 2`, `nvcc_threads = 1`, BuildKit `max-parallelism ≤ 2` (zram enabled via [`ZRAM`](ZRAM) for swap pressure, but not counted as compile budget).

**Cache-stable Dockerfile args** always use the 64 GiB reference (`max_jobs=32`, `nvcc_threads=2`, `vllm_ci_cache_bust=<version>` only). Runtime values (`MAX_JOBS`, `NVCC_THREADS`, `CARGO_BUILD_JOBS`) are passed via a BuildKit secret so layers compiled on a large self-hosted runner remain reusable on smaller GitHub-hosted arm64 runners (cache hit → no recompile).

Other build-args:

- `torch_cuda_arch_list=9.0 10.0+PTX`
- `RUN_WHEEL_CHECK=false`

CI prunes Docker and removes unused SDKs before build; the image needs ~tens of GiB under `/var/lib/docker` (e.g. `flashinfer-jit-cache` is a 2 GiB wheel with a multi-GiB extract).

Published as `ghcr.io/sparecores/vllm-gpu-base:main-arm64` and promoted to `:main` (no amd64 leg).

The actual build uses the cloned upstream Dockerfile (`DOCKERFILE`/`CONTEXT` metadata point at `vllm-common/.vllm-src`); the repo [`Dockerfile`](Dockerfile) is only a placeholder marker.
