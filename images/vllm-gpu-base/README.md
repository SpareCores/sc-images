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

GitHub Actions clones `v${VLLM_VERSION}` into `vllm-common/.vllm-src` and runs upstream `docker/Dockerfile` with target `vllm-openai` on **linux/arm64** only. See [`.github/workflows/push.yml`](../../.github/workflows/push.yml) job `build-vllm-gpu-base`.

Build-args (GH200-oriented, from [vLLM Docker docs](https://docs.vllm.ai/en/stable/deployment/docker/)):

- `max_jobs=66`
- `nvcc_threads=2`
- `torch_cuda_arch_list=9.0 10.0+PTX`
- `RUN_WHEEL_CHECK=false`

Published as `ghcr.io/sparecores/vllm-gpu-base:main-arm64` and promoted to `:main` (no amd64 leg).

The repo [`Dockerfile`](Dockerfile) is a placeholder; the image is produced only via CI.
