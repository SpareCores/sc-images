# vllm-cpu-base-avx2

amd64 vLLM `vllm-openai` CPU image built with **`VLLM_CPU_AVX2=true`** for hosts that have AVX2 but not AVX-512.

Hub `vllm/vllm-openai-cpu` amd64 targets AVX-512; [`benchmark-vllm-cpu-avx2`](../benchmark-vllm-cpu-avx2) uses this base.

Published as `ghcr.io/sparecores/vllm-cpu-base-avx2:main` (amd64 only). Built in [`.github/workflows/push.yml`](../../.github/workflows/push.yml) job `build-vllm-cpu-base-avx2`. Version pin: [`VLLM_VERSION`](../../vllm-common/VLLM_VERSION).
