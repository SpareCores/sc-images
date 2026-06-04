# benchmark-vllm-cpu

vLLM **CPU serving** on **amd64 (AVX-512)** and **arm64** using Hub `vllm-openai-cpu`, plus GuideLLM load tests.

Published as `ghcr.io/sparecores/benchmark-vllm-cpu:main`. Pins: [`VLLM_VERSION`](../../vllm-common/VLLM_VERSION), [`GUIDELLM_VERSION`](../../vllm-common/GUIDELLM_VERSION).

Inspector task `vllm` probes this image after GPU fails and before [`benchmark-vllm-cpu-avx2`](../benchmark-vllm-cpu-avx2) on AVX2-only x86.

**amd64 note:** Hub CPU amd64 requires **AVX-512**. AVX2-only hosts must use the AVX2 image.

Default CPU GuideLLM plan: `synchronous` + `throughput` per workload (faster than full sweep). Override with `GUIDELLM_CPU_PROFILES=sweep`.

Harness: [`benchmark.py`](../../vllm-common/benchmark.py).
