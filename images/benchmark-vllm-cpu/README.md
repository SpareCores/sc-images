# benchmark-vllm-cpu

vLLM **CPU serving** on **amd64 (AVX-512)** and **arm64** using Hub `vllm-openai-cpu`, plus GuideLLM load tests.

Published as `ghcr.io/sparecores/benchmark-vllm-cpu:main`. Pins: [`VLLM_VERSION`](../../vllm-common/VLLM_VERSION), [`GUIDELLM_VERSION`](../../vllm-common/GUIDELLM_VERSION).

Inspector task `vllm` probes this image after GPU fails and before [`benchmark-vllm-cpu-avx2`](../benchmark-vllm-cpu-avx2) on AVX2-only x86.

**amd64 note:** Hub CPU amd64 requires **AVX-512**. AVX2-only hosts must use the AVX2 image.

Default GuideLLM plan: `sweep` (3 steps per workload). Tighter: `GUIDELLM_SWEEP_SIZE=2`. Legacy: `GUIDELLM_CPU_PROFILES=legacy`.

Default model ladder matches GPU; models are skipped on CPU only when they do not fit in RAM or
use GPU-only serve flags (e.g. Llama-3.3-70B bitsandbytes). Gated models need `HF_TOKEN`.
Harness: [`benchmark.py`](../../vllm-common/benchmark.py).
