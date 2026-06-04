# benchmark-vllm-cpu-avx2

vLLM **CPU serving** for **x86_64 AVX2-only** hosts (no `avx512f`). Base: [`vllm-cpu-base-avx2`](../vllm-cpu-base-avx2) (`VLLM_CPU_AVX2=true` build).

Published as `ghcr.io/sparecores/benchmark-vllm-cpu-avx2:main` (amd64 only). Same GuideLLM harness as [`benchmark-vllm-cpu`](../benchmark-vllm-cpu).

Inspector uses this image when GPU and Hub CPU probes fail on AVX2-only x86.

Harness: [`benchmark.py`](../../vllm-common/benchmark.py).
