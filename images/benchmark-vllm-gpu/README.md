# benchmark-vllm-gpu

vLLM **GPU serving** benchmark: `vllm serve` with tensor parallelism on all visible GPUs, then [GuideLLM](https://github.com/vllm-project/guidellm) `sweep` per workload.

| Arch | Base |
|------|------|
| amd64 | `vllm/vllm-openai:v{VLLM_VERSION}` |
| arm64 | `ghcr.io/sparecores/vllm-gpu-base:main` |

Published as `ghcr.io/sparecores/benchmark-vllm-gpu:main`. Pins: [`VLLM_VERSION`](../../vllm-common/VLLM_VERSION), [`GUIDELLM_VERSION`](../../vllm-common/GUIDELLM_VERSION). Harness: [`benchmark.py`](../../vllm-common/benchmark.py).

## Models (default ladder)

SmolLM2-135M, Qwen2.5-0.5B, Gemma-2-2B, Qwen3-4B, Phi-4 (GPU), Llama-3.3-70B bnb-4bit (large VRAM).

## Workloads

- **chat**: 256 prompt / 128 output tokens  
- **rag**: 1024 / 256  
- **long**: 4096 / 512 (GPU only)

## Output

JSONL lines (`benchmark=vllm_serving`) with TTFT/TPOT/ITL/E2EL percentiles and throughputs per GuideLLM strategy. See [vllm-common/README](../../vllm-common/README.md).
