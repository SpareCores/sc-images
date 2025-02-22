## Benchmarking LLM inference speed

Calling `llama-bench` from [`llama.cpp`](https://github.com/ggerganov/llama.cpp)
to benchmark the speed of prompt processing and text generation,
using different models and different number of tokens.

Each benchmark scenario is repeated 5 times and run on its own to be able to
enforce a timeout, which is calculated based on the model size (to be loaded
into memory/VRAM using a conservative 250 MB/sec read speed), the number of
tokens tested, and expected min tokens/sec -- requiring faster inference speed
for more tokens as per below.

**Prompt processing performance targets:**

| Tokens | Expected tokens/sec |
|--------|-------------------|
| 16 | 2 |
| 128 | 10 |
| 512 | 25 |
| 1024 | 50 |
| 4096 | 250 |
| 16384 | 1000 |

**Text generation performance targets:**

| Tokens | Expected tokens/sec |
|--------|-------------------|
| 16 | 1 |
| 128 | 5 |
| 512 | 25 |
| 1024 | 50 |
| 4096 | 250 |

So running the benchmark on a hardware that can generate 512 tokens with 22
tokens/sec speed will not test 1024 and larger token lenghts and will stop early
to save compute resources. If you want to allow longer runs, use the
`--benchmark-timeout-scale` flag to increase the timeouts.

### Usage

```sh
docker run --gpus all --rm --init ghcr.io/sparecores/benchmark-llm:main
```

### Models

The default list of models to download and benchmark is:

- [SmolLM-135M](https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q4_K_M.gguf)
- [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf)
- [gemma-2b](https://huggingface.co/mlabonne/gemma-2b-GGUF/resolve/main/gemma-2b.Q4_K_M.gguf)
- [LLaMA-7b](https://huggingface.co/TheBloke/LLaMA-7b-GGUF/resolve/main/llama-7b.Q4_K_M.gguf)
- [phi-4](https://huggingface.co/microsoft/phi-4-gguf/resolve/main/phi-4-q4.gguf)
- [Llama-3.3-70B](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf)

You can override the default list of models by passing the `--model-urls` flag,
but note that the models should be GGUF files, and ordered by size (start with
the smallest).

The models are cached in the `/models` directory by default, which is a
temporary docker volume, but you can override this by passing the `--models-dir`
flag. If you might need to rerun the benchmark multiple times, you might want to
set a different models directory or attach an external location to avoid
re-downloading the same models.
