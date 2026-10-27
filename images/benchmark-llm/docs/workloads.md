# Workloads

## How a Run Works

Each run of the [harness](../benchmark.py) goes through the following steps:

1. It logs the CPU model and flags to stderr, and picks the CUDA (Compute
   Unified Device Architecture) or the CPU build of llama.cpp; see
   [GPU Run](./usage.md#gpu-run).
2. It starts downloading all models in one background process, at low
   priority (`nice 19`), one after another in list order. Model files already
   in `--models-dir` are reused.
3. For each model, in list order, it waits for the model's download, finds the
   number of [GPU layers](#gpu-layers) (CUDA build only), and then runs
   [prompt processing](#prompt-processing) and
   [text generation](#text-generation), each over increasing token counts.

Each measurement of one model, task, and token count runs as a separate
`llama-bench` process, one at a time, so that it can have its own timeout. A
timeout or failure skips the rest of the task or the run; see
[Timeouts and Early Stops](#timeouts-and-early-stops).

## Models

The default models are the following, from smallest to largest. The file
sizes are approximate:

| Model | Parameters | Quantization | File size | Download |
| --- | --- | --- | --- | --- |
| SmolLM | 135M | `Q4_K_M` | \~100 MB | [GGUF](https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/b35f229108e4c5a4deeccccf9bf75cd46006fa7b/SmolLM-135M.Q4_K_M.gguf) |
| Qwen1.5-Chat | 0.5B | `Q4_K_M` | \~400 MB | [GGUF](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/cfab082d2fef4a8736ef384dc764c2fb6887f387/qwen1_5-0_5b-chat-q4_k_m.gguf) |
| Gemma | 2B | `Q4_K_M` | \~1.5 GB | [GGUF](https://huggingface.co/mlabonne/gemma-2b-GGUF/resolve/8dd617a80664da2853999b6b1206595f1c33daa2/gemma-2b.Q4_K_M.gguf) |
| LLaMA | 7B | `Q4_K_M` | \~4 GB | [GGUF](https://huggingface.co/TheBloke/LLaMA-7b-GGUF/resolve/0a1200190c8c1baafaa0b496c599ba68ad3d4056/llama-7b.Q4_K_M.gguf) |
| Phi-4 | 14B | 4-bit (`q4`) | \~9 GB | [GGUF](https://huggingface.co/microsoft/phi-4-gguf/resolve/b1e764cfdbdd0a3ed824d6a8424129eb0a2232ff/phi-4-q4.gguf) |
| Llama-3.3-Instruct | 70B | `Q4_K_M` | \~42 GB | [GGUF](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/resolve/7bcd751e48e909680bd2168359dcf7fb53b12da8/Llama-3.3-70B-Instruct-Q4_K_M.gguf) |

Each URL is pinned to a commit of its Hugging Face repository, so every run
downloads the same files. The order matters: a model that fails its first
measurement stops the run and skips all larger models.

## Prompt Processing

Prompt processing measures how fast the model reads its input. Each
measurement processes a prompt of 16, 128, 512, 1,024, 4,096, or 16,384 tokens
without generating any (`llama-bench -p <n> -n 0`). Its output lines have
`n_gen: 0`, and `n_prompt` is the token count.

## Text Generation

Text generation measures how fast the model writes its output, one token after
another. Each measurement generates 16, 128, 512, 1,024, or 4,096 tokens without
a prompt (`llama-bench -n <n> -p 0`). Its output lines have `n_prompt: 0`, and
`n_gen` is the token count.

## llama-bench Settings

Every `llama-bench` measurement uses the following settings, set in
[`benchmark.py`](../benchmark.py):

- **Threads:** the number of physical CPU cores (`-t`), as `psutil` counts
  them; recorded as `n_threads`.
- **Split mode:** `layer` (`-sm layer`).
- **Flash attention:** on (`-fa 1`).
- **Batch sizes:** a logical batch size of 2048 (`-b`) and a physical batch
  size of 512 (`-ub`).
- **Repetitions:** the harness does not set `-r`, so `llama-bench`'s default
  applies; production output has five repetitions per line.
- **Output:** JSON Lines (`-o jsonl`), printed to stdout as is.

## GPU Layers

With the CUDA build, this benchmark sets how many model layers to offload to
the GPUs (`-ngl`) before it measures each model. It tries 999, 40, 24, and 12,
in this order, each in a one-repetition `llama-bench` run with a fixed timeout
of 120 seconds, and uses the first value whose run completes. If none
completes, or with the CPU build, it uses 0. The value is recorded as
`n_gpu_layers`.

## Timeouts and Early Stops

Each measurement's timeout, in seconds, depends on the model size, the token
count, and the expected speed for that task and token count:

```text
timeout = round((model_GiB / 0.25 + 1 + 5 × tokens / expected_speed) × scale)
```

The formula allows 4 seconds per GiB to load the model, 1 second of overhead,
and five repetitions at the expected speed. `scale` is
`--benchmark-timeout-scale` (default 1). The expected speeds, in tokens per
second, rise with the token count:

| Tokens | Prompt processing | Text generation |
| --- | --- | --- |
| 16 | 2 | 1 |
| 128 | 10 | 5 |
| 512 | 25 | 25 |
| 1024 | 50 | 50 |
| 4096 | 250 | 250 |
| 16384 | 1000 | Not run |

For example, text generation of 16 tokens with a model of 4 GiB gets
`4 / 0.25 + 1 + 5 × 16 / 1 = 97` seconds, and at 512 tokens 119 seconds. A
server that generates 15 tokens per second needs \~171 seconds for the five
repetitions at 512 tokens, so its measurement times out and the larger token
counts of that task are skipped. The load allowance adds some slack: a server
slightly below the expected speed can still finish in time.

After a successful measurement of 1,024 or more tokens that is not the last of
its task, this benchmark also estimates whether the next token count can meet
its expected speed. If the measurement took longer than the following limit,
the remaining token counts of that task are skipped:

```text
limit = timeout × expected_speed / next_expected_speed × 1.1
```

`timeout` and `expected_speed` are those of the measurement that just ran, and
`next_expected_speed` is the expected speed of the next token count. For prompt
processing at 1,024 tokens, that limit is 22% of the timeout (50 / 250 × 1.1).

When a measurement times out or fails, the following happens:

- On the first, 16-token measurement of either task, this benchmark stops and
  skips all larger models.
- On a later token count, the remaining token counts of that task are skipped
  for that model.

To allow slower servers more time, raise `--benchmark-timeout-scale`. See
[Missing Results](./limitations.md#missing-results) for the other reasons a
run can stop early and how Spare Cores Navigator shows them.
