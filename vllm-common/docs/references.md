# References

Terms of the vLLM benchmarks, alphabetized: the options and environment
variables of vLLM, and the LLM serving terms that the manual and the images'
pages use. Common terms
(e.g. GuideLLM, TTFT, tensor parallelism) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs.

- **`bitsandbytes`**: An open-source Python library that quantizes the weights
  of PyTorch models to 8 or 4 bits, so that large models need less GPU memory.
  Hugging Face Transformers and vLLM can load models quantized with it, and its
  4-bit format is also used to fine-tune large models on a single GPU.
  [bitsandbytes on GitHub](https://github.com/bitsandbytes-foundation/bitsandbytes)
- **data parallelism**, **Data parallelism**, **data-parallel**: A way to
  spread model training or inference across several devices by giving each a
  full copy of the model and splitting the data or requests among them. It adds
  throughput with each copy, but every device must hold all of the model's
  weights, unlike tensor and pipeline parallelism, which split one model across
  devices.
  [Hugging Face docs](https://huggingface.co/docs/transformers/perf_train_gpu_many)
- **`--dtype`**: The vLLM option that sets the data type of the model weights
  and activations, such as `bfloat16` or `float32`. With its default, `auto`,
  vLLM derives the data type from the model's configuration. A smaller data type
  needs less memory and bandwidth per weight, but only runs fast on hardware
  that supports it.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **E2EL** (End-to-End Request Latency): The time from sending a request to an
  LLM server until the last output token of its response arrives. It covers
  queuing, prompt processing, and the generation of every output token, so it
  is the time to first token plus the generation time of the further tokens,
  and it sets how long a full response takes.
  [NVIDIA NIM docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- **`--gpu-memory-utilization`**: The vLLM option that sets the fraction of
  device memory, from 0 to 1, that a vLLM worker may use for the model weights,
  intermediate results, and key-value cache. On GPUs, it applies to the memory
  of each GPU; on the CPU backend of vLLM, each worker applies it to the total
  memory of its NUMA node, despite the option's name. The memory left after the
  weights and intermediate results goes to the key-value cache, so a higher
  value lets the server hold more concurrent sequences.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **GuideLLM sweep**: A GuideLLM benchmark profile that measures a server
  across a range of loads. It first runs a synchronous stage, which sends one
  request at a time, and a throughput stage, which sends as many concurrent
  requests as allowed. It then runs the remaining stages, if any, at constant
  request rates between the rates of those two stages. The sweep size is the
  total number of stages.
  [GuideLLM on GitHub](https://github.com/vllm-project/guidellm)
- **Hugging Face model ID**, **Hugging Face model IDs**: The name of a model
  repository on the Hugging Face Hub, in the form `<owner>/<model>`, such as
  `Qwen/Qwen2.5-0.5B-Instruct`. Programs and libraries use it to download the
  model's files, and it appears as the title of the model's page on the Hub.
  [Hugging Face Hub docs](https://huggingface.co/docs/hub/models-the-hub)
- **ITL** (Inter-Token Latency): The time between two consecutive output tokens
  of a response that an LLM server streams. It shows how smoothly text appears
  while it is generated, and is usually reported as percentiles or the mean
  over all gaps between tokens.
  [NVIDIA NIM docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- **`--kv-cache-memory-bytes`**: The vLLM option that sets the size of the
  key-value cache of each worker in bytes. Without it, vLLM derives the cache
  size from `--gpu-memory-utilization`; this option gives finer control over how
  much memory the cache takes.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **`--max-model-len`**: The vLLM option that sets the context length of a
  model, the maximum number of tokens, prompt and output together, in one
  sequence. The key-value cache must have room for at least one sequence of this
  length, so a shorter context length needs less memory. Without the option,
  vLLM uses the length from the model's configuration.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **`--max-num-batched-tokens`**: The vLLM option that sets the maximum number
  of tokens that a vLLM engine processes in one scheduling step, summed over all
  sequences in the batch. Prompt tokens and generated tokens both count toward
  it, so a lower value limits the work and memory of each step, and a higher
  value lets more prompt tokens be processed at once.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **`--max-num-seqs`**: The vLLM option that sets the maximum number of
  sequences, or requests, that a vLLM engine processes together in one
  scheduling step. Further requests wait in a queue until a running sequence
  finishes. With data parallelism, each replica applies the limit on its own.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **pipeline parallelism**, **Pipeline parallelism**, **pipeline-parallel**: A
  way to split a model across several devices by assigning consecutive groups
  of its layers to each, so that each device holds only part of the weights.
  The output of one stage passes to the next, and the stages work on different
  requests or micro-batches at the same time to keep all devices busy. It needs
  less communication between devices than tensor parallelism.
  [Hugging Face docs](https://huggingface.co/docs/transformers/perf_train_gpu_many)
- **TPOT** (Time per Output Token): The average time an LLM server takes to
  generate each output token of a response after the first one, that is, the
  generation time after the first token divided by the number of further
  tokens. Together with the time to first token, it sets how long a full
  response takes.
  [NVIDIA NIM docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- **TPS** (Tokens per Second): The number of tokens an LLM server processes or
  generates per second. Benchmarks count it in different ways, over output
  tokens only or over prompt and output tokens, per request or summed over all
  concurrent requests, so a TPS figure is compared only with one counted the
  same way.
  [NVIDIA NIM docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)
- **`VLLM_CPU_OMP_THREADS_BIND`**: The environment variable of vLLM's CPU
  backend that sets which CPU cores the OpenMP threads of each worker run on. It
  takes CPU lists such as `0-31`, one per worker, separated by vertical bars, so
  `0-31|32-63` binds two workers to 32 cores each. The value `auto` binds each
  worker to the cores of one NUMA node, and `nobind` turns the binding off.
  [vLLM CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)
