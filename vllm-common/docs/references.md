# References

Terms of the vLLM benchmarks, alphabetized: the options and environment
variables of vLLM that the manual and the images' pages use. Common terms
(e.g. GuideLLM, TTFT, tensor parallelism) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs.

- **`--dtype`**: The vLLM option that sets the data type of the model weights
  and activations, such as `bfloat16` or `float32`. With its default, `auto`,
  vLLM derives the data type from the model's configuration. A smaller data type
  needs less memory and bandwidth per weight, but only runs fast on hardware
  that supports it.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **`--gpu-memory-utilization`**: The vLLM option that sets the fraction of
  device memory, from 0 to 1, that a vLLM worker may use for the model weights,
  intermediate results, and key-value cache. On GPUs, it applies to the memory
  of each GPU; on the CPU backend of vLLM, each worker applies it to the total
  memory of its NUMA node, despite the option's name. The memory left after the
  weights and intermediate results goes to the key-value cache, so a higher
  value lets the server hold more concurrent sequences.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
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
- **`VLLM_CPU_OMP_THREADS_BIND`**: The environment variable of vLLM's CPU
  backend that sets which CPU cores the OpenMP threads of each worker run on. It
  takes CPU lists such as `0-31`, one per worker, separated by vertical bars, so
  `0-31|32-63` binds two workers to 32 cores each. The value `auto` binds each
  worker to the cores of one NUMA node, and `nobind` turns the binding off.
  [vLLM CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)
