# References

Terms specific to this image, alphabetized. Common terms (e.g. AVX-512,
GuideLLM, NUMA) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs,
which also lists the terms each image uses.

- **`--dtype`**: The vLLM option that sets the data type of the model weights
  and activations, such as `bfloat16` or `float32`. With its default, `auto`,
  vLLM derives the data type from the model's configuration. A smaller data
  type needs less memory and bandwidth per weight, but only runs fast on
  hardware that supports it.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **`VLLM_CPU_OMP_THREADS_BIND`**: The environment variable of vLLM's CPU
  backend that sets which CPU cores the OpenMP threads of each worker run on.
  It takes CPU lists such as `0-31`, one per worker, separated by vertical
  bars, so `0-31|32-63` binds two workers to 32 cores each. The value `auto`
  binds each worker to the cores of one NUMA node, and `nobind` turns the
  binding off.
  [vLLM CPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/cpu.html)
