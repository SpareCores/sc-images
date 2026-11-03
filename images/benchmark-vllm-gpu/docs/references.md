# References

Terms specific to this image, alphabetized. Common terms (e.g. GuideLLM,
tensor parallelism, TTFT) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs,
which also lists the terms each image uses.

- **`--kv-cache-memory-bytes`**: The vLLM option that sets the size of the
  key-value cache on each GPU in bytes. Without it, vLLM derives the cache size
  from `--gpu-memory-utilization`; this option gives finer control over how much
  memory the cache takes.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
