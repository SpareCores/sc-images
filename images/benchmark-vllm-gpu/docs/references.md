# References

Terms specific to this image, alphabetized. Common terms (e.g. GuideLLM,
tensor parallelism, TTFT) are defined in the
[glossary](https://docs.sparecores.com/glossary) of the Spare Cores docs,
which also lists the terms each image uses.

- **`--gpu-memory-utilization`**: The vLLM option that sets the fraction of
  each GPU's memory, from 0 to 1, that a vLLM instance may use for the model
  weights, intermediate results, and key-value cache. The memory left after the
  weights and intermediate results goes to the key-value cache, so a higher
  value lets the server hold more concurrent sequences.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **`--max-model-len`**: The vLLM option that sets the context length of a
  model: the maximum number of tokens, prompt and output together, in one
  sequence. The key-value cache must have room for at least one sequence of
  this length, so a shorter context length needs less GPU memory. Without the
  option, vLLM uses the length from the model's configuration.
  [vLLM engine arguments](https://docs.vllm.ai/en/latest/configuration/engine_args.html)
- **GuideLLM sweep**: A GuideLLM benchmark profile that measures a server
  across a range of loads. It first runs a synchronous stage, which sends one
  request at a time, and a throughput stage, which sends as many concurrent
  requests as allowed. It then runs the remaining stages, if any, at constant
  request rates between the rates of those two stages. The sweep size is the
  total number of stages.
  [GuideLLM on GitHub](https://github.com/vllm-project/guidellm)
- **`vllm serve`**: The vLLM command that starts an HTTP server for a model,
  with an API compatible with OpenAI's, such as the `/v1/completions` and
  `/v1/chat/completions` endpoints. It loads the model on the available
  devices, batches concurrent requests, and answers on a `/health` endpoint
  once it is ready.
  [vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)
