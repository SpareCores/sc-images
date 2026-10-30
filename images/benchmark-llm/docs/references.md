# References

Terms specific to this image, alphabetized. Common terms, such as CUDA (Compute
Unified Device Architecture), GGUF (GGML Universal File), and token, are
defined in the [glossary](https://docs.sparecores.com/glossary) of the Spare
Cores docs, which also lists the terms each image uses.

- **`-b`**: The llama.cpp option that sets the logical batch size: the maximum
  number of prompt tokens passed to the model in one call. Each logical batch
  is split into physical batches of the size set by `-ub`, so a larger logical
  batch can process long prompts in fewer calls, at the cost of more memory.
  [llama-bench README](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
- **`-ngl`**: The llama.cpp option that sets how many layers of a model are
  offloaded to the GPUs. The remaining layers run on the CPU, so a model that
  does not fit in GPU memory can still use the GPU in part, and a value higher
  than the model's number of layers, such as 999, offloads all of them.
  [llama-bench README](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
- **`-sm`**: The llama.cpp option that sets the split mode: how a model is
  distributed across several GPUs. The `layer` mode assigns whole layers to
  each GPU, `row` splits the weight tensors of each layer across the GPUs, and
  `none` uses a single GPU.
  [llama-bench README](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
- **`-ub`**: The llama.cpp option that sets the physical batch size: the number
  of tokens computed at once within a logical batch (`-b`). It trades memory
  use against the speed of prompt processing.
  [llama-bench README](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
- **`llama-bench`**: The benchmark tool of llama.cpp. It measures the speed of
  prompt processing and text generation in tokens per second, for given
  models, token counts, and settings. Each test is repeated several times, and
  the tool reports the average and standard deviation, as a table or in
  machine-readable formats such as JSON Lines.
  [llama-bench README](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench)
- **`Q4_K_M`**: A llama.cpp quantization type of the k-quant family. It stores
  most weights with about 4 bits each, in blocks that share scale factors, and
  keeps some tensors at higher precision. The M (medium) variant is a common
  compromise between file size and output quality.
  [Hugging Face GGUF docs](https://huggingface.co/docs/hub/gguf)
