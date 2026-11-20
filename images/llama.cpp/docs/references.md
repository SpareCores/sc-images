# Glossary

Terms specific to this image, alphabetized. Common terms, such as CUDA
(Compute Unified Device Architecture), GGUF (GGML Universal File), and
llama.cpp, are defined in the
[Spare Cores Glossary](https://docs.sparecores.com/glossary), which also lists
the terms each image uses.

- **`llama-server`**: The HTTP server of llama.cpp. It loads a model in the
  GGUF format and serves it over an HTTP API, with endpoints for text
  completion, chat, and embeddings, a `/health` endpoint that reports whether
  the server is ready, and a web interface. By default, it listens on port
  8080 of `127.0.0.1`. Most of its command-line options can also be set as
  environment variables named `LLAMA_ARG_<OPTION>`, such as `LLAMA_ARG_PORT`.
  [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
