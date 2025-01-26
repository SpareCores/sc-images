FROM ghcr.io/ggerganov/llama.cpp:full AS base_cpu
# collect and copy shared libs for CPU-optimized benchmarks on AMD64,
# where the default build is CUDA
COPY extract-shared-cpu-libs.sh /tmp/extract-shared-cpu-libs.sh
RUN if [ "$TARGETARCH" = "amd64" ]; then /tmp/extract-shared-cpu-libs.sh; fi
RUN mv /app /llama_cpp_cpu

FROM ghcr.io/ggerganov/llama.cpp:full-cuda AS base_amd64
RUN mv /app /llama_cpp_gpu

FROM ghcr.io/ggerganov/llama.cpp:full AS base_arm64
RUN echo 2

ARG TARGETARCH
FROM base_${TARGETARCH} AS final
COPY --from=base_cpu /llama_cpp_cpu /llama_cpp_cpu

# using Q4_K/Q4_K_M quantization by default
# ADD \
#   https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/blob/main/SmolLM-135M.Q4_K_M.gguf \
#   https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/blob/main/qwen1_5-0_5b-chat-q4_k_m.gguf \
#   https://huggingface.co/bartowski/codegemma-2b-GGUF/blob/main/codegemma-2b-Q4_K_M.gguf \
#   # 8 B / 5 GB
#   https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/blob/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf \
#   # 14 B / 9 GB
#   https://huggingface.co/microsoft/phi-4-gguf/blob/main/phi-4-q4.gguf \
#   # 70 B / 50 GB
#   https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/blob/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf \
#   /models/

COPY benchmark.py /benchmark.py
ENTRYPOINT ["/bin/bash"]
CMD ["/benchmark.py"]
