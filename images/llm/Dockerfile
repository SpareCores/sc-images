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
RUN pip install psutil
RUN mkdir -p /models

COPY benchmark.py /benchmark.py
ENTRYPOINT ["/bin/bash"]
CMD ["/benchmark.py"]
