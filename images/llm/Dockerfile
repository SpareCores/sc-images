
FROM ghcr.io/ggerganov/llama.cpp:full-b4508 AS cpu_base
RUN mv /app /app_cpu


FROM --platform=linux/amd64 ghcr.io/ggerganov/llama.cpp:full-cuda-b4508 AS final
# there's no CUDA build for arm64
FROM --platform=linux/arm64 ghcr.io/ggerganov/llama.cpp:full-b4508 AS final

# move GPU build
RUN if [ "$TARGETARCH" = "amd64" ]; then mv /app /app_gpu; fi

# CPU build
COPY --from=cpu_base /app /app


