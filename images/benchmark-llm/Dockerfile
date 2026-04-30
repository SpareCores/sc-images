# when updating the build number, make sure both AMD64 and ARM64 builds are available
ARG LLAMA_CPP_BUILD=b4753
FROM ghcr.io/ggml-org/llama.cpp:full-${LLAMA_CPP_BUILD} AS base_amd64_cpu
# collect and copy shared libs for CPU-optimized benchmarks on AMD64,
# where the default build is CUDA
COPY extract-shared-cpu-libs.sh /tmp/extract-shared-cpu-libs.sh
ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then /tmp/extract-shared-cpu-libs.sh; fi

FROM ghcr.io/ggml-org/llama.cpp:full-cuda-${LLAMA_CPP_BUILD} AS base_amd64
RUN mv /app /llama_cpp_gpu
COPY --from=base_amd64_cpu /app /llama_cpp_cpu

FROM ghcr.io/ggml-org/llama.cpp:full-${LLAMA_CPP_BUILD} AS base_arm64
RUN mv /app /llama_cpp_cpu

ARG TARGETARCH
FROM base_${TARGETARCH} AS final
RUN pip install psutil
RUN apt-get update -y && apt-get install -y --no-install-recommends curl ca-certificates tar && rm -rf /var/lib/apt/lists/*
RUN ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')" && \
    URL="$(curl -fsSL https://api.github.com/repos/SpareCores/resource-tracker-rs/releases/latest | sed -n "s#.*\"browser_download_url\": \"\\([^\"]*resource-tracker-[^\"]*-linux-${ARCH}-musl.tar.gz\\)\".*#\\1#p" | sed -n '1p')" && \
    curl -fsSL "$URL" | tar -xvzf - resource-tracker && \
    install -m 0755 resource-tracker /usr/local/bin/resource-tracker && \
    rm -f resource-tracker
VOLUME /models

COPY benchmark.py /benchmark.py
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "/benchmark.py"]
