ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
ARG VLLM_VERSION=0.22.1
ARG GUIDELLM_VERSION=0.6.0
FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker

FROM vllm/vllm-openai-cpu:v${VLLM_VERSION}-x86_64 AS base_amd64
FROM vllm/vllm-openai-cpu:v${VLLM_VERSION}-arm64 AS base_arm64
ARG TARGETARCH
FROM base_${TARGETARCH} AS vllm_base
ARG GUIDELLM_VERSION

RUN pip install --no-cache-dir psutil "guidellm[recommended]==${GUIDELLM_VERSION}"
COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY vllm-common/benchmark.py vllm-common/VLLM_VERSION vllm-common/GUIDELLM_VERSION /

ENV TRACKER_QUIET=true
ENV BENCHMARK_VLLM_MODE=cpu
ENV VLLM_CPU_OMP_THREADS_BIND=auto
ENV USER=benchmark
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor
ENV HF_HOME=/models
ENV HUGGINGFACE_HUB_CACHE=/models
VOLUME /models

ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark.py"]
