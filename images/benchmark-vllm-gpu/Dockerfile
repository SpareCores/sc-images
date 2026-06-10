ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
ARG VLLM_VERSION=0.22.1
ARG GUIDELLM_VERSION=0.6.0
FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker

FROM vllm/vllm-openai:v${VLLM_VERSION}
ARG GUIDELLM_VERSION

RUN pip install --no-cache-dir psutil "guidellm[recommended]==${GUIDELLM_VERSION}"
COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY vllm-common/benchmark.py vllm-common/VLLM_VERSION vllm-common/GUIDELLM_VERSION /

ENV TRACKER_QUIET=true
ENV BENCHMARK_VLLM_MODE=gpu
ENV USER=benchmark
ENV HOME=/tmp
ENV TORCHINDUCTOR_CACHE_DIR=/tmp/torch_inductor
ENV HF_HOME=/models
ENV HUGGINGFACE_HUB_CACHE=/models
VOLUME /models

ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark.py"]
