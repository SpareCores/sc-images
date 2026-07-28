ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main

FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker
FROM postgres:18

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    mkdir -p /var/lib/apt/lists/partial && \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      python3 \
      python3-psycopg \
      zstd \
    && rm -rf /var/lib/apt/lists/*

COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY benchmark.py db_dataset_cache.py /benchmark/

ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark/benchmark.py"]
