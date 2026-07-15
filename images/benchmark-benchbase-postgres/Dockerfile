ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
ARG BENCHBASE_IMAGE=benchbase.azurecr.io/benchbase-postgres@sha256:9a25ca9a86e0a2168a9b16e09095c6d63018f8b590c01c0032b7d8aeb83967c9

FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker
FROM ${BENCHBASE_IMAGE}

USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    mkdir -p /var/lib/apt/lists/partial && \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-psycopg \
      curl \
      zstd \
      postgresql-client \
    && rm -rf /var/lib/apt/lists/*

COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY benchmark.py db_dataset_cache.py /benchmark/

ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark/benchmark.py"]
