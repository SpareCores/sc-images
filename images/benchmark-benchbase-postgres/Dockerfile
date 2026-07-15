ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
ARG BENCHBASE_IMAGE=benchbase.azurecr.io/benchbase-postgres@sha256:9a25ca9a86e0a2168a9b16e09095c6d63018f8b590c01c0032b7d8aeb83967c9
ARG POSTGRES_IMAGE=postgres:18

FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker
FROM ${POSTGRES_IMAGE} AS postgres_client
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
    && rm -rf /var/lib/apt/lists/*

COPY --from=postgres_client /usr/lib/postgresql/18/bin/pg_dump /usr/local/bin/pg_dump
COPY --from=postgres_client /usr/lib/postgresql/18/bin/psql /usr/local/bin/psql

COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY benchmark.py db_dataset_cache.py /benchmark/

ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark/benchmark.py"]
