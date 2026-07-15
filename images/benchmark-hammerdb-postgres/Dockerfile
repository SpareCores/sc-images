ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
ARG HAMMERDB_VERSION=5.0
ARG HAMMERDB_IMAGE=tpcorg/hammerdb:v5.0-postgres

FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker
FROM ${HAMMERDB_IMAGE}

USER root
ENV DEBIAN_FRONTEND=noninteractive

# Install PostgreSQL 18 clients so pg_dump/psql match benchmark-postgres-server:main.
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    mkdir -p /var/lib/apt/lists/partial && \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      gnupg \
      python3 \
      python3-psycopg \
      zstd \
    && curl -fsSL https://www.postgresql.org/media/keys/ACCC4CF8.asc \
      | gpg --dearmor -o /usr/share/keyrings/postgresql.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/postgresql.gpg] http://apt.postgresql.org/pub/repos/apt noble-pgdg main" \
      > /etc/apt/sources.list.d/pgdg.list \
    && apt-get update --error-on=any \
    && apt-get install -y --no-install-recommends postgresql-client-18 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY benchmark.py db_dataset_cache.py /benchmark/

ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark/benchmark.py"]
