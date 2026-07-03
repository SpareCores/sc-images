ARG RESOURCE_TRACKER_IMAGE=ghcr.io/sparecores/resource-tracker:main
ARG HAMMERDB_IMAGE=ghcr.io/sparecores/hammerdb-postgres:main

FROM ${RESOURCE_TRACKER_IMAGE} AS resource_tracker
FROM ${HAMMERDB_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y --no-install-recommends \
      python3 \
      python3-psycopg \
    && rm -rf /var/lib/apt/lists/*

COPY --from=resource_tracker /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
COPY benchmark.py /benchmark.py

ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--", "python3", "/benchmark.py"]
