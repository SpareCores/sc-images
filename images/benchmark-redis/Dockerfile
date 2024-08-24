FROM redis:7.4.0
ENV DEBIAN_FRONTEND=noninteractive
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y lsb-release curl gpg && \
    curl -fsSL https://packages.redis.io/gpg | gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/redis.list && \
    apt-get update --error-on=any && \
    apt-get install -y memtier-benchmark
COPY benchmark.sh /usr/local/bin/
# Override the ENTRYPOINT, so we can get redis version easier
ENTRYPOINT []