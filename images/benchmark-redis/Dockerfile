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
    apt-get install -y memtier-benchmark python-is-python3 python3-psutil
COPY benchmark.py /usr/local/bin/
COPY --from=ghcr.io/sparecores/resource-tracker:main /usr/local/bin/resource-tracker /usr/local/bin/resource-tracker
ENV TRACKER_QUIET=true
ENTRYPOINT ["/usr/local/bin/resource-tracker", "--"]
CMD ["nice", "-n", "-20", "python", "/usr/local/bin/benchmark.py"]