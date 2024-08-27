FROM ubuntu:24.04 as base

RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y python-is-python3 python3-psutil

FROM base as build_base
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y build-essential curl git unzip cargo
FROM build_base as binserve
ENV BINSERVE_VER=1e32f94
RUN --mount=type=tmpfs,target=/tmp,rw \
    cargo install --git https://github.com/mufeedvh/binserve.git --rev ${BINSERVE_VER}
FROM build_base as wrk
ENV WRK_VER=4.2.0
RUN --mount=type=tmpfs,target=/tmp,rw \
    git clone --depth 1 --branch 4.2.0 https://github.com/wg/wrk.git /usr/local/wrk && \
    cd /usr/local/wrk && \
    make -j $(nproc)

FROM base as final
ADD binserve/ /usr/local/binserve/
COPY --from=binserve /root/.cargo/bin/binserve /usr/local/bin/binserve
COPY --from=wrk /usr/local/wrk/wrk /usr/local/bin/wrk

COPY benchmark.py /usr/local/bin/
CMD ["python", "/usr/local/bin/benchmark.py"]