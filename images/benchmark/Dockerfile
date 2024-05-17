FROM ubuntu:24.04 as base
ENV TURBOBENCH_VER = d68171773d7576c4323bb7966d01b7d4d2491b65

FROM base as build
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y build-essential curl git && \
    cd /tmp && \
      git clone --recursive https://github.com/powturbo/TurboBench.git && \
      cd /tmp/TurboBench && \
      git checkout $TURBOBENCH_VER && \
      make -C /tmp/TurboBench && \
      mkdir /usr/local/turbobench && \
      cp /tmp/TurboBench/turbobench /tmp/TurboBench/turbobench.ini /usr/local/turbobench

FROM base as final
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt_final,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt_final,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y colorized-logs
ENV PATH=${PATH}:/usr/local/bin
COPY --from=build /usr/local/turbobench /usr/local/turbobench

