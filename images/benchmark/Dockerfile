FROM ubuntu:24.04 as base

ENV PATH=/venv/bin:/usr/local/bin:${PATH}
ENV VIRTUAL_ENV="/venv"
ENV LD_LIBRARY_PATH="/usr/local/lib:/usr/local/lib64"
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y python3-venv

FROM base as build_base
ENV OPENSSL_VER=3.3.0
ENV CCACHE_DIR=/ccache
ENV CCACHE_NOHASHDIR=true
ENV CCACHE_SLOPPINESS=include_file_ctime
ENV PATH=/usr/lib/ccache:${PATH}
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y build-essential curl git unzip cargo ccache && \
    mkdir /usr/src/openssl && \
    curl -sSLfo - https://github.com/openssl/openssl/releases/download/openssl-${OPENSSL_VER}/openssl-${OPENSSL_VER}.tar.gz | tar zxf - -C /usr/src/openssl --strip-components=1
FROM build_base as openssl
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=ccache,type=cache,target=/ccache,sharing=shared \
    cd /usr/src/openssl && \
      # leave out unnecessary stuff to make it compile faster
      ./Configure no-afalgeng no-async no-autoerrinit no-capieng no-cmp no-cms no-comp no-ct no-dgram no-docs no-fips-securitychecks \
        no-gost no-http no-multiblock no-nextprotoneg no-ocsp no-rfc3779 no-sock no-srp no-srtp no-ssl-trace no-tests no-quic no-ts no-ui-console \
        no-uplink no-tls no-dtls no-bf no-cast no-chacha no-cmac no-des no-dh no-dsa no-ecdh no-ecdsa no-idea no-md4 no-mdc2 no-ocb no-poly1305 \
        no-rc2 no-rc4 no-rmd160 no-scrypt no-seed no-siphash no-siv no-sm2 no-sm3 no-whirlpool && \
      make -j $(nproc) && make install_sw
FROM build_base as lmbench
RUN apt-get update --error-on=any && apt-get install -y lmbench && \
    # different path by CPU architecture
    cp $(dpkg -L lmbench | grep '\/bw_mem$') /
FROM build_base as geekbench
ENV GEEKBENCH_VER=6.3.0
RUN --mount=type=tmpfs,target=/tmp,rw \
    mkdir -p /usr/local/geekbench-x86_64 /usr/local/geekbench-aarch64 && \
    curl -sSLfo - https://cdn.geekbench.com/Geekbench-${GEEKBENCH_VER}-Linux.tar.gz | tar zxf - -C /usr/local/geekbench-x86_64 --strip-components=1 && \
    curl -sSLfo - https://cdn.geekbench.com/Geekbench-${GEEKBENCH_VER}-LinuxARMPreview.tar.gz | tar zxf - -C /usr/local/geekbench-aarch64 --strip-components=1
FROM build_base as python
RUN apt-get update \
    --mount=id=ccache,type=cache,target=/ccache,sharing=shared \
    && apt-get install -y python3-dev && python3 -m venv --without-pip /venv && \
    curl -sSLf https://bootstrap.pypa.io/get-pip.py | /venv/bin/python - && \
    /venv/bin/pip install brotli bzip3 pyzstd lz4 https://github.com/bra-fsn/pyzpaq/archive/master.zip psutil && \
    git clone https://github.com/MiloszKrajewski/SilesiaCorpus.git /tmp/silesia && \
      cd /tmp/silesia && \
      find . -name "*.zip" | xargs -n 1 unzip && rm -f *.zip *.md && \
      mkdir /usr/local/silesia && \
      cp /tmp/silesia/* /usr/local/silesia/
FROM build_base as binserve
ENV BINSERVE_VER=1e32f94
RUN --mount=type=tmpfs,target=/tmp,rw \
    cargo install --git https://github.com/mufeedvh/binserve.git --rev ${BINSERVE_VER}


FROM base as final
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt_final,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt_final,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get upgrade -y && \
    apt-get install -y colorized-logs libgomp1 wrk
ADD openssl.sh /usr/local/bin/openssl.sh
ADD bw_mem.sh /usr/local/bin/bw_mem.sh
ADD geekbench.sh /usr/local/bin/geekbench.sh
ADD compress.py /usr/local/bin/compress.py
ADD secrets.enc /secrets.enc
ADD binserve/ /usr/local/binserve/
COPY --from=geekbench /usr/local/geekbench-x86_64 /usr/local/geekbench-x86_64
COPY --from=geekbench /usr/local/geekbench-aarch64 /usr/local/geekbench-aarch64
COPY --from=openssl /usr/local /usr/local
COPY --from=lmbench /bw_mem /usr/local/bin/bw_mem
COPY --from=python /venv /venv
COPY --from=python /usr/local/silesia /usr/local/silesia
COPY --from=binserve /root/.cargo/bin/binserve /usr/local/bin/binserve
