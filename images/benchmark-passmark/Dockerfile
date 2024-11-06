FROM ubuntu:24.04
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y curl unzip sudo dmidecode

ARG INTEL_URL="https://www.passmark.com/downloads/pt_linux_x64.zip"
ARG ARM64_URL="https://www.passmark.com/downloads/pt_linux_arm64.zip"
ARG LIBTINFO_INTEL="http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libtinfo5_6.4-2_amd64.deb"
ARG LIBNCURSES_INTEL="http://archive.ubuntu.com/ubuntu/pool/universe/n/ncurses/libncurses5_6.4-2_amd64.deb"
ARG LIBTINFO_ARM="http://ports.ubuntu.com/pool/universe/n/ncurses/libtinfo5_6.2-0ubuntu2.1_arm64.deb"
ARG LIBNCURSES_ARM="http://ports.ubuntu.com/pool/universe/n/ncurses/libncurses5_6.2-0ubuntu2.1_arm64.deb"

RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        DOWNLOAD_URL=$ARM64_URL && \
        LIBTINFO_URL=$LIBTINFO_ARM && \
        LIBNCURSES_URL=$LIBNCURSES_ARM; \
    else \
        DOWNLOAD_URL=$INTEL_URL && \
        LIBTINFO_URL=$LIBTINFO_INTEL && \
        LIBNCURSES_URL=$LIBNCURSES_INTEL; \
    fi && \
    curl -L "$DOWNLOAD_URL" -o pt_linux.zip && \
    unzip pt_linux.zip -d /usr/local && \
    rm pt_linux.zip && \
    if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        mv /usr/local/PerformanceTest/pt_linux_arm64 /usr/local/bin/pt_linux; \
    else \
        mv /usr/local/PerformanceTest/pt_linux_x64 /usr/local/bin/pt_linux; \
    fi && \
    # Download and install ncurses5 libraries
    curl -L "$LIBTINFO_URL" -o libtinfo5.deb && \
    curl -L "$LIBNCURSES_URL" -o libncurses5.deb && \
    dpkg -i libtinfo5.deb libncurses5.deb && \
    rm libtinfo5.deb libncurses5.deb

RUN chmod +x /usr/local/bin/pt_linux

# PassMark expects a terminal
ENV TERM=xterm
CMD ["bash", "-c", "nice -n -20 /usr/local/bin/pt_linux -r 3 1>&2; cat results_all.yml"]
