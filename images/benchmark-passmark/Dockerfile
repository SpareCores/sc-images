FROM ubuntu:24.04
RUN --mount=type=tmpfs,target=/tmp,rw \
    --mount=id=var_cache_apt,type=cache,target=/var/cache/apt,sharing=locked \
    --mount=id=var_lib_apt,type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update --error-on=any && \
    apt-get install -y curl unzip sudo dmidecode

# Use local files, as they might be deleted from the mirrors
COPY pkgs/libtinfo5_6.4-2_amd64.deb pkgs/libncurses5_6.4-2_amd64.deb /tmp/amd64/
COPY pkgs/libtinfo5_6.2-0ubuntu2.1_arm64.deb pkgs/libncurses5_6.2-0ubuntu2.1_arm64.deb /tmp/arm64/

ARG INTEL_URL="https://www.passmark.com/downloads/pt_linux_x64.zip"
ARG ARM64_URL="https://www.passmark.com/downloads/pt_linux_arm64.zip"

RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        DOWNLOAD_URL=$ARM64_URL && \
        LIBTINFO_DEB="/tmp/arm64/libtinfo5_6.2-0ubuntu2.1_arm64.deb" && \
        LIBNCURSES_DEB="/tmp/arm64/libncurses5_6.2-0ubuntu2.1_arm64.deb"; \
    else \
        DOWNLOAD_URL=$INTEL_URL && \
        LIBTINFO_DEB="/tmp/amd64/libtinfo5_6.4-2_amd64.deb" && \
        LIBNCURSES_DEB="/tmp/amd64/libncurses5_6.4-2_amd64.deb"; \
    fi && \
    curl -L "$DOWNLOAD_URL" -o pt_linux.zip && \
    unzip pt_linux.zip -d /usr/local && \
    rm pt_linux.zip && \
    if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        mv /usr/local/PerformanceTest/pt_linux_arm64 /usr/local/bin/pt_linux; \
    else \
        mv /usr/local/PerformanceTest/pt_linux_x64 /usr/local/bin/pt_linux; \
    fi && \
    # Install local ncurses5 libraries
    dpkg -i "$LIBTINFO_DEB" "$LIBNCURSES_DEB" && \
    rm "$LIBTINFO_DEB" "$LIBNCURSES_DEB"

RUN chmod +x /usr/local/bin/pt_linux

ADD passmark.sh /usr/local/bin/passmark.sh
# PassMark expects a terminal
ENV TERM=xterm
CMD /usr/local/bin/passmark.sh
