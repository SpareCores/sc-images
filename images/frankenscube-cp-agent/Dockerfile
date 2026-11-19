# Control-plane node agent of the frankenscube cluster. The image holds tools and scripts only:
# credentials, tokens and account details arrive at run time (env files, the desired docs in S3).
FROM alpine:3.22@sha256:49340b346798875b5592e5d420241f2f9cfba5f47e62ebb1446a1b34734e898f

# Tool versions: the alpine release is pinned by digest above, the apk packages to their minor
# version (alpine drops superseded patch builds from the release, so an exact pin would stop
# building). The checks below fail the build when a tool is not what the agent needs.
ARG HEADSCALE_VERSION=0.29.7
ARG HEADSCALE_SHA256=e827fc9b12d958bc95b081c4401dd63f8e0471d025b45b508535a920532150a8
RUN apk add --no-cache \
      'bash~=5.2' 'coreutils~=9.7' 'curl~=8.14' 'jq~=1.8' 'openssl~=3.5' 'util-linux~=2.41' \
      'iputils~=20240905' 'sqlite~=3.49' 'aws-cli~=2.27' 'docker-cli~=28.3' 'lego~=4.23' && \
    curl -fsSL -o /usr/local/bin/headscale \
      "https://github.com/juanfont/headscale/releases/download/v${HEADSCALE_VERSION}/headscale_${HEADSCALE_VERSION}_linux_amd64" && \
    echo "${HEADSCALE_SHA256}  /usr/local/bin/headscale" | sha256sum -c - && \
    chmod 0755 /usr/local/bin/headscale && \
    # conditional writes (lease): put-object must know If-Match and If-None-Match
    aws s3api put-object --generate-cli-skeleton | grep -q '"IfMatch"' && \
    aws s3api put-object --generate-cli-skeleton | grep -q '"IfNoneMatch"' && \
    headscale version | grep -q "${HEADSCALE_VERSION}"

COPY bin /opt/cp-agent/bin
COPY lib /opt/cp-agent/lib
# lib/shared is vendored from sc-infra (the host's k3s-prestart.sh uses the same files): see README.md
RUN cd /opt/cp-agent/lib/shared && sha256sum -c SHA256SUMS && \
    bash -n /opt/cp-agent/bin/cp-agent && \
    for f in /opt/cp-agent/lib/*.sh; do bash -n "$f"; done

ENTRYPOINT ["/opt/cp-agent/bin/cp-agent"]
CMD ["agent"]
