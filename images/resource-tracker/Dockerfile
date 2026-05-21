FROM alpine:3.21 AS fetch
ARG RESOURCE_TRACKER_VERSION=0.1.6
RUN apk add --no-cache curl tar
RUN ARCH="$(uname -m | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')" && \
    URL="https://github.com/SpareCores/resource-tracker-rs/releases/download/${RESOURCE_TRACKER_VERSION}/resource-tracker-${RESOURCE_TRACKER_VERSION}-linux-${ARCH}.tar.gz" && \
    curl -fsSL "$URL" | tar -xvzf - resource-tracker && \
    install -m 0755 resource-tracker /resource-tracker

FROM scratch
COPY --from=fetch /resource-tracker /usr/local/bin/resource-tracker
