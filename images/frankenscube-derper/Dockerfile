# DERP relay of the frankenscube cluster: tailscale's derper, built from source at a pinned
# version. Nothing secret is baked in: the hostname, certificate directory and client
# verification are runtime flags.
FROM golang:1.27-alpine@sha256:70bd7ddea8b6cbe5d6943cde279d6b67feb10cc924fb9100d1d8757c1575b280 AS build
ARG TAILSCALE_VERSION=1.104.7
ENV CGO_ENABLED=0 GOFLAGS=-trimpath
RUN go install -ldflags="-s -w" "tailscale.com/cmd/derper@v${TAILSCALE_VERSION}"

FROM gcr.io/distroless/static-debian12@sha256:d75cdd72874d4790092fcb1b058493ecf6bb5bf2b2b897045b00ff01d91843f2
COPY --from=build /go/bin/derper /usr/local/bin/derper
ENTRYPOINT ["/usr/local/bin/derper"]
