# DERP relay of the frankenscube cluster: tailscale's derper, built from source at a pinned
# version. Nothing secret is baked in: the hostname, certificate directory and client
# verification are runtime flags.
#
# Patch: upstream derper serves TLS only when -a is :443 or -certmode is manual
# (cmd/derper/derper.go, serveTLS); with -certmode letsencrypt on any other port it serves
# plain HTTP. The cluster's derpers listen on :8443 (headscale holds :443) with Let's Encrypt
# (HTTP-01 on -http-port), so the patch makes derper always serve TLS. The build fails if the
# upstream line changes.
FROM golang:1.27-alpine@sha256:70bd7ddea8b6cbe5d6943cde279d6b67feb10cc924fb9100d1d8757c1575b280 AS build
ARG TAILSCALE_VERSION=1.104.7
ENV CGO_ENABLED=0 GOFLAGS=-trimpath
WORKDIR /src
RUN go mod download "tailscale.com@v${TAILSCALE_VERSION}" \
 && cp -r "$(go env GOMODCACHE)/tailscale.com@v${TAILSCALE_VERSION}/." . && chmod -R u+w . \
 && grep -q 'serveTLS := tsweb.IsProd443(\*addr) || \*certMode == "manual"$' cmd/derper/derper.go \
 && sed -i 's/serveTLS := tsweb.IsProd443(\*addr) || \*certMode == "manual"$/serveTLS := true/' cmd/derper/derper.go \
 && go build -ldflags="-s -w" -o /go/bin/derper ./cmd/derper

FROM gcr.io/distroless/static-debian12@sha256:d75cdd72874d4790092fcb1b058493ecf6bb5bf2b2b897045b00ff01d91843f2
COPY --from=build /go/bin/derper /usr/local/bin/derper
ENTRYPOINT ["/usr/local/bin/derper"]
