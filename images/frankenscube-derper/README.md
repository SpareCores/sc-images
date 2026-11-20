# frankenscube-derper

[Tailscale's DERP relay](https://tailscale.com/kb/1232/derp-servers) (`derper`), built from
source at a pinned tailscale version, for the DERP regions of the frankenscube cluster's
headscale. The image holds no configuration or secrets: the hostname, ports and certificate
directory are run-time flags.

## Run

```shell
docker run --rm --name derper --network=host -v /var/lib/derper:/var/lib/derper \
  -v /var/run/tailscale:/var/run/tailscale \
  ghcr.io/sparecores/frankenscube-derper:main \
  -hostname derp1.example.com -a :8443 -http-port 80 -stun-port 3478 \
  -certmode letsencrypt -certdir /var/lib/derper -verify-clients
```

`-verify-clients` makes the relay accept only nodes of the tailnet of the host's local
`tailscaled`, which it asks through the socket in `/var/run/tailscale`.

Upstream derper serves TLS only on `-a :443` or with `-certmode manual`; with
`-certmode letsencrypt` on any other port it serves plain HTTP, which no DERP client accepts.
The `Dockerfile` patches derper to always serve TLS, so `-a :8443` with Let's Encrypt works
(certificates through HTTP-01 on `-http-port`, cached in `-certdir`).

The tailscale version is the `TAILSCALE_VERSION` build argument in the `Dockerfile`.
`tests/smoke.sh <image>` checks that the binary starts and serves TLS on `:8443` with
`-certmode letsencrypt`.
