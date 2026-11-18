# frankenscube-derper

[Tailscale's DERP relay](https://tailscale.com/kb/1232/derp-servers) (`derper`), built from
source at a pinned tailscale version, for the DERP regions of the frankenscube cluster's
headscale. The image holds no configuration or secrets: the hostname, ports and certificate
directory are run-time flags.

## Run

```shell
docker run --rm --name derper --network=host -v /var/lib/derper:/var/lib/derper \
  ghcr.io/sparecores/frankenscube-derper:main \
  -hostname derp1.example.com -a :8443 -http-port 80 -stun-port 3478 \
  -certmode letsencrypt -certdir /var/lib/derper -verify-clients
```

`-verify-clients` makes the relay accept only nodes of the tailnet of the host's local
`tailscaled`.

The tailscale version is the `TAILSCALE_VERSION` build argument in the `Dockerfile`.
`tests/smoke.sh <image>` checks that the binary starts.
