# frankenscube-cp-agent

The control-plane (CP) node agent of the frankenscube Kubernetes cluster. It runs on each of
the three CP nodes next to `k3s`, polls the cluster's desired-state documents in S3, and makes
sure that exactly one CP runs the cluster's [headscale](https://headscale.net) (the Tailscale
control server) at any time.

The image contains tools and scripts only. It holds no credentials, tokens, keys or account
details: the node's AWS credentials arrive as an environment file, everything else (litestream
and restore credentials, the tailnet key) in the desired-state documents at run time.

## What it does

The agent runs three independent loops: a watchdog (every second), the lease tick (every 10 s) and
the rest of the work (every 30 s):

- **Documents:** fetches and validates `desired.json` and `desired-secrets.json`. An invalid
  document is logged and ignored, the last valid one stays in force.
- **Lease:** holds one S3 object (`lease/headscale.json`) with conditional writes
  (`If-None-Match` to acquire, `If-Match` to take over or renew). The holder renews every 10 s
  and stops headscale and litestream when its last renewal is 20 s old, before anyone can take
  the lease at 40 s. The watchdog does that on a monotonic clock, independently of the other
  loops; every renewal call is bounded by the time left; the slow work of becoming the holder
  (certificate, litestream restore, start) is a background job.
- **Holder:** restores headscale from the highest `gen-*` litestream replica, replicates to the
  prefix of its own lease epoch, writes the optional `headscale.derp_map` to `derp.yaml`, serves with the certificate stored in S3 (issued once with
  `lego` through DNS-01 when none is stored or it has under 30 days left), and bootstraps an
  empty replica only under the `headscale_bootstrap` directive.
- **Every node:** heartbeats, marker uploads for `k3s-prestart.sh`, the node's own `tailscale
  up`, the `reset` directive, daily database backup, and optional health pings.

The host side of the same contract is `k3s-prestart.sh` in
[sc-infra](https://github.com/SpareCores/sc-infra).

Restore: while a `restore` directive exists, the agent resolves the newest etcd snapshot name from
the restore bucket into `restore.env`, because `k3s etcd-snapshot ls` needs a running server.

## Run

```shell
docker run --rm --name cp-agent --network=host --pid=host --privileged \
  --env-file /etc/frankenscube/node.env --env-file /etc/frankenscube/aws.env \
  -v /run/frankenscube:/run/frankenscube -v /var/lib/frankenscube:/var/lib/frankenscube \
  -v /var/lib/headscale:/var/lib/headscale -v /var/run/docker.sock:/var/run/docker.sock \
  -v /var/lib/rancher/k3s:/var/lib/rancher/k3s:ro \
  ghcr.io/sparecores/frankenscube-cp-agent:main agent
```

Other commands: `full` runs one full cycle, `lease` one lease tick.

## Contents

- Tools: `bash`, `curl`, `jq`, `aws-cli` v2 (conditional writes), `docker-cli`, `sqlite3`,
  `lego`, the headscale CLI (version and checksum pinned in the `Dockerfile`).
- `lib/shared/`: `common.sh` and `interlock.sh`, the peer-health interlock that the host's
  `k3s-prestart.sh` uses too. They are vendored from sc-infra, where they live canonically
  (`prod/cluster/cluster/ignition/scripts/lib/`). `lib/shared/SHA256SUMS` lists their hashes;
  the build and `tests/bats/shared.bats` fail when a file changes without its hash, and the
  CI of both repos compares the copies (this repo's job needs the `SC_INFRA_READ_TOKEN`
  secret, since sc-infra is private).

## Development

```shell
bats images/frankenscube-cp-agent/tests/bats
```

The tests run the agent against fakes of S3 (with ETag and conditional-write semantics),
docker, the host, headscale and lego, with a controllable clock.
