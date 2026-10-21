# OpenSSL Speed

## Running the Script

```shell
docker run --rm --privileged ghcr.io/sparecores/benchmark:main openssl.sh
```

In production, the median run takes \~3.8 minutes. See
[Run Duration](./usage.md#run-duration) for how it was measured, and
[Production Setup](./usage.md#production-setup) for how production runs differ
from a plain `docker run`.

## Workload

[`openssl.sh`](../openssl.sh) runs `openssl speed` once for each of the
following 11 algorithms, one algorithm at a time:

- hashes: `blake2b512`, `sha256`, `sha512`, `sha3-256`, `sha3-512`,
  `shake128`, and `shake256`
- ciphers: `aes-256-cbc`, `aria-256-cbc`, `camellia-256-cbc`, and `sm4-cbc`

Each run uses the following options:

- `-multi $(nproc)`: one forked process per logical CPU, all in parallel
- `-mr`: machine-readable output
- `-mlock`: memory locking
- `-elapsed`: wall-clock timing

The script runs `openssl speed` under `nice -n -20` and sets no duration, so
`openssl speed` measures each block size for its default duration (3 seconds in
production output). `openssl speed` also picks the block sizes itself;
production output shows 16, 64, 256, 1024, 8192, and 16384 bytes.

## OpenSSL Build

The `openssl` binary is a custom OpenSSL 3.4.0 build in `/opt/openssl`,
configured with many algorithms and features disabled to shorten the build.
Only the 11 algorithms in [Workload](#workload) are measured. The script puts
the build first on `PATH`. Ubuntu's system OpenSSL stays on the image-wide
`PATH`; the Geekbench capture server uses it to create its certificate.

## Output

For each algorithm, `openssl.sh` prints a delimiter line,
`ALGO: <args> ----------------------------------------`, then the raw
`openssl speed -mr` output, including `openssl speed`'s stderr. The `+H:` line
lists the block sizes in bytes, and the `+F:` line the matching speeds in bytes
per second.

Production parses this output into a list of `algo`, `block_size`, and `speed`
entries, and [Spare Cores Navigator](https://docs.sparecores.com/overview)
publishes `speed` in bytes per second.
