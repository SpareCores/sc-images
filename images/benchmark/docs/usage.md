# Usage

## Running a Script

The image has no default command. Pass one script as the command; the
scripts are in `/usr/local/bin`, which is on `PATH`:

```shell
docker run --rm --privileged ghcr.io/sparecores/benchmark:main openssl.sh
docker run --rm --privileged ghcr.io/sparecores/benchmark:main bw_mem.sh
docker run --rm --privileged ghcr.io/sparecores/benchmark:main python /usr/local/bin/compress.py
docker run --rm --privileged ghcr.io/sparecores/benchmark:main geekbench.sh
```

The scripts take no arguments; all benchmark settings are fixed in the
scripts. Each script has its own page, with what it runs and what it prints:

- [OpenSSL Speed](./openssl.md): `openssl.sh`
- [Memory Bandwidth](./bw-mem.md): `bw_mem.sh`
- [Compression](./compression.md): `compress.py`
- [Geekbench](./geekbench.md): `geekbench.sh`

The entrypoint runs every command under Resource Tracker (`resource-tracker`),
which samples the container's resource usage. Its metrics are not published.

## Production Setup

The Inspector of Spare Cores Navigator
([`sc-inspector`](https://github.com/SpareCores/sc-inspector)) runs each script
as a separate task. A plain `docker run` differs from it as noted below.

### Container Settings

All four tasks run privileged with host networking, and each has a 1 hour
timeout. Production runs `compress.py` and `geekbench.sh` under
`nice -n -20` from its own command line; `openssl.sh` and `bw_mem.sh` set it
themselves. A plain `docker run` without `--privileged` cannot raise the
priority.

The `geekbench` task also sets the following:

- a memory limit, including swap, of 85% of available memory
- swappiness 0

### Server Coverage

Production runs every task on every server type except the following:

- `geekbench`: servers with more than 32 vCPUs or less than 2.1 GiB of memory
- `compression_text` (`compress.py`): servers with less than 1 GiB of memory
- `bw_mem`: the OVH `a10-180` and `l4-360` server types, where it crashes or
  hangs

The image enforces none of these limits.

### Run Duration

The median production run, across \~3400-3800 runs per task, takes
the following time:

| Script | Production task | Median duration (minutes) |
| --- | --- | --- |
| `openssl.sh` | `openssl` | \~3.8 |
| `bw_mem.sh` | `bw_mem` | \~4.0 |
| `geekbench.sh` | `geekbench` | \~5.8 |
| `compress.py` | `compression_text` | \~10.7 |

## Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `BENCHMARK_SECRETS_PASSPHRASE` | Passphrase that decrypts the Geekbench license shipped in the image. Optional: without it, Geekbench runs without a license. | Unset |
| `TRACKER_QUIET` | Quiet mode for Resource Tracker. | `true` |
