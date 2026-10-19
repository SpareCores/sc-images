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
scripts. See [Workloads](./workloads.md) for what each one runs.

The entrypoint runs every command under Resource Tracker (`resource-tracker`),
which samples the container's resource usage. Its metrics are not published.

## Geekbench

[`geekbench.sh`](../geekbench.sh) keeps Geekbench's results inside the
container. It works as follows:

1. Decrypts the Geekbench license shipped in the image, if
   `BENCHMARK_SECRETS_PASSPHRASE` is set. Without the variable, decryption
   fails, the script continues, and Geekbench runs without a license.
2. Points `browser.geekbench.com` at `127.0.0.1` in `/etc/hosts`.
3. Starts [`geekbench_capture.py`](../geekbench_capture.py), a local HTTPS
   (Hypertext Transfer Protocol Secure) server on `127.0.0.1:443`, with a
   self-signed certificate for `browser.geekbench.com` that it installs as
   trusted. The script exits with status 1 if the server does not start within
   about 10 seconds.
4. Runs Geekbench with `--upload`. The capture server saves the uploaded result
   document and answers as the Geekbench Browser would, so Geekbench reports a
   successful upload. Nothing is sent to the real Geekbench Browser.
5. Prints the captured document to stderr and stops the capture server.

The script edits `/etc/hosts`, installs a certificate, and binds port 443, so
it must run as root, the image's default user. With host networking, as in
production, port 443 on the host's loopback interface must be free.

## Outputs

### `openssl.sh`

For each algorithm, `openssl.sh` prints a delimiter line,
`ALGO: <args> ----------------------------------------`, then the raw
`openssl speed -mr` output, including `openssl speed`'s stderr. The `+H:` line
lists the block sizes in bytes, and the `+F:` line the matching speeds in bytes
per second.

Production parses this output into a list of `algo`, `block_size`, and `speed`
entries, and the Navigator publishes `speed` in bytes per second.

### `bw_mem.sh`

`bw_mem.sh` prints one line per successful run, `<op> <size> <bandwidth>`,
such as `rd 0.016384 147885.61`:

- `<op>` is `rd`, `wr`, or `rdwr`.
- `<size>` is the per-process buffer size in MB of 10^6 bytes, so `16k`
  prints as `0.016384`.
- `<bandwidth>` is the total MB/s across all processes.

A size that fails or prints unexpected output is skipped without an error, so
a run can have fewer than 33 lines.

### `compress.py`

`compress.py` prints one indented JSON object,
`{<compressor>: {<level>: [<result>, ...]}}`. Levels are string keys, and
`bzip3`, which has no level, uses the key `"null"`. Each result has the
following fields:

- `threads`: the number of parallel processes.
- `extra_args`: the extra arguments, such as `block_size` in bytes for `bzip3`
  and `lz4`.
- `ratio`: compressed size divided by original size, as a percentage; lower is
  better.
- `compress` and `decompress`: bytes of uncompressed data per second, summed
  over all processes.

A measurement that fails, for example because its worker process was killed,
has `ratio`, `compress`, and `decompress` set to `null`, and its traceback goes
to stderr. Production runs on small servers show such failures.

### `geekbench.sh`

`geekbench.sh` writes the following:

- **stdout:** Geekbench's console report: system information, then the
  Single-Core and Multi-Core tables of each workload's score and rate. It ends
  with an upload notice and the placeholder URL
  `https://browser.geekbench.com/v6/cpu/99999999`, which does not lead to a
  result.
- **stderr:** Geekbench's own stderr, then the captured result document as
  JSON between a `GEEKBENCH_UPLOAD_DOCUMENT` and a
  `GEEKBENCH_UPLOAD_DOCUMENT_END` line. The document holds each workload's
  runtime, rate, and score, and system metrics.

The capture server logs to `/tmp/geekbench-capture.log` inside the container;
the script prints the log only if the server fails to start.

Production builds a `results.json` from the captured document, with a `Score`
and per-workload scores under `Single-Core Performance` and
`Multi-Core Performance`.

## Production Setup

The Inspector of Spare Cores Navigator
([`sc-inspector`](https://github.com/SpareCores/sc-inspector)) runs each script
as a separate task. A plain `docker run` differs from it as noted below.

### Container Settings

All four tasks run privileged with host networking, and each has a 1-hour
timeout. Production runs `compress.py` and `geekbench.sh` under
`nice -n -20` from its own command line; `openssl.sh` and `bw_mem.sh` set it
themselves. A plain `docker run` without `--privileged` cannot raise the
priority.

The `geekbench` task also sets the following:

- a memory limit, including swap, of 85% of available memory
- swappiness 0
- a minimum of 2.1 GB of memory and a maximum of 32 vCPUs

### Server Coverage

Production runs every task on every server type except the following:

- `geekbench`: servers with more than 32 vCPUs or less than 2.1 GB of memory
- `compression_text` (`compress.py`): servers with less than 1 GB of memory
- `bw_mem`: the OVH `a10-180` and `l4-360` server types, where it crashes or
  hangs

The image enforces none of these limits.

### Run Duration

The median production run, across about 3,400 to 3,800 runs per task, takes
the following time:

| Script | Production task | Median duration |
| --- | --- | --- |
| `openssl.sh` | `openssl` | \~230 s |
| `bw_mem.sh` | `bw_mem` | \~240 s |
| `geekbench.sh` | `geekbench` | \~350 s |
| `compress.py` | `compression_text` | \~640 s |

## Environment Variables

| Variable | Meaning | Default |
| --- | --- | --- |
| `BENCHMARK_SECRETS_PASSPHRASE` | Passphrase that decrypts the Geekbench license shipped in the image. Optional: without it, Geekbench runs without a license. | Unset |
| `TRACKER_QUIET` | Quiet mode for Resource Tracker. | `true` |
