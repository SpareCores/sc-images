# Geekbench

## Running the Script

```shell
docker run --rm --privileged ghcr.io/sparecores/benchmark:main geekbench.sh
```

In production, the median run takes \~5.8 minutes. See
[Run Duration](./usage.md#run-duration) for how it was measured, and
[Production Setup](./usage.md#production-setup) for how production runs differ
from a plain `docker run`.

## Workload

[`geekbench.sh`](../geekbench.sh) runs Geekbench 6.7.1 once with its default
settings. Geekbench runs its own Single-Core and Multi-Core suites, each made
up of several workloads; the image configures no workload, iteration count, or
duration. The image ships both the `x86_64` build and the Arm preview build for
`aarch64`, and the script picks one by the machine architecture.

Geekbench runs with `--upload`, but the result never reaches the Geekbench
Browser. See [Upload Capture](#upload-capture) for how the script captures the
upload inside the container.

## Upload Capture

[`geekbench.sh`](../geekbench.sh) keeps Geekbench's results inside the
container. It works as follows:

1. Decrypts the Geekbench license shipped in the image, if
   `BENCHMARK_SECRETS_PASSPHRASE` is set. Without the variable, decryption
   fails, the script continues, and Geekbench runs without a license.
   Production sets the variable, so the `x86_64` build runs with the Pro
   license; the Arm preview build does not support a license. According to
   the Geekbench documentation, this makes no difference to the scores.
2. Points `browser.geekbench.com` at `127.0.0.1` in `/etc/hosts`.
3. Starts [`geekbench_capture.py`](../geekbench_capture.py), a local HTTPS
   (Hypertext Transfer Protocol Secure) server on `127.0.0.1:443`, with a
   self-signed certificate for `browser.geekbench.com` that it installs as
   trusted. The script exits with status 1 if the server does not start within
   \~10 seconds.
4. Runs Geekbench with `--upload`. The capture server saves the uploaded result
   document and answers as the Geekbench Browser would, so Geekbench reports a
   successful upload. Nothing is sent to the real Geekbench Browser.
5. Prints the captured document to stderr and stops the capture server.

The script edits `/etc/hosts`, installs a certificate, and binds port 443, so
it must run as root, the image's default user. With host networking, as in
production, port 443 on the host's loopback interface must be free.

The script's exit status is that of the `egrep` filter on Geekbench's stdout,
not Geekbench's own. To confirm that a run produced results, check stderr for
the `GEEKBENCH_UPLOAD_DOCUMENT` block.

## Output

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

## Server Range

Production runs Geekbench only on servers within two limits:

- **At most 32 vCPUs:** based on our experience running benchmarks on
  thousands of server SKUs, Geekbench does not scale beyond 32 cores.
- **At least 2.1 GiB of memory:** smaller servers run out of memory during the
  benchmark.

The image enforces neither limit, so `geekbench.sh` also runs on larger and
smaller servers.
