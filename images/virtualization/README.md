# Virtualization

This image runs in a container and checks whether KVM (Kernel-based Virtual
Machine) hardware virtualization can be used on a server. It runs
`kvm-ok` once (from Ubuntu's `cpu-checker` package) and reports its answer as
JSON. It is a yes/no check, not a benchmark: nothing is timed or repeated, and
it does not measure virtualization performance.

This image prints one line of JSON: `{"kvm":true}` when KVM can be used, or
`kvm` set to `false` with the message of `kvm-ok` in `kvm_error`.
[Spare Cores Navigator](https://docs.sparecores.com/overview) publishes the
`kvm` flag as the `hw_virt` attribute on each server type's details page. The
Navigator does not use `kvm_error`.

## Usage

To run the check via Docker:

```bash
docker run --rm --privileged ghcr.io/sparecores/virtualization:main \
  /usr/local/bin/check_virt.sh
```

The check reports on the machine whose kernel runs the container. The check
needs `--privileged`. This image keeps the `/bin/bash` default command of
`ubuntu:24.04`, so pass the script path, `/usr/local/bin/check_virt.sh`, after
the image name. The
script reads no environment variables, options, or arguments.
[Production runs](#production-setup) take a few seconds.

This image is built for AMD64 and Arm64, and runs the check as root. Unlike
the benchmark images, it does not run the check under `resource-tracker`,
because the check uses almost no resources.

### Production Setup

The Navigator's Inspector runs the same script with `--privileged`, but its
runs differ from the command above in the following ways:

- **Docker Options:** Production runs the container detached and with host
  networking (`--network host`). The command above uses Docker's default
  network.
- **Timeout:** Production stops the check after 5 minutes. A plain
  `docker run` sets no time limit.
- **Servers:** Production runs the check on every server type, except those
  on the blocklist in
  [`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
  in `sc-crawler`.

The production settings are in the `virtualization` task of
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
in `sc-inspector`, with the default Docker options of
[`inspector/lib.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/lib.py).

### Output

This image prints one line of JSON on stdout and writes no files. The JSON has
the following fields:

| Field | Type | When | Meaning |
| ------ | ------ | ------ | ------ |
| `kvm` | boolean | always | `true` if `kvm-ok` exited 0, otherwise `false` |
| `kvm_error` | string | only when `kvm` is `false` | The stdout and stderr of `kvm-ok` together on one line, or `kvm-ok command not found` |

The script exits 0 whenever `kvm-ok` ran, whether KVM can be used or not. It
exits 1 only when `kvm-ok` is missing from this image, which means this image
itself is broken. Read `kvm`, not the exit status, to tell whether KVM can be
used.

To fit the output of `kvm-ok` on one line, the script replaces each newline
with a space and escapes double quotes. A blank line in the output therefore
leaves a run of spaces, and a failure without output leaves `kvm_error` empty.
The script escapes no other characters, such as backslashes or tabs. An
unescaped backslash or control character would make the line invalid JSON;
none of the production outputs contains one.

### Production Results

Since January 2025, production runs in the
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data)
repository have returned three distinct outputs.

KVM can be used on server types such as
[`gcp/n2-standard-2`](https://sparecores.com/server/gcp/n2-standard-2)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/gcp/n2-standard-2/0/virtualization/stdout)):

```json
{"kvm":true}
```

On server types such as
[`aws/m7i.large`](https://sparecores.com/server/aws/m7i.large)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7i.large/0/virtualization/stdout))
and [`hcloud/cx22`](https://sparecores.com/server/hcloud/cx22)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/hcloud/cx22/0/virtualization/stdout)),
the `kvm-ok` command reported `Your CPU does not support KVM extensions`:

```json
{"kvm":false,"kvm_error":"INFO: Your CPU does not support KVM extensions KVM acceleration can NOT be used"}
```

The `kvm-ok` command prints this message when the CPU flags in `/proc/cpuinfo`
include neither `vmx` nor `svm`, a check it makes only on AMD64.

On Arm64 server types such as
[`aws/c7g.large`](https://sparecores.com/server/aws/c7g.large)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/virtualization/stdout)),
the `kvm-ok` command reported that `/dev/kvm` does not exist:

```json
{"kvm":false,"kvm_error":"INFO: /dev/kvm does not exist HINT:   sudo modprobe kvm /usr/sbin/kvm-ok: 93: modprobe: not found"}
```

In this third message, `kvm-ok` suggests loading the `kvm` kernel module with
`modprobe`. It then runs `modprobe` itself, not to load `kvm` but to load the
`msr` module, which reads the CPU's model-specific registers. This image does
not include `modprobe`, so `kvm-ok` stops with `modprobe: not found`.

## How the Check Works

The [`check_virt.sh`](./check_virt.sh) script takes the following steps:

1. It checks that `kvm-ok` is installed. If it is not, the script prints
   `{"kvm":false,"kvm_error":"kvm-ok command not found"}` and exits 1.
2. It runs `kvm-ok` once and captures its stdout and stderr together.
3. It prints `{"kvm":true}` if `kvm-ok` exited 0, or `kvm` set to `false`
   with the output of `kvm-ok` in `kvm_error` otherwise.

The result depends on the server and on whether `/dev/kvm` exists inside the
container. On AMD64, `kvm-ok` checks that the CPU flags in `/proc/cpuinfo`
include `vmx` (Intel) or `svm` (AMD), then that `/dev/kvm` exists. On Arm64,
it checks only that `/dev/kvm` exists.

The [`Dockerfile`](./Dockerfile) installs `cpu-checker` on `ubuntu:24.04`
without a pinned version, so the `kvm-ok` version can change between image
builds. As of November 2026, Ubuntu 24.04 ships `cpu-checker`
0.7-1.3build2. Production records no tool version for this check.

## References

The [references](./docs/references.md) define `kvm-ok`, the tool of this
image. The [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs defines common terms, such as KVM, `/dev/kvm`, and
`modprobe`.
