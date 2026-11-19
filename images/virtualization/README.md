# Virtualization

This image runs in a container and checks whether KVM (Kernel-based Virtual
Machine) hardware virtualization can be used on a server. It runs
`kvm-ok` once (from Ubuntu's `cpu-checker` package) and reports its answer as
JSON. It is a yes/no check, not a benchmark: nothing is timed or repeated, and
it does not measure virtualization performance.

This image prints one line of JSON. When KVM can be used, the line is
`{"kvm":true}`. When it cannot, `kvm` is `false` and `kvm_error` holds what
`kvm-ok` printed. [Spare Cores Navigator](https://docs.sparecores.com/overview) publishes the
`kvm` flag as its hardware virtualization attribute, `hw_virt`, on each server
type's details page. The
Navigator does not use `kvm_error`.

## Usage

To run the check via Docker:

```bash
docker run --rm --privileged ghcr.io/sparecores/virtualization:main \
  /usr/local/bin/check_virt.sh
```

The check reports on the machine whose kernel runs the container. The check
needs `--privileged`. This image keeps the `/bin/bash` default command of
`ubuntu:24.04`, so pass the script path,
[`/usr/local/bin/check_virt.sh`](./check_virt.sh), after the image name. The
script reads no environment variables, options, or arguments.
[Production runs](#production-setup) take a few seconds.

This image is built for AMD64 and Arm64, and runs the check as root. Unlike
the [benchmark images](../../README.md#benchmarks), it
does not run the check under `resource-tracker`, because the check uses almost
no resources.

### Production Setup

The Navigator's [Inspector](https://github.com/SpareCores/sc-inspector), which
runs the images on each server type, runs the same script with
`--privileged`, but its
runs differ from the command above in the following ways:

- **Docker Options:** Production runs the container detached and with host
  networking (`--network host`). The command above uses Docker's default
  network.
- **Timeout:** Production stops the check after 5 minutes. A plain
  `docker run` sets no time limit.
- **Servers:** Production runs the check on every server type, except those
  on the blocklist in
  [`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
  in [`sc-crawler`](https://github.com/SpareCores/sc-crawler).

The production settings are in the `virtualization` task of
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
in [`sc-inspector`](https://github.com/SpareCores/sc-inspector), with the
default Docker options of
[`inspector/lib.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/lib.py).

### Output

This image prints one line of JSON on stdout and writes no files. The JSON has
the following fields:

| Field | Type | When | Meaning |
| ------ | ------ | ------ | ------ |
| `kvm` | boolean | always | `true` if KVM can be used (`kvm-ok` exited with code 0), otherwise `false` |
| `kvm_error` | string | only when `kvm` is `false` | Everything `kvm-ok` printed to stdout and stderr, joined into one line. If `kvm-ok` is not installed, it reads `kvm-ok command not found`. |

To tell whether KVM can be used, read `kvm`, not the script's exit code. The
script exits with code 0 whenever `kvm-ok` ran, whether KVM can be used or
not, and with code 1 only when `kvm-ok` is not installed, which means this
image is broken.

To fit the output of `kvm-ok` on one line, the script replaces each newline
with a space and escapes double quotes. A blank line in the output therefore
leaves a run of spaces, and a failure without output leaves `kvm_error` empty.
The script escapes no other characters, such as backslashes or tabs. An
unescaped backslash or control character would make the line invalid JSON;
none of the production outputs contains one.

### Production Results

Since production started running this check in January 2025, the runs in the
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data)
repository have returned only the following three outputs. Each comes with an
example server type and its recorded output.

**KVM Can Be Used:** Server types such as
[`gcp/n2-standard-2`](https://sparecores.com/server/gcp/n2-standard-2)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/gcp/n2-standard-2/0/virtualization/stdout)):

```json
{"kvm":true}
```

**No CPU Virtualization Flags:** Server types such as
[`aws/m7i.large`](https://sparecores.com/server/aws/m7i.large)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m7i.large/0/virtualization/stdout))
and [`hcloud/cx22`](https://sparecores.com/server/hcloud/cx22)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/hcloud/cx22/0/virtualization/stdout)):

```json
{"kvm":false,"kvm_error":"INFO: Your CPU does not support KVM extensions KVM acceleration can NOT be used"}
```

On AMD64, `kvm-ok` prints this message when the CPU flags in `/proc/cpuinfo`
include neither `vmx` nor `svm`.

**No `/dev/kvm`:** Arm64 server types such as
[`aws/c7g.large`](https://sparecores.com/server/aws/c7g.large)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/c7g.large/0/virtualization/stdout)):

```json
{"kvm":false,"kvm_error":"INFO: /dev/kvm does not exist HINT:   sudo modprobe kvm /usr/sbin/kvm-ok: 93: modprobe: not found"}
```

Here, `kvm-ok` suggests loading the `kvm` kernel module with `modprobe`. It
then runs `modprobe` itself, not to load `kvm` but to load the `msr` module,
which reads the CPU's model-specific registers. This image does not include
`modprobe`, so `kvm-ok` stops with `modprobe: not found`.

## How the Check Works

The [`check_virt.sh`](./check_virt.sh) script takes the following steps:

1. It checks that `kvm-ok` is installed. If it is not, the script prints
   `{"kvm":false,"kvm_error":"kvm-ok command not found"}` and exits with
   code 1.
2. It runs `kvm-ok` once and keeps everything it prints to stdout and
   stderr in a shell variable, for `kvm_error`.
3. If `kvm-ok` exited with code 0, it prints `{"kvm":true}`. Otherwise, it
   prints `kvm` as `false` and that output, joined into one line, as
   `kvm_error`.

The result depends on the server and on whether `/dev/kvm` exists inside the
container. On AMD64, `kvm-ok` checks that the CPU flags in `/proc/cpuinfo`
include `vmx` (Intel) or `svm` (AMD), then that `/dev/kvm` exists. On Arm64,
it checks only that `/dev/kvm` exists.

The [`Dockerfile`](./Dockerfile) installs `cpu-checker` on `ubuntu:24.04`
without a pinned version, so the `kvm-ok` version can change between image
builds. Ubuntu 24.04 ships `cpu-checker` 0.7-1.3build2, the version of its
April 2024 release. Production records no tool version for this check.

## References

This image's [Glossary](./docs/references.md) defines its tool, `kvm-ok`.
The [Spare Cores Glossary](https://docs.sparecores.com/glossary) defines
common terms, such as KVM, `/dev/kvm`, and `modprobe`.
