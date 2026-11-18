# dmidecode

This image runs `dmidecode` to read a server's SMBIOS (System Management BIOS)
tables, also called DMI (Desktop Management Interface) tables. The tables
describe a server as the cloud vendor's firmware or hypervisor reports it, such
as its firmware, system, baseboard, processors, and memory devices. The image
only installs `dmidecode` from Ubuntu 24.04, with no harness or default command
of its own. This image is not a benchmark, so nothing is timed or repeated.

This image prints the `dmidecode` text on stdout, one block per SMBIOS
structure. In production, the Inspector of
[Spare Cores Navigator](https://docs.sparecores.com/overview) also parses the
text into JSON. The Navigator takes the memory generation and speed of each
server type from that JSON, and some processor details when `lscpu` reports
none.

## Usage

To print all SMBIOS structures of a host via Docker, as production does:

```shell
docker run --rm --privileged ghcr.io/sparecores/dmidecode:main dmidecode
```

The image sets no default command, so pass `dmidecode` after the image name.
The `--privileged` option is required. To pass `dmidecode` options, such as
`-t memory`, append them after the command:

```shell
docker run --rm --privileged ghcr.io/sparecores/dmidecode:main \
  dmidecode -t memory
```

The image defines no environment variables or build arguments, and runs
`dmidecode` as root. Unlike the benchmark images, it does not run
`dmidecode` under `resource-tracker`: measuring the resource usage of a
hardware inspection that takes seconds would add nothing. The image is built
for AMD64 and Arm64.

The [`Dockerfile`](./Dockerfile) installs `dmidecode` from the Ubuntu 24.04
archive after an `apt-get upgrade`, without a pinned version, so the version
can change between image builds. All successful production runs in the
[`sc-inspector-data`](https://github.com/SpareCores/sc-inspector-data)
repository recorded version 3.5.

### Production Setup

The Navigator's Inspector runs the same command, `dmidecode` with no options,
but its runs differ from the commands above in the following ways:

- **Docker Options:** Production runs the container detached, with
  `--privileged` and host networking (`--network host`). The commands above
  use Docker's default network.
- **Timeout:** Production stops the run after 5 minutes. A plain `docker run`
  sets no time limit. The median production run took \~10 seconds, and 95% of
  runs finished within \~27 seconds.
- **Version and Parsing:** Production records the output of
  `dmidecode --version` as the version of each run, and parses the text into
  JSON, as [Output](#output) describes. A plain `docker run` prints the text
  only.
- **Servers:** Production runs `dmidecode` on every server type, except those
  on the blocklist in
  [`inspector_rules.py`](https://github.com/SpareCores/sc-crawler/blob/main/src/sc_crawler/inspector_rules.py)
  in `sc-crawler`.

The production settings are in the `dmidecode` task of
[`inspector/tasks.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/tasks.py)
in `sc-inspector`, with the default Docker options of
[`inspector/lib.py`](https://github.com/SpareCores/sc-inspector/blob/main/inspector/lib.py).

### Output

This image adds nothing to the output of `dmidecode`, and writes no files. The
text on stdout starts with a header that gives the `dmidecode` and SMBIOS
versions and the number of structures. One block per SMBIOS structure follows,
with the following lines:

1. A header line with the structure's handle, DMI type, and size in bytes,
   such as `Handle 0x0009, DMI type 17, 34 bytes`.
2. The structure's name, such as `Memory Device`.
3. The structure's fields as indented `Key: Value` lines. A field with several
   values, such as `Characteristics`, lists one value per line, indented
   further.

The content depends on what the cloud vendor's firmware or hypervisor exposes.
On AWS (Amazon Web Services), the `System Information` structure gives
`Amazon EC2` as the manufacturer and the server type, such as `m5.large`, as
the product name. On Hetzner Cloud, it gives `Hetzner` and `vServer`. The
SMBIOS version and the number of structures also vary by server type: for
example, `aws/m5.large` reports SMBIOS 2.7 with 13 structures, and
`hcloud/cx22` reports SMBIOS 2.8 with 11.

The output also includes identifiers of the individual server, such as the
`Serial Number` and `UUID` fields and, on AWS, the instance ID as the
baseboard's `Asset Tag`.

Production parses the text into a JSON array, `parsed.json`, with one object
per SMBIOS structure, using the Inspector's `parse.dmidecode` parser. The
parser is part of the Inspector, not of this image. Each object has the
following fields:

| Field | Meaning |
| ------ | ------ |
| `handle` | The `id`, `type`, and `bytes` of the structure's header line, all as strings |
| `name` | The structure's name, such as `BIOS Information` or `Memory Device` |
| `props` | The structure's fields as key-value pairs, with a list for a field with several values, and each size as a number with its unit in a separate `<key>:unit` field, such as `"ROM Size": 65536` and `"ROM Size:unit": "B"` |

### Production Results

On [`aws/m5.large`](https://sparecores.com/server/aws/m5.large)
([recorded output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m5.large/0/dmidecode/stdout)),
`dmidecode` printed the following, shortened here to the header and the
`Memory Device` structure:

```text
# dmidecode 3.5
Getting SMBIOS data from sysfs.
SMBIOS 2.7 present.
13 structures occupying 567 bytes.
Table at 0xBBE5A000.
...
Handle 0x0009, DMI type 17, 34 bytes
Memory Device
    Array Handle: 0x0008
    ...
    Size: 8 GB
    Form Factor: DIMM
    ...
    Type: DDR4
    ...
    Speed: 2666 MT/s
    ...
```

The Inspector parsed the same structure into the following object
([recorded parsed output](https://github.com/SpareCores/sc-inspector-data/blob/main/data/aws/m5.large/0/dmidecode/parsed.json)),
shortened here:

```json
{
  "handle": {
    "id": "0x0009",
    "type": "17",
    "bytes": "34"
  },
  "name": "Memory Device",
  "props": {
    "Array Handle": "0x0008",
    ...
    "Size": 8589934592,
    "Type": "DDR4",
    "Speed": 2666000000,
    ...
    "Size:unit": "B",
    "Speed:unit": "T/s"
  }
}
```

## How the Navigator Uses the Output

The Navigator reads only the following SMBIOS structures from the parsed
output to fill in the attributes of each server type:

- **Memory Device:** The memory generation comes from `Type`, and the memory
  speed from `Speed`. The memory amount comes from the memory devices only
  when the tools of the [Hardware Info](../hwinfo/README.md) image report
  none.
- **Processor Information:** The CPU speed (`Max Speed`) and the CPU
  manufacturer, family, and model (`Version`) come from this structure, each
  only when `lscpu` reports none. The Navigator discards a CPU speed of 2 GHz
  on GCP (Google Cloud Platform) as unreliable.

## References

The [references](./docs/references.md) define the terms of this image's own
tool, such as DMI type. The [glossary](https://docs.sparecores.com/glossary)
of the Spare Cores docs defines common terms, such as SMBIOS, SMBIOS structure,
and `dmidecode`.
