# Usage

## Running the Image

The image's entrypoint runs `sh` under `resource-tracker`, and has no default
command. Pass the command to run with `-c`:

```bash
docker run --rm ghcr.io/sparecores/stress-ng:main \
  -c 'python3 /usr/local/bin/run_stressng_benchmarks.py' > stress-ng.json
```

The stressor suite needs no environment variables. For a quick check, the
following command runs the first five stressors for 2 s each:

```bash
docker run --rm -e STRESSNG_QUICK=1 -e STRESSNG_LIMIT=5 \
  ghcr.io/sparecores/stress-ng:main \
  -c 'python3 /usr/local/bin/run_stressng_benchmarks.py'
```

The image includes `stress-ng` itself, so `-c` also accepts any `stress-ng`
command, such as `-c 'stress-ng --cpu 1 -t 10 --metrics'`.

## Production Setup

The Navigator's Inspector runs this image in two production tasks, each with
`--privileged` and `--network host`, and with the command under
`nice -n -20`. `nice -n -20` needs `--privileged`; a plain run, without
`nice`, runs at normal priority. The image itself does not set a priority.

The stressor suite runs as follows:

```bash
docker run --rm --privileged --network host ghcr.io/sparecores/stress-ng:main \
  -c 'nice -n -20 python3 /usr/local/bin/run_stressng_benchmarks.py'
```

The CPU scaling loop runs the `cpu` stressor at each worker count that
[`count.awk`](../count.awk) prints for the CPU count:

```bash
docker run --rm --privileged --network host ghcr.io/sparecores/stress-ng:main \
  -c 'for ncpu in $(awk -f /usr/local/bin/count.awk $(nproc)); do echo -n "$ncpu,"; nice -n -20 stress-ng --metrics --cpu $ncpu --cpu-method div16 -t 10 | egrep "metrc.*cpu" | awk "{print \$9}"; done'
```

Both tasks run on every server type that the Inspector benchmarks.

## Run Duration

The stressor suite took the following times in production:

- about 7 minutes at the shortest, on a server with 1 vCPU
- about 15 minutes at the median
- about 66 minutes at the longest

The CPU scaling loop took the following times in production with
`stress-ng` 0.20.01:

- about 19 s at the shortest
- about 2.8 minutes at the median
- about 50 minutes at the longest

## Output

### Stressor Suite

The stressor suite prints nothing while it runs: it captures the output of
each `stress-ng` run and prints one JSON document to stdout at the end,
indented with two spaces. It has the following fields:

| Field | Type | Meaning |
| ----- | ---- | ------- |
| `stressors.<name>."1"` | object or `null` | Result with one worker |
| `stressors.<name>."<N>"` | object or `null` | Result with N workers, where N is `meta.physical_cpus`; present only when N is greater than 1 |
| `stressors.<name>.<workers>.bogo_ops_per_sec` | float | Throughput in stress-ng bogo ops/s (bogus operations per second), in real time, the total across all workers |
| `categories.<category>` | list of strings | The stressors of each category that `stress-ng --stressors` lists, in category order |
| `meta.timeout_sec_per_stressor` | integer | Seconds per stressor run: `5`, or `2` with `STRESSNG_QUICK=1`; does not reflect the 15 s of `stream` |
| `meta.physical_cpus` | integer | Worker count of the multi-worker runs: the number of logical CPUs available to the container |
| `meta.stressor_count` | integer | Number of stressors run, after `STRESSNG_LIMIT` |

`categories` is not shortened by `STRESSNG_LIMIT`, so it can list stressors
that did not run.

The following is an excerpt from a production run on a server with 2 vCPUs:

```json
"atomic": {"1": {"bogo_ops_per_sec": 869.33}, "2": {"bogo_ops_per_sec": 868.79}}
```

The value is the bogo ops/s (real time) column of the stress-ng metrc line
that `stress-ng --metrics-brief` prints for each stress-ng stressor. See
[`data/hcloud/ccx13/0/stressng_benchmarks/stdout`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/hcloud/ccx13/0/stressng_benchmarks/stdout)
for a full document.

The script exits with code 0 even when some stressors fail. If `stress-ng`
cannot list its stressors, it prints `{"error": "could not get stressor list"}`
to stderr and exits with code 1.

### CPU Scaling Loop

The CPU scaling loop prints one `<workers>,<bogo ops/s>` line per worker count,
for example:

```text
1,3059.78
2,3062.79
```

See
[`data/hcloud/ccx13/0/stressngfull/stdout`](https://github.com/SpareCores/sc-inspector-data/blob/main/data/hcloud/ccx13/0/stressngfull/stdout)
for a production example.

## Environment Variables

| Variable | Meaning | Default |
| -------- | ------- | ------- |
| `STRESSNG_QUICK` | `1` runs each stressor for 2 s instead of 5 s; `stream` still runs for 15 s. Any other value keeps 5 s. | unset |
| `STRESSNG_LIMIT` | Runs only the first N stressors in alphabetical order. A value that is not an integer is ignored. | unset (all stressors) |
| `TRACKER_QUIET` | Set in the image to quiet `resource-tracker`. | `true` |
