#!/bin/sh

available_memory=$(LC_ALL=C free -m | awk '/^Mem:/ {print $7}')

# on small-mem machines run the CPU suite only
# -r, -R enum enable autorun, enum values are:
# 1 - CPU Suite only
# 2 - Memory Suite only
# 3 - All Suites
if [ "$available_memory" -lt 512 ]; then
    nice -n -20 /usr/local/bin/pt_linux -r 1 1>&2; cat results_cpu.yml
else
    nice -n -20 /usr/local/bin/pt_linux -r 3 1>&2; cat results_all.yml
fi
