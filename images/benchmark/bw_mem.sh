#!/bin/sh

for what in rd wr rdwr
  do
    for size in 16k 256k 1M 2M 4M 8M 16M 32M 64M 256M 512M
      do
        # writes to stderr
        result=$(nice -n -20 bw_mem -P $(nproc) $size $what 2>&1)
        if [ $? -eq 0 ] && echo "$result" | grep -Eq '^[+-]?[0-9]+([.][0-9]+)?$'; then
          echo $what $result
        fi
      done
  done
