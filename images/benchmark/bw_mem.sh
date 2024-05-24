#!/bin/sh

for what in rd wr rdwr
  do
    for size in 16k 256k 1M 2M 4M 8M 16M 32M 64M 256M 512M
      do
        echo -n "$what "
        bw_mem -P $(nproc) $size $what
      done
  done
