#!/bin/sh

cd /usr/local/binserve
ulimit -n 1000000

for size in 1k 16k 64k 256k 512k
  do
    truncate -s $size data/$size
done

binserve &
sleep 10

length="10"
url="http://localhost:8080"

# with smaller file sizes we likely need to open more connections to saturate the machine
for size in 1k 16k 64k
  do
    for threadmulti in 1 2 4
      do
        for connsmulti in 1 2 4 8 16 32
          do
            conns=$(($(nproc)*$connsmulti))
            threads=$(($(nproc)*$threadmulti))
            if [ $conns -lt $threads ]; then
              continue
            fi
            wrk -d $length -t $threads -c $conns ${url}/${size}
        done
    done
done

for size in 256k 512k
  do
    for threadmulti in 1 2 4
      do
        for connsmulti in 1 2 4 8 16
          do
            conns=$(($(nproc)*$connsmulti))
            threads=$(($(nproc)*$threadmulti))
            if [ $conns -lt $threads ]; then
              continue
            fi
            wrk -d $length -t $threads -c $conns ${url}/${size}
          done
      done
done