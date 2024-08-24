#!/bin/sh

base_port=7000
procs=$(nproc)
threads="1 2 4"
pipelines="1 16 64 128 256"

for i in $(seq $procs)
do
  port=$((base_port + i))
  mkdir -p /data/redis$port
  redis-server --dir /data/redis$port --port $port > /dev/null 2>&1 &
done

sleep 5

for thread in $threads
do
  for pipeline in $pipelines
  do
    # dump DBs in each round
    for i in $(seq $procs)
    do
      port=$((base_port + i))
      redis-cli -p $port flushall > /dev/null 2>&1 &
    done
    # wait a little
    sleep 1
    pids=""
    for i in $(seq $procs)
    do
      port=$((base_port + i))
      memtier_benchmark -p $port --ratio=1:0 --test-time=10 -t $thread --hide-histogram --pipeline=$pipeline --json-out-file=/tmp/port=$port,t=$thread,p=$pipeline -o /dev/null &
      pids="$pids $!"
    done

    # Wait for only the processes from the second loop to finish
    for pid in $pids; do
      wait "$pid"
    done
  done
done

for thread in $threads
do
  for pipeline in $pipelines
  do
    for i in $(seq $procs)
    do
      port=$((base_port + i))
      echo "#### threads=$thread,pipeline=$pipeline,port=$port"
      cat /tmp/port=$port,t=$thread,p=$pipeline
    done
  done
done