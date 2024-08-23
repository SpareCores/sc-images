#!/bin/sh

base_port=7000
procs=$(nproc)
mkdir /results

increment=50000
runtime_target=5
n=100000

for i in $(seq $procs)
do
  port=$((base_port + i))
  mkdir -p /data/redis$port
  redis-server --dir /data/redis$port --port $port > /dev/null 2>&1 &
done

sleep 5

while true; do
    # Start the benchmark and measure its runtime
    start_time=$(date +%s)
    redis-benchmark -p $((base_port + 1)) -t SET -q -n $n > /dev/null 2>&1
    benchmark_status=$?
    # If redis-benchmark failed, exit the loop
    if [ "$benchmark_status" -ne 0 ]; then
        echo "redis-benchmark failed to run with -n $n. Exiting."
        exit 1
    fi

    end_time=$(date +%s)

    # Calculate elapsed time
    elapsed_time=$((end_time - start_time))

    # Check if elapsed time is greater than or equal to preset time
    if [ "$elapsed_time" -ge "$runtime_target" ]; then
        echo "Benchmark with -n $n ran for $elapsed_time seconds."
        break
    fi

    # Increase the parameter for the next iteration
    n=$((n + increment))
done

for test in PING_INLINE PING_MBULK SET GET INCR LPUSH RPUSH LPOP RPOP SADD \
  HSET SPOP ZADD ZPOPMIN LRANGE_600 MSET XADD
do
  pids=""
  for i in $(seq $procs)
  do
    port=$((base_port + i))
    if [ "$test" = "PING_INLINE" ]; then
      # only write CSV header on the first test
      redis-benchmark -p $port -t $test --csv -n $n >> /results/$port &
    else
      # and remove it from subsequent ones
      redis-benchmark -p $port -t $test --csv -n $n | grep -v avg_latency_ms >> /results/$port &
    fi
    pids="$pids $!"
  done

  # Wait for only the processes from the second loop to finish
  for pid in $pids; do
    wait "$pid"
  done
done

for i in $(seq $procs)
do
  port=$((base_port + i))
  echo "Results for redis-benchmark#$port"
  cat /results/$port
done