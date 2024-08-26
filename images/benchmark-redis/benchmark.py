import subprocess
import resource
import os
import time
import select
import psutil

BASE_PORT = 7000

def num_cpus():
    """Returns the number of CPUs available for running code."""
    return len(os.sched_getaffinity(0))


def get_pid_rusage(processes):
    cpu_times = {}
    for port, process in processes.items():
        cpu_times[port] = psutil.Process(process.pid).cpu_times()
    return cpu_times


def get_children_cpu_times():
    """
    Return resource usage statistics for all children of the calling process that have terminated and been waited for.
    These statistics will include the resources used by grandchildren, and further removed descendants, if all of the
    intervening descendants waited on their terminated children.
    """
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    user_time = usage.ru_utime
    sys_time = usage.ru_stime
    return user_time, sys_time


def sum_cpu_times(start, end):
    user_time = 0
    sys_time = 0
    for port, cputime in end.items():
        user_time += cputime.user - start[port].user
        sys_time += cputime.system - start[port].system
    return user_time, sys_time


def flushall(wait=False, wait_iters=10):
    for _ in range(wait_iters):
        res = []
        for i in range(0, num_cpus()):
            port = BASE_PORT + i
            r = subprocess.run(["redis-cli", "-p", str(port), "flushall"], capture_output=True)
            if r.returncode:
                res.append(False)
            else:
                res.append(True)
        if not wait:
            return res
        if all(res):
            return res
        time.sleep(1)
    return res


def start_server():
    processes = {}

    for i in range(0, num_cpus()):
        port = BASE_PORT + i
        os.makedirs(f"/data/redis-{port}", exist_ok=True)
        processes[port] = subprocess.Popen(
            ["redis-server", "--dir", f"/data/redis-{port}", "--port", str(port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return processes


def start_client(opts: list[str] = tuple()):
    processes = {}

    for i in range(0, num_cpus()):
        port = BASE_PORT + i
        processes[port] = subprocess.Popen(
            ["memtier_benchmark", "-4", "-p", str(port), "--ratio=1:0", "--test-time=10", "--hide-histogram"] + opts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line buffered
        )
    return processes


def get_outputs(processes):
    outputs = {port: {"stdout": "", "stderr": "", "exit_code": None} for port in processes.keys()}

    # Continuously monitor the processes
    while processes:
        # Prepare lists of stdout and stderr file descriptors to monitor
        read_fds = [process.stdout for process in processes.values()] + \
                   [process.stderr for process in processes.values()]

        # Use select to monitor these file descriptors
        readable, _, _ = select.select(read_fds, [], [])

        for port, process in list(processes.items()):
            if process.stdout in readable:
                # Read a line from stdout
                line = process.stdout.readline()
                if line:
                    outputs[port]["stdout"] += line
            if process.stderr in readable:
                # Read a line from stderr
                line = process.stderr.readline()
                if line:
                    outputs[port]["stderr"] += line

            # Check if the process has finished
            if process.poll() is not None:
                # Read any remaining output
                outputs[port]["stdout"] += process.stdout.read()
                outputs[port]["stderr"] += process.stderr.read()

                # Close the streams
                process.stdout.close()
                process.stderr.close()

                # Remove the finished process from the dictionary
                del processes[port]
    return outputs


def parse_outputs(outputs):
    """
    Sample output on stdout:
    4         Threads
    50        Connections per thread
    10        Seconds


    ALL STATS
    ============================================================================================================================
    Type         Ops/sec     Hits/sec   Misses/sec    Avg. Latency     p50 Latency     p99 Latency   p99.9 Latency       KB/sec
    ----------------------------------------------------------------------------------------------------------------------------
    Sets        68976.91          ---          ---         2.89753         2.67100         9.91900        31.23100      5313.79
    Gets            0.00         0.00         0.00             ---             ---             ---             ---         0.00
    Waits           0.00          ---          ---             ---             ---             ---             ---          ---
    Totals      68976.91         0.00         0.00         2.89753         2.67100         9.91900        31.23100      5313.79
    """
    parsed = {"rps": 0, "latency": 0}
    for port, output in outputs.items():
        for line in output["stdout"].splitlines():
            if line.startswith("Sets"):
                data = line.split()
                parsed["rps"] += float(data[1])
                parsed["latency"] += float(data[4])
    # avg
    parsed["latency"] = parsed["latency"] / len(outputs)
    return parsed


servers = start_server()
print("operation,threads,pipeline,rps,latency,server_usr,server_sys,client_usr,client_sys")
for thread in (1, 2, 4):
    for pipeline in (1, 4, 16, 64, 256, 512):
        flushall(wait=True)
        # record the user, sys CPU resource usage
        start_client_cpu_times = get_children_cpu_times()
        start_server_cpu_times = get_pid_rusage(servers)
        clients = start_client(["-t", str(thread), f"--pipeline={pipeline}"])
        outputs = get_outputs(clients)
        # after the clients have finished, get the CPU times again
        end_client_cpu_times = get_children_cpu_times()
        end_server_cpu_times = get_pid_rusage(servers)
        client_usr = end_client_cpu_times[0] - start_client_cpu_times[0]
        client_sys = end_client_cpu_times[1] - start_client_cpu_times[1]
        server_usr, server_sys = sum_cpu_times(start_server_cpu_times, end_server_cpu_times)
        parsed = parse_outputs(outputs)
        print(
            f"SET,{thread},{pipeline},{int(parsed['rps'])},{parsed['latency']:.2f},"
            f"{server_usr:.2f},{server_sys:.2f},{client_usr:.2f},{client_sys:.2f}"
        )
