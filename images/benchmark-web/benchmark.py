import os
import psutil
import re
import resource
import select
import subprocess
import time
import urllib.request


BASE_PORT = 8080
BINSERVE_PATH = "/usr/local/binserve"
# keep this in sync with binserve config
SIZES = [1, 16, 64, 256, 512]


def create_or_truncate_file(file_path, size):
    # Open the file with write and create permissions
    fd = os.open(file_path, os.O_WRONLY | os.O_CREAT)

    try:
        # Truncate or extend the file to the specified size
        os.ftruncate(fd, size)
    finally:
        os.close(fd)


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


def start_server():
    processes = {}
    for size in SIZES:
        create_or_truncate_file(f"{BINSERVE_PATH}/data/{size}k", size * 1024)

    processes[BASE_PORT] = subprocess.Popen(
        ["binserve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=BINSERVE_PATH,
    )

    request = urllib.request.Request(f"http://localhost:{BASE_PORT}/1k", method="HEAD")
    for i in range(10):
        try:
            r = urllib.request.urlopen(request)
            assert r.status == 200
            break
        except Exception as e:
            print(e)
            time.sleep(1)
    else:
        return {}
    return processes


def start_client(opts: list[str] = tuple()):
    processes = {}

    processes[BASE_PORT] = subprocess.Popen(
            ["wrk"] + opts,
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
    Running 10s test @ http://localhost:8080/1k
      32 threads and 32 connections
      Thread Stats   Avg      Stdev     Max   +/- Stdev
        Latency    76.07us  448.23us  20.06ms   97.57%
        Req/Sec    46.73k    21.88k  122.69k    59.98%
      15026207 requests in 10.10s, 17.83GB read
      Non-2xx or 3xx responses: 1066764  <- optional, on errors
    Requests/sec: 1487738.70
    Transfer/sec:      1.77GB
    """
    parsed = {"rps": 0, "latency": 0, "failed": 0}
    for port, output in outputs.items():
        if m := re.search(r"Latency[\s]+([0-9.]+)(us|ms|s)", output["stdout"]):
            parsed["latency"] = float(m.group(1))
            suffix = m.group(2)
            if suffix == "us":
                parsed["latency"] = parsed["latency"] * 10**-6
            elif suffix == "ms":
                parsed["latency"] = parsed["latency"] * 10**-3
        parsed["rps"] = int(re.search(r"Requests/sec: ([0-9]+)", output["stdout"]).group(1))
        if m := re.search(r"Non-2xx or 3xx responses: ([0-9]+)", output["stdout"]):
            parsed["failed"] = int(m.group(1))

    return parsed


servers = start_server()
print("size,threads,connections,rps,latency,failed,server_usr,server_sys,client_usr,client_sys")
# we start a redis-memtier_benchmark pair for each CPU, so using more threads doesn't improve performance
for size in ("1k", "16k", "64k", "256k", "512k"):
    for threadmulti in (1, 2, 4):
        for connsmulti in (1, 2, 4, 8, 16, 32):
            if size in ("256k", "512k") and connsmulti > 16:
                # with smaller file sizes, we likely need to open more connections to saturate the machine
                continue
            conns = num_cpus() * connsmulti
            threads = num_cpus() * threadmulti
            if conns < threads:
                continue
            opts=(["-d", "10", "-t", str(threads), "-c", str(conns), f"http://localhost:{BASE_PORT}/{size}"])
            # record the user, sys CPU resource usage
            start_client_cpu_times = get_children_cpu_times()
            start_server_cpu_times = get_pid_rusage(servers)
            clients = start_client(opts)
            outputs = get_outputs(clients)
            # after the clients have finished, get the CPU times again
            end_client_cpu_times = get_children_cpu_times()
            end_server_cpu_times = get_pid_rusage(servers)
            client_usr = end_client_cpu_times[0] - start_client_cpu_times[0]
            client_sys = end_client_cpu_times[1] - start_client_cpu_times[1]
            server_usr, server_sys = sum_cpu_times(start_server_cpu_times, end_server_cpu_times)
            parsed = parse_outputs(outputs)
            print(
                f"{size},{threads},{conns},{int(parsed['rps'])},{parsed['latency']:.9f},"
                f"{server_usr:.2f},{server_sys:.2f},{client_usr:.2f},{client_sys:.2f}"
            )
