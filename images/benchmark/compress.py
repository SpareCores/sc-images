from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait, as_completed
import brotli
import bz2
import bz3
import gzip
import json
import lz4.frame
import lzma
import math
import multiprocessing
import os
import psutil
import pyzstd
import time
import traceback
import zpaq

# The number of usable/available physical processors, whichever is smaller
NUMCPUS = min(psutil.cpu_count(logical=False), len(os.sched_getaffinity(0)))
REPEAT = 2
DATA = open("/usr/local/silesia/dickens", "rb").read()
NO_RESULT = {
    "compress": None,
    "decompress": None,
    "ratio": None,
}

TASKS = {
    "brotli": {
        "levels": [0, 4, 8, 11],
        "threads": [1, NUMCPUS],
        "compress": lambda level, kwargs: brotli.compress(DATA, quality=level),
        "decompress": lambda data: brotli.decompress(data),
    },
    "gzip": {
        "levels": [1, 5, 9],
        "threads": [1, NUMCPUS],
        "compress": lambda level, kwargs: gzip.compress(DATA, compresslevel=level),
        "decompress": lambda data: gzip.decompress(data),
    },
    "bzip2": {
        "levels": [1, 5, 9],
        "threads": [1, NUMCPUS],
        "compress": lambda level, kwargs: bz2.compress(DATA, compresslevel=level),
        "decompress": lambda data: bz2.decompress(data),
    },
    "lzma": {
        "levels": [1, 5, 9],
        "threads": [1, NUMCPUS],
        "compress": lambda level, kwargs: lzma.compress(DATA, preset=level),
        "decompress": lambda data: lzma.decompress(data),
    },
    "zstd": {
        "levels": [1, 7, 14, 22],
        "threads": [1, NUMCPUS],
        # https://pyzstd.readthedocs.io/en/stable/#cparameter
        "compress": lambda level, kwargs: pyzstd.compress(DATA,
                                                                   level_or_option={
                                                                       pyzstd.CParameter.compressionLevel: level,
                                                                       **kwargs,
                                                                   }),
        "decompress": lambda data: pyzstd.decompress(data),
    },
    "bzip3": {
        "levels": [None],
        "threads": [1, NUMCPUS],
        "compress": lambda level, kwargs: bz3.compress(DATA, **kwargs),
        "decompress": lambda data: bz3.decompress(data),
        "extra_args": [
            dict(block_size=1024 ** 2),  # default
            dict(block_size=64 * 1024 ** 2),
        ],
    },
    "zpaq": {
        "levels": [1, 3, 5],
        "threads": [1, NUMCPUS],
        "compress": lambda level, kwargs: zpaq.compress(DATA, level),
        "decompress": lambda data: zpaq.decompress(data),
    },
    "lz4": {
        "levels": [1, 6, 12, 16],
        "threads": [1, NUMCPUS],
        "compress": lambda level, kwargs: lz4.frame.compress(DATA, level, **kwargs),
        "decompress": lambda data: lz4.frame.decompress(data),
        "extra_args": [
            dict(block_size=0),  # default, currently 64k
            dict(block_size=6),  # 1M
            dict(block_size=7),  # 4M
        ],
    },
}


def measured_f(event, func, *args, **kwargs):
    event.wait()
    st = time.time()
    TASKS[compressor][func](*args, **kwargs)
    return time.time() - st


def measure(compressor, idx, threads, extra_args):
    """
    Measure the speed/ratio of a given compressor by running one instance on each CPU.
    To avoid locking issues, we run the compressors in their own processes. We pass a multiprocessing.Event() object
    and wait after the processes have been created in order to synchronize them, so all compress/decompress jobs
    start and run in about the same time.
    """
    res = {"threads": threads, "extra_args": extra_args}
    best_time_compress = math.inf
    best_time_decompress = math.inf

    # measure compression level
    level = TASKS[compressor]["levels"][idx]
    # first we measure the compression ratio on one thread because we don't want to pass data between processes
    compressed_data = TASKS[compressor]["compress"](level, extra_args)
    res["ratio"] = len(compressed_data) / len(DATA) * 100
    decompressed_data = TASKS[compressor]["decompress"](compressed_data)
    # test if the compression/decompression cycle is working
    assert decompressed_data == DATA

    for _ in range(REPEAT):
        # compress
        with ProcessPoolExecutor() as executor:
            with multiprocessing.Manager() as manager:
                event = manager.Event()
                futures = {executor.submit(measured_f, event, "compress", level, extra_args) for i in range(threads)}
                time.sleep(1)
                event.set()
                elapsed = sum([future.result() for future in as_completed(futures)]) / threads
                if elapsed < best_time_compress:
                    best_time_compress = elapsed
                    res["compress"] = (len(DATA) * threads) / elapsed

        # decompress
        with ProcessPoolExecutor() as executor:
            with multiprocessing.Manager() as manager:
                event = manager.Event()
                futures = {executor.submit(measured_f, event, "decompress", compressed_data) for i in range(threads)}
                time.sleep(1)
                event.set()
                elapsed = sum([future.result() for future in as_completed(futures)]) / threads
                if elapsed < best_time_decompress:
                    best_time_decompress = elapsed
                    res["decompress"] = (len(DATA) * threads) / elapsed
    return res

results = defaultdict(lambda: {})
for compressor, methods in TASKS.items():
    for idx in range(len(methods["levels"])):
        level = methods["levels"][idx]
        results[compressor][level] = []
        for threads in TASKS[compressor].get("threads", [1]):
            for extra_args in TASKS[compressor].get("extra_args", [{}]):
                with ProcessPoolExecutor(max_workers=1) as executor:
                    # start measure in its own process, so we have a chance to survive the OOM-killer should that
                    # kick in
                    f = executor.submit(measure, compressor, idx, threads, extra_args)
                    try:
                        results[compressor][level].append(f.result())
                    except Exception:
                        results[compressor][level].append(NO_RESULT | dict(threads=threads, extra_args=extra_args))
                        import traceback
                        traceback.print_exc()

print(json.dumps(results, indent=2))
