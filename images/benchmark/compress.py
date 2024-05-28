from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import brotli
import bz2
import bz3
import gzip
import json
import lz4.frame
import lzma
import math
import os
import psutil
import pyzstd
import time
import zpaq


# The number of usable/available physical processors, whichever is smaller
NUMCPUS = min(psutil.cpu_count(logical=False), len(os.sched_getaffinity(0)))
REPEAT = 3
DATA = open("/usr/local/silesia/dickens", "rb").read()
NO_RESULT = {
    "compress": None,
    "decompress": None,
    "ratio": None,
}

TASKS = {
    "brotli": {
        "levels": [0, 4, 8, 11],
        "compress": lambda level, threads, kwargs: brotli.compress(DATA, quality=level),
        "decompress": lambda data, threads: brotli.decompress(data),
    },
    "gzip": {
        "levels": [1, 5, 9],
        "compress": lambda level, threads, kwargs: gzip.compress(DATA, compresslevel=level),
        "decompress": lambda data, threads: gzip.decompress(data),
    },
    "bzip2": {
        "levels": [1, 5, 9],
        "compress": lambda level, threads, kwargs: bz2.compress(DATA, compresslevel=level),
        "decompress": lambda data, threads: bz2.decompress(data),
    },
    "lzma": {
        "levels": [1, 5, 9],
        "compress": lambda level, threads, kwargs: lzma.compress(DATA, preset=level),
        "decompress": lambda data, threads: lzma.decompress(data),
    },
    "zstd": {
        "levels": [1, 7, 14, 22],
        "threads": [0, NUMCPUS],
        # https://pyzstd.readthedocs.io/en/stable/#cparameter
        "compress": lambda level, threads, kwargs: pyzstd.compress(DATA,
                                                                   level_or_option={
                                                                       pyzstd.CParameter.compressionLevel: level,
                                                                       pyzstd.CParameter.nbWorkers: threads,
                                                                       **kwargs,
                                                                   }),
        "decompress": lambda data, threads: pyzstd.decompress(data),
    },
    "bzip3": {
        "levels": [None],
        "threads": [1, NUMCPUS],
        "compress": lambda level, threads, kwargs: bz3.compress(DATA, num_threads=threads, **kwargs),
        "decompress": lambda data, threads: bz3.decompress(data, num_threads=threads),
        "extra_args": [
            dict(block_size=1024 ** 2),  # default
            dict(block_size=64 * 1024 ** 2),
        ],
    },
    "zpaq": {
        "levels": [1, 3, 5],
        "compress": lambda level, threads, kwargs: zpaq.compress(DATA, level),
        "decompress": lambda data, threads: zpaq.decompress(data),
    },
    "lz4": {
        "levels": [1, 6, 12, 16],
        "compress": lambda level, threads, kwargs: lz4.frame.compress(DATA, level, **kwargs),
        "decompress": lambda data, threads: lz4.frame.decompress(data),
        "extra_args": [
            dict(block_size=0),  # default, currently 64k
            dict(block_size=6),  # 1M
            dict(block_size=7),  # 4M
        ],
    },
}


def measure(compressor, idx, threads, extra_args):
    res = {"threads": threads, "extra_args": extra_args}
    best_time_compress = math.inf
    best_time_decompress = math.inf
    for _ in range(REPEAT):
        # measure compression
        level = TASKS[compressor]["levels"][idx]
        st = time.time()
        compressed_data = TASKS[compressor]["compress"](level, threads, extra_args)
        elapsed = time.time() - st
        if elapsed < best_time_compress:
            best_time_compress = elapsed
            res["ratio"] = len(compressed_data) / len(DATA) * 100
            res["compress"] = len(DATA) / elapsed

        # measure decompression
        st = time.time()
        decompressed_data = TASKS[compressor]["decompress"](compressed_data, threads)
        elapsed = time.time() - st
        assert len(decompressed_data) == len(DATA)
        if elapsed < best_time_decompress:
            best_time_decompress = elapsed
            res["decompress"] = len(DATA) / elapsed
    return res


results = defaultdict(lambda: {})
for compressor, methods in TASKS.items():
    for idx in range(len(methods["levels"])):
        level = methods["levels"][idx]
        results[compressor][level] = []
        for threads in TASKS[compressor].get("threads", [1]):
            for extra_args in TASKS[compressor].get("extra_args", [{}]):
                with ProcessPoolExecutor(max_workers=1) as executor:
                    f = executor.submit(measure, compressor, idx, threads, extra_args)
                    try:
                        results[compressor][level].append(f.result())
                    except Exception:
                        results[compressor][level].append(NO_RESULT | dict(threads=threads, extra_args=extra_args))
                        import traceback
                        traceback.print_exc()

print(json.dumps(results, indent=2))
