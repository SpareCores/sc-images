#!/usr/bin/env python3

import re
from argparse import ArgumentParser
from functools import cache
from logging import DEBUG, StreamHandler, basicConfig, getLogger
from multiprocessing import Manager, Process
from os import chdir, listdir, nice, path, rename, unlink
from signal import SIGINT, SIGTERM, signal
from subprocess import run
from sys import exit as sys_exit
from sys import stderr
from time import time
from typing import Optional
from urllib.request import Request, urlopen, urlretrieve

from psutil import cpu_count

basicConfig(
    level=DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler(stderr)],
)
logger = getLogger("benchmark")

cli_parser = ArgumentParser(description="Benchmark LLM model inference speed")
cli_parser.add_argument(
    "--version",
    action="store_true",
    help="Print version information and exit",
)
cli_parser.add_argument(
    "--model-urls",
    nargs="+",
    type=str,
    default=[
        "https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/b35f229108e4c5a4deeccccf9bf75cd46006fa7b/SmolLM-135M.Q4_K_M.gguf",  # 135 M / 100 MB
        "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/cfab082d2fef4a8736ef384dc764c2fb6887f387/qwen1_5-0_5b-chat-q4_k_m.gguf",  # 0.5 B / 400 MB
        "https://huggingface.co/mlabonne/gemma-2b-GGUF/resolve/8dd617a80664da2853999b6b1206595f1c33daa2/gemma-2b.Q4_K_M.gguf",  # 2B / 1.5 GB
        "https://huggingface.co/TheBloke/LLaMA-7b-GGUF/resolve/0a1200190c8c1baafaa0b496c599ba68ad3d4056/llama-7b.Q4_K_M.gguf",  # 7B / 4 GB
        "https://huggingface.co/microsoft/phi-4-gguf/resolve/b1e764cfdbdd0a3ed824d6a8424129eb0a2232ff/phi-4-q4.gguf",  # 14 B / 9 GB
        "https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/resolve/7bcd751e48e909680bd2168359dcf7fb53b12da8/Llama-3.3-70B-Instruct-Q4_K_M.gguf",  # 70 B / 42 GB
    ],
    help="List of URLs of quantized LLM models (gguf) to download and benchmark.",
)
cli_parser.add_argument(
    "--models-dir",
    type=str,
    default="/models",
    help="Directory to cache/store downloaded models.",
)
cli_parser.add_argument(
    "--benchmark-timeout-scale",
    type=int,
    default=1,
    help="Scale the benchmark timeout by this factor. The default is 1, using a dynamic timeout for each benchmark secnario based on the model size (estimated time to load into memory/VRAM), the task (text generation is slower than prompt processing), and the number of tokens as requiring higher tokens/sec for larger number of tokens. Setting this to 2 means the timeout will be doubled etc.",
)
cli_parser.add_argument(
    "--download-timeout",
    type=int,
    default=60 * 15,
    help="Timeout for downloading models in seconds.",
)
cli_args = cli_parser.parse_args()

# #############################################################################

# default command for llama-bench
COMMAND = [
    "./llama-bench",
    # best performance is achieved with all physical cores
    "-t",
    str(cpu_count(logical=False)),
    # split by layer .. don't bother with rows, as although
    # it might be useful to harness smalerr GPUs, but much slower
    # and we don't want to benchmark very specific cases/needs
    "-sm",
    "layer",
    # flash attention is always faster
    "-fa",
    "1",
    # use default batch sizes
    "-ub",
    "512",
    "-b",
    "2048",
    # output to jsonl
    "-o",
    "jsonl",
]

BENCHMARKS = [
    {
        "name": "prompt processing",
        "iterations": [16, 128, 512, 1024, 4096, 16384],
        "iteration_param": "-p",
        "expected_tps": [2, 10, 25, 50, 250, 1000],
        "extra_params": ["-n", "0"],
    },
    {
        "name": "text generation",
        "iterations": [16, 128, 512, 1024, 4096],
        "iteration_param": "-n",
        "expected_tps": [1, 5, 25, 50, 250],
        "extra_params": ["-p", "0"],
    },
]

# #############################################################################


@cache
def get_llama_cpp_path():
    """Check if GPU/CUDA is available, if not, use CPU-build of llama.cpp."""
    llama_cpp_path = "/llama_cpp_gpu"
    # ARM64 doesn't have the CUDA binaries, so we need to use the CPU-build with early return
    if not path.exists(llama_cpp_path):
        logger.info("Using CPU-build of llama.cpp")
        return "/llama_cpp_cpu"
    # check if we can start the CUDA binary
    result = run(["./llama-cli", "--version"], cwd=llama_cpp_path, capture_output=True)
    if result.returncode != 0:
        llama_cpp_path = "/llama_cpp_cpu"
        logger.info("Using CPU-build of llama.cpp")
    else:
        logger.info("Using GPU-build of llama.cpp")
    return llama_cpp_path


def cuda_available():
    return get_llama_cpp_path() == "/llama_cpp_gpu"


def get_model_url_size(model_url: str):
    """Use a HEAD request to get the size of an URL.

    Returns:
        int: The size of the model in MB.
    """
    with urlopen(Request(model_url, method="HEAD")) as response:
        return int(response.headers.get("Content-Length"), 0) / 1024**2


def download_models(
    model_urls: list[str],
    models_dir: str,
    model_events: dict = {},
    download_speeds: dict = {},
    renice: Optional[int] = None,
):
    """Download gguf models from provided URLs."""
    if renice:
        nice(renice)
    for model_url in model_urls:
        model_name = model_url.split("/")[-1]
        model_path = path.join(models_dir, model_name)
        if path.exists(model_path):
            logger.debug(f"Model {model_name} already exists, skipping download")
        else:
            logger.debug(f"Downloading model {model_name} from {model_url}")
            timer_start = time()
            temp_path = model_path + ".part"
            urlretrieve(model_url, temp_path)
            rename(temp_path, model_path)
            model_size = path.getsize(model_path) / 1024**2
            download_time = time() - timer_start
            download_speeds[model_name] = model_size / download_time
            logger.debug(
                f"Downloaded model {model_name} ({model_size:.2f} MB) in "
                f"{download_time:.2f} sec ({download_speeds[model_name]:.2f} MB/s)"
            )
        model_events[model_name].set()


def download_models_background(model_urls: list[str], models_dir: str):
    """Download gguf models from provided URLs in a background process.

    Returns:
        tuple[Process, dict[str, Event], dict[str, float]]: The background process, model download completion events, and download speeds.
    """
    manager = Manager()
    model_events = manager.dict()
    download_speeds = manager.dict()

    for url in model_urls:
        model_name = url.split("/")[-1]
        model_events[model_name] = manager.Event()

    renice = 19
    process = Process(
        target=download_models,
        args=(model_urls, models_dir, model_events, download_speeds, renice),
    )
    process.start()
    return process, model_events, download_speeds


def cleanup_partially_downloaded_models(models_dir: str):
    """Remove all .part files in the models directory."""
    for filename in listdir(models_dir):
        if filename.endswith(".part"):
            try:
                unlink(path.join(models_dir, filename))
                logger.debug(f"Deleted partially downloaded model file: {filename}")
            except FileNotFoundError:
                pass
            except Exception as e:
                logger.error(f"Failed to remove {filename}: {e}")


def signal_handler(signum, frame):
    """Handle interrupt signals by cleaning up and exiting."""
    logger.info("Received interrupt signal, cleaning up...")
    cleanup_partially_downloaded_models(cli_args.models_dir)
    sys_exit(128 + signum)


def list_models(models_dir: str):
    """List all .gguf model files in the models directory."""

    return [
        f
        for f in listdir(models_dir)
        if path.isfile(path.join(models_dir, f)) and f.endswith(".gguf")
    ]


def max_ngl(model: str):
    """Find max ngl that doesn't fail so that we can offload as many layers as possible."""
    if not cuda_available():
        return 0
    for ngl in [999, 40, 24, 12]:
        try:
            result = run(
                COMMAND + ["-m", model, "-ngl", str(ngl), "-r", "1", "-t", "1"],
                capture_output=True,
                timeout=120,  # static 2 mins that is longer than any tested scenario
            )
            if result.returncode == 0:
                return ngl
        except Exception as e:
            logger.debug(f"Error testing ngl {ngl} for model {model}: {e}")
            continue
    return 0


def get_llama_cpp_version():
    """Get the version of llama.cpp."""
    llama_cpp_path = get_llama_cpp_path()
    result = run(
        ["./llama-cli", "--version"], cwd=llama_cpp_path, capture_output=True, text=True
    )
    if result.returncode == 0:
        # looking for the pattern like "version: 4753 (51f311e0)"
        match = re.search(r"version: \d+ \(([a-f0-9]+)\)", result.stderr)
        if match:
            return match.group(1)  # extract commit hash (51f311e0)
    return "unknown"


# #############################################################################

chdir(get_llama_cpp_path())
signal(SIGINT, signal_handler)
signal(SIGTERM, signal_handler)

if cli_args.version:
    version = get_llama_cpp_version()
    print(version)
    sys_exit(0)

models_download_process, models_downloaded, download_speeds = (
    download_models_background(
        model_urls=cli_args.model_urls, models_dir=cli_args.models_dir
    )
)

for model_url in cli_args.model_urls:
    model_name = model_url.split("/")[-1]
    logger.info(f"Benchmarking model {model_name} ...")

    # models are downloaded in the background, so hopefully the model is already downloaded,
    # but if not, we estimate if we can finish downloading it without timing out
    if len(download_speeds) and not models_downloaded[model_name].is_set():
        try:
            downloaded_file_size = (
                path.getsize(path.join(cli_args.models_dir, model_name + ".part"))
                / 1024**2
            )
        except FileNotFoundError:
            downloaded_file_size = 0
        target_file_size = get_model_url_size(model_url)
        avg_download_speed = sum(download_speeds.values()) / len(download_speeds)
        remaining_file_size = target_file_size - downloaded_file_size
        # estimate if we can finish downloading the model in time
        if remaining_file_size > avg_download_speed * cli_args.download_timeout:
            logger.error(
                f"Downloading {model_name} ({remaining_file_size:.2f} MB remaining out of {target_file_size:.2f} MB) "
                f"would take too long with {avg_download_speed:.2f} MB/s, giving up."
            )
            models_download_process.terminate()
            models_download_process.join()
            sys_exit(0)
        logger.info(
            f"Waiting for {model_name} to be downloaded ({remaining_file_size:.2f} MB) ... ETA: {remaining_file_size / avg_download_speed:.2f} sec"
        )

    if not models_downloaded[model_name].wait(timeout=cli_args.download_timeout):
        logger.error(f"{model_name} was not downloaded in time.")
        models_download_process.terminate()
        models_download_process.join()
        sys_exit(0)

    model_path = path.join(cli_args.models_dir, model_name)
    model_size_gb = path.getsize(model_path) / 1024**3
    logger.debug(f"Model {model_name} found at {model_path} ({model_size_gb:.2f} GB)")
    ngl = max_ngl(model_path)
    logger.debug(f"Using ngl {ngl} for model {model_name}")

    cmd = COMMAND + ["-m", model_path, "-ngl", str(ngl)]
    for benchmark in BENCHMARKS:
        for i, iteration in enumerate(benchmark["iterations"]):
            start_time = time()
            timeout = round(
                (
                    # conservative estimate for loading the model into memory/VRAM with 250 MB/s
                    model_size_gb / 0.25
                    # 1 sec overhead
                    + 1
                    # 5 repeats in a benchmark
                    + (5 * iteration)
                    # make sure to stop at max allowed tokens/sec
                    / benchmark["expected_tps"][i]
                )
                * cli_args.benchmark_timeout_scale
            )
            logger.debug(
                f"Benchmarking {benchmark['name']} with {iteration} tokens for max {timeout} sec"
            )
            try:
                run(
                    cmd
                    + [benchmark["iteration_param"], str(iteration)]
                    + benchmark["extra_params"],
                    timeout=timeout,
                )
            except Exception as e:
                logger.error(f"Error: {e}")
                if i == 0:
                    logger.info(
                        "Benchmarking failed with simplest task, so skipping larger models."
                    )
                    models_download_process.terminate()
                    models_download_process.join()
                    sys_exit(0)
                elif i != len(benchmark["iterations"]) - 1:
                    logger.info(
                        f"Skipping {benchmark['name']} benchmarks "
                        f"with {iteration}+ tokens due to time constraints."
                    )
                break
            # finish early if we're unlikely to finish in time,
            # as speedups are less likely to happen with 1024+ tokens
            if i != len(benchmark["iterations"]) - 1 and iteration >= 1024:
                if time() - start_time > (
                    timeout
                    # adjust for expected tokens/sec increase
                    * (benchmark["expected_tps"][i] / benchmark["expected_tps"][i + 1])
                    # adjust for potential small speedup when using more tokens
                    * 1.1
                ):
                    logger.error(
                        f"Skipping {benchmark['name']} benchmarks with {iteration}+ tokens "
                        f"as it's unlikely to hit the expected {benchmark['expected_tps'][i + 1]} tokens/sec."
                    )
                    break
