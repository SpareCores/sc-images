#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import cache
from logging import DEBUG, StreamHandler, basicConfig, getLogger
from os import chdir, listdir, path
from subprocess import run
from sys import stderr
from urllib.request import urlretrieve

from psutil import cpu_count

basicConfig(
    level=DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[StreamHandler(stderr)],
)
logger = getLogger("benchmark")

cli_parser = ArgumentParser(description="Benchmark LLM model inference speed")
cli_parser.add_argument(
    "--model-urls",
    nargs="+",
    type=str,
    default=[
        "https://huggingface.co/QuantFactory/SmolLM-135M-GGUF/resolve/main/SmolLM-135M.Q4_K_M.gguf",
        "https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat-GGUF/resolve/main/qwen1_5-0_5b-chat-q4_k_m.gguf",
        # "https://huggingface.co/bartowski/codegemma-2b-GGUF/blob/main/codegemma-2b-Q4_K_M.gguf",
        # "https://huggingface.co/microsoft/phi-4-gguf/blob/main/phi-4-q4.gguf",
        # "https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/blob/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf",
    ],
    help="List of URLs of quantized LLM models (gguf) to download andbenchmark.",
)
cli_parser.add_argument(
    "--models-dir",
    type=str,
    default="/models",
    help="Directory to cache/store downloaded models.",
)
cli_args = cli_parser.parse_args()

# #############################################################################

TIMEOUT = 60
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
    # offload as many layers as possible
    "-ngl",
    "999",  # TODO update after testing per model
    # use default batch sizes
    "-ub",
    "512",
    "-b",
    "2048",
    # output to jsonl
    "-o",
    "jsonl",
]


# #############################################################################


@cache
def get_llama_cpp_path():
    """Check if GPU/CUDA is available, if not, use CPU-build of llama.cpp."""
    llama_cpp_path = "/llama_cpp_gpu"
    result = run(["./llama-cli", "--version"], cwd=llama_cpp_path, capture_output=True)
    if result.returncode != 0:
        llama_cpp_path = "/llama_cpp_cpu"
        logger.info("Using CPU-build of llama.cpp")
    else:
        logger.info("Using GPU-build of llama.cpp")
    return llama_cpp_path


def cuda_available():
    return get_llama_cpp_path() == "/llama_cpp_gpu"


def download_models(model_urls: list[str], models_dir: str):
    """Download gguf models from provided URLs."""
    for model_url in model_urls:
        model_name = model_url.split("/")[-1]
        model_path = path.join(models_dir, model_name)
        if path.exists(model_path):
            logger.debug(f"Model {model_name} already exists, skipping download")
        else:
            logger.debug(f"Downloading model {model_name} from {model_url}")
            urlretrieve(model_url, model_path)


def list_models(models_dir: str):
    """List all .gguf model files in the models directory."""

    return [
        f
        for f in listdir(models_dir)
        if path.isfile(path.join(models_dir, f)) and f.endswith(".gguf")
    ]


def max_ngl(model: str):
    """Find max ngl that doesn't fail."""
    if not cuda_available():
        return 0
    for ngl in [999, 40, 24, 12]:
        try:
            result = run(
                COMMAND + ["-m", model, "-ngl", str(ngl), "-r", "1", "-t", "1"],
                capture_output=True,
                timeout=TIMEOUT,
            )
            if result.returncode == 0:
                return ngl
        except Exception as e:
            logger.debug(f"Error testing ngl {ngl} for model {model}: {e}")
            continue
    return 0


# #############################################################################


chdir(get_llama_cpp_path())
download_models(model_urls=cli_args.model_urls, models_dir=cli_args.models_dir)
models = list_models(cli_args.models_dir)


#     -p 0,16,32,128,256,512,1024,2048,4096,8192,16384,32768 \
#     -n 1,16,32,128,512,1024,2048,4096,8192 \
#     -o jsonl


for model in models:
    try:
        ngl = max_ngl(path.join(cli_args.models_dir, model))
        print(ngl)
        # run(COMMAND + ["-m", model, "-ngl", str(ngl)])
    except Exception as e:
        logger.error(f"Failed to process model {model}: {e}")
        continue
