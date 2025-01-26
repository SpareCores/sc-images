#!/usr/bin/env python3

from argparse import ArgumentParser
from functools import cache
from os import path
from subprocess import run
from urllib.request import urlretrieve

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


@cache
def get_llama_cpp_path():
    """Check if GPU/CUDA is available, if not, use CPU-build of llama.cpp."""
    llama_cpp_path = "/llama_cpp_gpu"
    result = run(["./llama-cli", "--version"], cwd=llama_cpp_path, capture_output=True)
    if result.returncode != 0:
        llama_cpp_path = "/llama_cpp_cpu"
    return llama_cpp_path


def download_models(model_urls: list[str], models_dir: str):
    for model_url in model_urls:
        model_name = model_url.split("/")[-1]
        model_path = path.join(models_dir, model_name)
        if not path.exists(model_path):
            urlretrieve(model_url, model_path)


print(get_llama_cpp_path())
download_models(model_urls=cli_args.model_urls, models_dir=cli_args.models_dir)
