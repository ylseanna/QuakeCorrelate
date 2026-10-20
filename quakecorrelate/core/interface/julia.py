import logging
from os import environ

import juliapkg as jlPkg
from dotenv import load_dotenv
from juliacall import Main as jl

# from pathlib import Path

logger = logging.getLogger(__name__)

jl.eval("restart()")


def JuliaInterface(update: bool = False):
    load_dotenv()

    environ["PYTHON_JULIACALL_THREADS"] = "1"
    environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    logger.debug("Setting Julia to use Squiggles")
    jl.seval("using Squiggles")

    if environ.get("QUAKECORRELATE_GPU_ARCH") == "CUDA":
        jl.seval("using CUDA")
    elif environ.get("QUAKECORRELATE_GPU_ARCH") == "Metal":
        jl.seval("using Metal")

    jl.Squiggles.set_GPUbackend()

    return jl
