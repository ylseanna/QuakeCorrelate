import logging
from os import environ
from typing import Literal
from sys import stdout

# from pathlib import Path

logger = logging.getLogger(__name__)


def JuliaInterface(
    update: bool = False, gpu_architecture: Literal["CUDA", "Metal"] | None = None
):
    from dotenv import load_dotenv

    import juliapkg as jlPkg
    from juliacall import Main as jl

    jl.eval("restart()")

    load_dotenv()

    environ["PYTHON_JULIACALL_THREADS"] = "1"
    environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"

    logger.debug("Setting Julia to use Squiggles")
    jl.seval("using Squiggles")

    if "QUAKECORRELATE_GPU_ARCH" in environ and not gpu_architecture:
        if environ.get("QUAKECORRELATE_GPU_ARCH") == "CUDA":
            jl.seval("using CUDA")
        elif environ.get("QUAKECORRELATE_GPU_ARCH") == "Metal":
            jl.seval("using Metal")
    elif gpu_architecture:
        jl.seval(f"using {gpu_architecture}")
    else:  # default to CUDA
        logger.info(
            "No GPU architecture set in the environment variables or manually set, defaulting to setting Julia to use CUDA"
        )
        jl.seval("using CUDA")

    jl.Squiggles.set_GPUbackend()

    return jl


def setup_julia(
    gpu_architecture: Literal["CUDA", "Metal"] | None = None,
    CUDA_runtime_version: str | None = None,
):
    from dotenv import load_dotenv

    from juliacall import Main as jl

    logger.info("Setting up Julia for QuakeCorrelate...")

    load_dotenv()

    jl.seval("using Pkg")

    jl.Pkg.add(
        "Squiggles",
        uuid="d934da09-a6a1-40ff-a16a-ac00174578b4",
        url="git@github.com:gbene/Squiggles.jl.git",
    )

    if "QUAKECORRELATE_GPU_ARCH" in environ and not gpu_architecture:
        if environ.get("QUAKECORRELATE_GPU_ARCH") == "CUDA":
            jl.Pkg.add("CUDA")
        elif environ.get("QUAKECORRELATE_GPU_ARCH") == "Metal":
            jl.Pkg.add("Metal")
    elif gpu_architecture:
        jl.Pkg.add(gpu_architecture)
    else:
        logger.warning(
            "No GPU architecture set in the environment variables (use QUAKECORRELATE_GPU_ARCH in your current environment or .env file.) or in the function call, defaulting to CUDA"
        )
        jl.Pkg.add("CUDA")

    logger.debug("Resolving jlPkg")
    jl.Pkg.resolve()

    if "QUAKECORRELATE_GPU_ARCH" in environ and not gpu_architecture:
        if environ.get("QUAKECORRELATE_GPU_ARCH") == "CUDA":
            logger.info("Setting julia to use CUDA")
            jl.seval("using CUDA")

            if environ.get("QUAKECORRELATE_CUDA_RUNTIME_VERSION"):
                jl.seval(
                    f'CUDA.set_runtime_version!(v"{environ.get("QUAKECORRELATE_CUDA_RUNTIME_VERSION")}")'
                )
            else:
                logger.info(
                    "Julia will use the default runtime version of CUDA, NOTE: may cause errors on HPC/node-based clusters, use: QUAKECORRELATE_CUDA_RUNTIME_VERSION in your current environment or .env file."
                )
        elif environ.get("QUAKECORRELATE_GPU_ARCH") == "Metal":
            logger.info("Setting julia to use Metal")
            jl.seval("using Metal")
    if gpu_architecture or CUDA_runtime_version:
        if gpu_architecture == "CUDA":
            logger.info("Setting julia to use CUDA")
            jl.seval("using CUDA")

            if CUDA_runtime_version:
                jl.seval(f'CUDA.set_runtime_version!(v"{CUDA_runtime_version}")')
            else:
                logger.info(
                    "Julia will use the default runtime version of CUDA, NOTE: may cause errors on HPC/node-based clusters, use: QUAKECORRELATE_CUDA_RUNTIME_VERSION in your current environment or .env file, or specify in the function call."
                )
        elif gpu_architecture == "Metal":
            logger.info("Setting julia to use Metal")
            jl.seval("using Metal")
    else:
        logger.warning(
            "No GPU architecture or QUAKECORRELATE_CUDA_RUNTIME_VERSION set in the environment variables or in the function call, setting Julia to use CUDA"
        )
        jl.seval("using CUDA")
