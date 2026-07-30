import logging
from os import environ

from dotenv import load_dotenv
from juliacall import Main as jl
import juliapkg as jlPkg

# from pathlib import Path

logger = logging.getLogger(__name__)

jl.eval("restart()")


def JuliaInterface(update: bool = False):
    load_dotenv()
    

    environ["PYTHON_JULIACALL_THREADS"] = "1"
    environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
    
    installed_julia_packages = dict(jlPkg.installed()).keys()
    logger.debug(f"Installed Julia packages: {installed_julia_packages}")

    if (
        "Squiggles" not in installed_julia_packages
        or environ.get("GPU_ARCH") not in installed_julia_packages
    ):
        logger.info("Not all packages found, adding Julia packages...")

        if "Squiggles" not in installed_julia_packages:
            jlPkg.add(url="git@github.com:gbene/Squiggles.jl.git")

        if environ.get("GPU_ARCH") == "CUDA" and "CUDA" not in installed_julia_packages:
            jlPkg.add("CUDA")
        elif (
            environ.get("GPU_ARCH") == "Metal"
            and "Metal" not in installed_julia_packages
        ):
            jlPkg.add("Metal")
        
        logger.debug("Resolving jlPkg")
        jlPkg.resolve()

            
    if update:
        jlPkg.update()

        logger.debug("Resolving jlPkg")
        jlPkg.resolve()

    logger.debug("Setting Julia to use Squiggles")
    jl.seval("using Squiggles")

    if environ.get("GPU_ARCH") == "CUDA":
        jl.seval("using CUDA")
    elif environ.get("GPU_ARCH") == "Metal":
        jl.seval("using Metal")

    jl.Squiggles.set_GPUbackend()

    return jl
