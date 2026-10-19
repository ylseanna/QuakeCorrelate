import logging
import sys
from os import environ
from dotenv import load_dotenv

import juliapkg as jlPkg
from juliacall import Main as jl

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger("root")

logger.info("Setting up Julia for QuakeCorrelate...")

load_dotenv()

jlPkg.add(
    "Squiggles",
    uuid="d934da09-a6a1-40ff-a16a-ac00174578b4",
    url="git@github.com:gbene/Squiggles.jl.git",
)

if environ.get("QUAKECORRELATE_GPU_ARCH") == "CUDA":
    jlPkg.add("CUDA")
elif environ.get("QUAKECORRELATE_GPU_ARCH") == "Metal":
    jlPkg.add("Metal")

logger.debug("Resolving jlPkg")
jlPkg.resolve()

if environ.get("QUAKECORRELATE_GPU_ARCH") == "CUDA":
    jl.seval("using CUDA")

    if environ.get("QUAKECORRELATE_CUDA_RUNTIME_VERSION"):
        jl.seval(
            f'CUDA.set_runtime_version!(v"{environ.get("QUAKECORRELATE_CUDA_RUNTIME_VERSION")}")'
        )
if environ.get("QUAKECORRELATE_GPU_ARCH") == "Metal":
    jl.seval("using Metal")


# if update:
#     jlPkg.update()

#     logger.debug("Resolving jlPkg")
#     jlPkg.resolve()
