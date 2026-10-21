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
else:
    logger.warning("No GPU architecture set in the environment variables (use QUAKECORRELATE_GPU_ARCH in your current environment or .env file.), defaulting to CUDA")
    jlPkg.add("CUDA")

logger.debug("Resolving jlPkg")
jlPkg.resolve()

if environ.get("QUAKECORRELATE_GPU_ARCH") == "CUDA":
    logger.info("Setting julia to use CUDA")
    jl.seval("using CUDA")

    if environ.get("QUAKECORRELATE_CUDA_RUNTIME_VERSION"):
        jl.seval(
            f'CUDA.set_runtime_version!(v"{environ.get("QUAKECORRELATE_CUDA_RUNTIME_VERSION")}")'
        )
    else:
        logger.info("Julia will use the default runtime version of CUDA, NOTE: may cause errors on HPC/node-based clusters, use: QUAKECORRELATE_CUDA_RUNTIME_VERSION in your current environment or .env file.")
elif environ.get("QUAKECORRELATE_GPU_ARCH") == "Metal":
    logger.info("Setting julia to use Metal")
    jl.seval("using Metal")
else:
    logger.warning("No GPU architecture set in the environment variables, setting Julia to use CUDA")
    jl.seval("using CUDA")


# if update:
#     jlPkg.update()

#     logger.debug("Resolving jlPkg")
#     jlPkg.resolve()
