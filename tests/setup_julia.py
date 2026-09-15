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

if environ.get("GPU_ARCH") == "CUDA":
    jlPkg.add("CUDA")
elif environ.get("GPU_ARCH") == "Metal":
    jlPkg.add("Metal")

logger.debug("Resolving jlPkg")
jlPkg.resolve()

if environ.get("GPU_ARCH") == "CUDA":
    jl.seval("using CUDA")
    jl.seval('CUDA.set_runtime_version!(v"12.4.0")')

# if update:
#     jlPkg.update()

#     logger.debug("Resolving jlPkg")
#     jlPkg.resolve()
