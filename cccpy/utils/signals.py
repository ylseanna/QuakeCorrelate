"""Generate synthetic signals (using CCC)"""

from os import environ

from juliacall import Main as jl
from juliacall import newmodule
from juliacall import Pkg as jlPkg

from dotenv import load_dotenv

load_dotenv()

# set julia environments
if environ.get("CCC_TYPE") == "path":
    # activate environment
    jlPkg.activate(environ.get("CCC_PATH"))

    # activate package
    newmodule(environ.get("CCC_PATH"))

def generate_signals():
    pass

