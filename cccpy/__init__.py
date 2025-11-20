__version__ = "0.1.0"
__author__ = "Ylse Anna de Vries <yad2@hi.is>, Gabriele Benedetti <gab28@hi.is>"

from .core import ncc  # noqa: F401
from .seismic import cross_correlate_seismic_events  # noqa: F401

from . import signals  # noqa: F401

# set up environment
from os import environ
from dotenv import load_dotenv

from juliacall import Main as jl
from juliacall import newmodule
from juliacall import Pkg as jlPkg


load_dotenv()

# set julia environments
if environ.get("CCC_TYPE") == "path":
    # activate environment
    jlPkg.activate(environ.get("CCC_PATH"))

    # jlPkg.add("IntervalArithmetic")

    # activate CCC package
    newmodule(environ.get("CCC_PATH"))

    jlPkg.resolve()


    jl.seval("using CCC")
