import logging

logger = logging.getLogger(__name__)

def JuliaInterface():
    from os import environ

    from dotenv import load_dotenv
    from juliacall import Main as jl
    from juliacall import Pkg as jlPkg
    from juliacall import newmodule

    load_dotenv()

    # set julia environments
    if environ.get("CCC_TYPE") == "path":
        logger.debug(f"Retrieving CCC from the path provided {environ.get('CCC_PATH')}")
        # activate environment
        logger.debug("Activating packages from CCC environment")
        jlPkg.activate(environ.get("CCC_PATH"))

        # jlPkg.add("IntervalArithmetic")

        # activate CCC package
        logger.debug("Activating CCC")
        newmodule(environ.get("CCC_PATH"))

        # logger.debug("Resolving jlPkg")
        # jlPkg.resolve()
        
        logger.debug("Setting Julia to use CCC")
        jl.seval("using CCC")

    return jl
