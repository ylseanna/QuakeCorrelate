def Julia():
    from os import environ

    from dotenv import load_dotenv
    from juliacall import Main as jl
    from juliacall import Pkg as jlPkg
    from juliacall import newmodule

    load_dotenv()

    # set julia environments
    if environ.get("CCC_TYPE") == "path":
        # activate environment
        jlPkg.activate(environ.get("CCC_PATH"))

        # jlPkg.add("IntervalArithmetic")

        # activate CCC package
        newmodule(environ.get("CCC_PATH"))

        # jlPkg.resolve()

        jl.seval("using CCC")

    return jl
