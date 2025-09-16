from os import environ
from dotenv import load_dotenv
from typing import Literal
import numpy as np


from juliacall import Main as jl
from juliacall import newmodule
from juliacall import Pkg as jlPkg


load_dotenv()

# set julia environments
if environ.get("CCC_TYPE") == "path":
    # activate environment
    jlPkg.activate(environ.get("CCC_PATH"))

    # activate package
    newmodule(environ.get("CCC_PATH"))


jl.seval("using CCC")


def nnc(
    template: np.ndarray[tuple[int], float],
    signals: np.ndarray[tuple[int], float] | np.ndarray[tuple[int, int], float],
    return_corrgrams: bool = False,
    method: Literal["time", "frequency"] = "time",
    platform: Literal["cpu", "gpu"] = "cpu",
) -> (
    tuple[float, float]
    | tuple[np.ndarray[tuple[int], float], np.ndarray[tuple[int], float]]
    | tuple[
        np.ndarray[tuple[int], float],
        np.ndarray[tuple[int], float],
        np.ndarray[tuple[int, int], float],
    ]
):
    """Normalized cross correlation core function of CCC.

    Wraps CCC.jl's functions with the same name."""

    if signals.ndim == 1:
        result, lag = jl.NCC.ncc(template, signals, method, platform)

        return result, lag
    else:
        results = np.zeros(len(signals))
        lags = np.zeros(len(signals))

        if not return_corrgrams:
            results, lags = jl.NCC.ncc(
                template, signals, results, lags, method, platform
            )

            return results, lags
        else:
            corrgrams = np.zeros((len(signals), len(signals[0])))

            results, lags, corrgrams = jl.NCC.ncc(
                template, signals, results, lags, corrgrams, method, platform
            )

            return results, lags, corrgrams
