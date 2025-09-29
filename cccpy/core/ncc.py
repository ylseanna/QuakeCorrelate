
from typing import Literal
import numpy as np

from juliacall import Main as jl


def ncc(
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

    if np.array(signals).ndim == 1:
        result, lag = jl.CCC.ncc(
            template,
            signals,
            method=method,
            platform=platform,
        )

        return result, lag
    else:
        results = np.zeros(len(signals))
        lags = np.zeros(len(signals))

        if not return_corrgrams:
            results, lags = jl.CCC.ncc(
                template,
                signals,
                results,
                lags,
                method=method,
                platform=platform,
            )

            return results, lags
        else:
            corrgrams = np.zeros((len(signals), len(signals[0])))

            jl.CCC.ncc(
                template,
                signals,
                results,
                lags,
                corrgrams,
                method=method,
                platform=platform,
            )

            return results, lags, corrgrams
