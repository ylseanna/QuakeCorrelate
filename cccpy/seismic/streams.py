from obspy import Stream
from typing import Literal, Dict
from pathlib import Path
import numpy as np

from ..core.io import Correlations

def cross_correlate_seismic_events(
    stream_dict: Dict[str, Stream],
    outfile: str | Path | None = None,
    method: Literal["time", "frequency"] = "time",
    platform: Literal["gpu", "cpu"] = "gpu",
    stationfile: str | Path | None = None,
):
    """Function to calculate cross correlations for seismic events, on a per station, per component basis.

    Args:
        stream_dict (Dict[str, Stream]):
            A dictionary containing event streams keyed by their event id (string)
        outfile (str, Path, None, optional):
            Determines which correlation file to output your correlations to, if None is passed the correlations will return as a dictionary in-memory.
        method ([&quot;time&quot;, &quot;frequency&quot;], optional):
            Determines whether to cross correlate using time domain functions or frequency domain functions. Defaults to "time".
        platform ([&quot;gpu&quot;, &quot;cpu&quot;], optional):
            Whether to pass correlations to the gpu or cpu. Defaults to "gpu".
        stationfile (str, Path, None, optional):
            Station file can be passed to minimize pre-inventorizing which can get slow if many streams are provided.
    """

    # organise data
    if not stationfile:
        stations = np.unique(
            np.concatenate(
                [
                    [trace.stats.station for trace in stream.traces]
                    for eventid, stream in stream_dict.items()
                ]
            )
        )
        # find all the unique sorted station ids among all the traces of all the streams

    print(stations)
    
    Correlations(filename="correlation.sqlite", seismic_data=True)
