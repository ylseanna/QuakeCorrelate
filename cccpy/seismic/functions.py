from obspy import Stream
from typing import Literal, Dict, List
from pathlib import Path
import numpy as np

from .seismic_correlator import SeismicCorrelator


def cross_correlate_seismic_events(
    stream_dict: Dict[str, Stream],
    corr_file: str | Path | None = None,
    method: Literal["time", "frequency"] = "time",
    platform: Literal["gpu", "cpu"] = "gpu",
    stationfile: str | Path | None = None,
    channels: List[str] | None = None,
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
            Station file can be passed to minimize pre-inventorizing - which can get slow if many streams are provided - or filter which channels to process.
        channels (List[str]), None, optional):
            Channel names (e.g. ["EHN", "EHE", "EHZ]) can be passed to minimize pre-inventorizing - which can get slow if many streams are provided - or filter which channels to process.
    """
    # manage args
    if corr_file:
        corr_file = Path(corr_file)
    else:
        corr_file = Path("./correlation.sqlite")

    # organise data
    if not stationfile:
        stations = np.unique(
            np.concatenate(
                [
                    [trace.stats.station for trace in stream.traces]
                    for stream in stream_dict.values()
                ]
            )
        )
        # find all the unique sorted station ids among all the traces of all the streams

    if not channels:
        channels = np.unique(
            np.concatenate(
                [
                    [trace.stats.channel for trace in stream.traces]
                    for stream in stream_dict.values()
                ]
            )
        )
        # find all the unique sorted channel ids among all the traces of all the streams

    correlator = SeismicCorrelator(
        corr_file=corr_file,
        overwrite_corr_file=True,
        stations=stations,
        channels=channels,
    )

    correlator.cross_correlate(stream_dict)

    # stations
    print(correlator.correlation_file.get_stations(include_channels=True, as_df=False))
