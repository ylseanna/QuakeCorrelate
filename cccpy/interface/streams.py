from obspy import Stream
from typing import Literal, Dict
from pathlib import Path

def cross_correlate_seismic_events(
    stream_dict: Dict[str, Stream],
    outfile: str | Path | None= None,
    method: Literal["time", "frequency"] = "time",
    platform: Literal["gpu", "cpu"] = "gpu",
):
    """Function to calculate cross correlations for seismic events, on a per station, per component basis.

    Args:
        stream_dict (Dict[str, Stream]):
            A dictionary containing event streams keyed by their event id (string)
        outfile (str, Path, None, optional):
            Determines which correlation file to output your correlations to, if None is passed the correlations will return as a dictionary in-memory.
        method ([&quot;time&quot;, &quot;frequency&quot;], optional):
            Determines whether to cross correlate using time domain functions or frequency domain functions. Defaults to "time".
        platform ([&quot;gpu&quot;, &quot;cpu&quot;], optional): Whether to pass correlations to the gpu or cpu. Defaults to "gpu".
    """

    pass
