from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np
from obspy import Catalog, Stream

from .correlation_file import CorrelationFile
from .correlator import Correlator


def cross_correlate_seismic_events(
    stream_dict: Dict[str, Stream],
    catalog: Catalog,
    corr_file: str | Path = "./correlations.zarr",
    overwrite_corr_file: bool = True,
    method: Literal["time", "frequency"] = "time",
    platform: Literal["gpu", "cpu"] = "gpu",
    write_block_size: int = 1000,
    read_block_size: int = 100,
    stations: List[str] | Literal["all"] = "all",
    channels: List[str] | Literal["all"] = "all",
    prepick: float = 0.5,
    postpick: float = 1.5,
    length: float | None = None,
    padding: float = 0,
    shift_len: float = 1,
    plot_corr_matrix: bool | str | Path = False,
    export_to_dtcc: bool | str | Path = False,
    event_id_mapping: Dict[str, str] | Literal["index"] | None = None,
    min_corr_value: float = 0,
    horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
    vertical_channels_postfix: Tuple[str] = ("Z", "3"),
) -> CorrelationFile:
    """Function to calculate cross correlations for seismic events, on a per station, per component basis.

    Args:
        stream_dict:
            A dictionary containing event streams keyed by their event id.
        catalog:
            Obspy catalog containing (at least) all the events that will be correlated (may contain more).
        corr_file:
            The path to the .zarr file the correlations will be written to.
        overwrite_corr_file:
            Determines whether to reset the .zarr file when running this function. As this function is assumed to be run indepenently it is set to **overwrite** by default, unset to keep previous results.
        method:
            Determines whether to cross correlate using time domain functions or frequency domain functions.
        platform:
            Whether to pass correlations to the gpu or cpu.
        write_block_size:
            Determines the size of the chunks in the correlation file that also determine the chunks of the computation matrix that are computed in one go (analagous to shard_size for the corelation file).
        read_block_size:
            Determines the size of the chunks in the correlation file that also determine the chunks that are read into memory in one go when exporting (analagous to chunk_size for the corelation file).
        stations:
            When provided only these station IDs will be used for the correlation. If unset all station IDs present in the streams will be detected and used.
        channels:
            When provided only these channel IDs will be used for the correlation. If unset all channel IDs present in the streams will be detected and used.
        prepick:
            Number of seconds prior to the pick time that should be extracted from the original Trace data when extracting the data from the streams.
        postpick:
            Number of seconds after the pick time that should be extracted from the original Trace data when extracting the data from the streams. Overwritten by length if length is set.
        length:
            Number of seconds after prepick to cut out (i.e. total length of the extracted trace) when extracting the data from the streams, overrides postpick if not None.
        padding:
            Amount of padding in seconds taken from the stream at the start and the end.
        shift_len:
            Maximum amount in seconds by which the lag is allowed to vary.
        plot_corr_matrix:
            Determines whether to plot correlation matrices on the go. If a str or Path object is provided this will be assumed to the output directory for the correlation plots (defaults to 'correlation_plots').
        export_to_dtcc:
            Determines whether to export a dt.cc (differential time) file after finishing the correlation computations. If a str or Path object is provided this will be assumed to the output file name (defaults to 'dt.cc').
        event_id_mapping:
            Event ID mapping used for exporting the dt.cc file, only used if *export_to_dtcc* is set to True.
        min_corr_value:
            Minimum correlation value that will be exported to the dt.cc file, only used if *export_to_dtcc* is set to True.
    NOTE:
        The shift_len will be increased to a 2^n amount of samples, at the sampling rate of the streams provided. Due to a current limitation of the code.
    Returns:
        CorrelationFile
            The correlation file object  that the correlations were written to.
    """
    # manage args
    corr_file = Path(corr_file)

    if method != "time":
        raise NotImplementedError(
            "Other methods than the time domain cross correlation are not yet implemented."
        )

    if platform != "gpu":
        raise NotImplementedError(
            "No cpu-based cross correlations have been implemented yet."
        )

    correlator = Correlator(
        corr_file=corr_file,
        overwrite_corr_file=overwrite_corr_file,
        shardsize=write_block_size,
        chunksize=read_block_size,
        horizontal_channels_postfix=horizontal_channels_postfix,
        vertical_channels_postfix=vertical_channels_postfix,
    )

    correlator.cross_correlate(
        stream_dict,
        catalog=catalog,
        stations="all" if not stations else stations,
        channels="all" if not channels else channels,
        prepick=prepick,
        postpick=postpick,
        length=length,
        padding=padding,
        shift_len=shift_len,
        plot_corr_matrix=True if plot_corr_matrix else False,
        plot_dir=(
            Path(plot_corr_matrix)
            if isinstance(plot_corr_matrix, (str, Path))
            else "correlation_plots"
        ),
    )

    if export_to_dtcc:
        correlator.correlation_file.to_dtcc(
            outfile=(
                export_to_dtcc if isinstance(export_to_dtcc, (str, Path)) else "dt.cc"
            ),
            event_id_mapping=event_id_mapping,
            min_corr_value=min_corr_value,
        )

    return correlator.correlation_file
