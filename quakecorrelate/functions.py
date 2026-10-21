from obspy import Stream, Catalog
from typing import Literal, Dict, List
from pathlib import Path
import numpy as np

from .core import Correlator
from .core.correlation_file import CorrelationFile


def cross_correlate_seismic_events(
    stream_dict: Dict[str, Stream],
    catalog: Catalog,
    corr_file: str | Path = "./correlations.zarr",
    overwrite_corr_file: bool = True,
    method: Literal["time", "frequency"] = "time",
    platform: Literal["gpu", "cpu"] = "gpu",
    write_block_size: int = 1000,
    read_block_size: int = 100,
    stations: List[str] | None = None,
    channels: List[str] | None = None,
    length: float | None = None,
    padding: float = 0,
    shift_len: float = 1,
    plot_corr_matrix: bool | str | Path = False,
    export_to_dtcc: bool | str | Path = False,
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
        length:
            Length in seconds of the provided streams that will be used, measured from the start of the stream, plus any padding, if set.
        padding:
            Amount of padding in seconds taken from the stream at the start and the end.
        shift_len:
            Maximum amount in seconds by which the lag is allowed to vary.
        plot_corr_matrix:
            Determines whether to plot correlation matrices on the go. If a str or Path object is provided this will be assumed to the output directory for the correlation plots (defaults to 'correlation_plots').
        export_to_dtcc:
            Determines whether to export a dt.cc (differential time) file after finishing the correlation computations. If a str or Path object is provided this will be assumed to the output file name (defaults to 'dt.cc').
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

    # organise data
    if stations is None:
        stations = np.unique(
            np.concatenate(
                [
                    [trace.stats.station for trace in stream.traces]
                    for stream in stream_dict.values()
                ]
            )
        )
        # find all the unique sorted station ids among all the traces of all the streams

    if channels is None:
        channels = np.unique(
            np.concatenate(
                [
                    [trace.stats.channel for trace in stream.traces]
                    for stream in stream_dict.values()
                ]
            )
        )
        # find all the unique sorted channel ids among all the traces of all the streams

    correlator = Correlator(
        corr_file=corr_file,
        overwrite_corr_file=overwrite_corr_file,
        catalog=catalog,
        stations=stations,
        channels=channels,
        shardsize=write_block_size,
        chunksize=read_block_size,
    )

    correlator.cross_correlate(
        stream_dict,
        plot=plot_corr_matrix,
        length=length,
        padding=padding,
        shift_len=shift_len,
        plot_corr_matrix=True if plot_corr_matrix else False,
        out_dir=(
            Path(plot_corr_matrix)
            if isinstance(plot_corr_matrix, (str, Path))
            else None
        ),
    )

    if export_to_dtcc:
        correlator.correlation_file.to_dtcc(
            outfile=export_to_dtcc if isinstance(export_to_dtcc, (str, Path)) else None
        )

    return correlator.correlation_file
