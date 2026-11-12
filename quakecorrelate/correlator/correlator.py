import logging
import time
from itertools import chain
from math import ceil
from os import environ
from pathlib import Path
from typing import Dict, List, Tuple, Literal

import numpy as np
from matplotlib import use as mpl_use
from obspy import Catalog, Stream
from dotenv import load_dotenv

from ..util.helper_functions import get_chunks_to_process, next_power_of_2, to_length
from ..correlation_file import CorrelationFile
from .functions import extract_traces

mpl_use("Agg")
from matplotlib.pyplot import close, figure

# from scipy.interpolate import CubicSpline

logger = logging.getLogger(__name__)


class Correlator:
    """Utility class for running the seismic cross correlation, and some related functions, such as plotting the calculated correlations.

    Args:
        corr_file: Correlation file to be used by the correlator to write the correlations to, will generate a new one if it is not present.
        overwrite_corr_file: Whether to empty any existing correlation file, i.e. run a clean slate run.
        shardsize: Size of the 'shards' in the data matrices. Corresponds to the size of the matrices that can be written independently.
        chunksize: Size of the 'chunks' in the data matrices. Corresponds to the size of the matrices that can be read independently.

    """

    def __init__(
        self,
        corr_file: str | Path = Path("correlations.zarr"),
        overwrite_corr_file: bool = False,
        shardsize: int = 1000,
        chunksize: int = 100,
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
    ):
        # ensuring input type
        corr_file = Path(corr_file)

        logger.info("Initialising Seismic Correlator")

        # assigning variables
        self.shardsize = shardsize
        self.chunksize = chunksize
        self.jl = None

        # set/get .zarr file
        logger.info(
            f"Creating .zar directory at {corr_file}, overwriting?: {overwrite_corr_file}"
        )

        self.correlation_file = CorrelationFile(
            filename=corr_file,
            overwrite_file=overwrite_corr_file,
            shardsize=shardsize,
            chunksize=chunksize,
            horizontal_channels_postfix=horizontal_channels_postfix,
            vertical_channels_postfix=vertical_channels_postfix,
        )

    def cross_correlate(
        self,
        stream_dict: Dict[str, Stream],
        catalog: Catalog | None = None,
        stations: List[str] | Literal["all"] | None = None,
        channels: List[str] | Literal["all"] | None = None,
        length: float | None = None,
        padding: float = 0.0,
        shift_len: float = 1.0,
        interpolate: int | None = None,
        threads_per_block: int = 256,
        lower_triangle: bool = True,
        plot_corr_matrix: bool = False,
        plot_dir: str | Path = "correlation_plots",
        plot_filename: str | Path = "correlation.png",
    ):
        """_summary_

        Args:
            stream_dict: Dictionary keyed by Event ID with the Obspy Stream object that contain the data for each station and phase to be correlated.
            catalog: Obspy catalog whose events are to be processed. Will be written to the correlation file metadata. Setting to :code:`None` will cause the metadata to remain untouched (useful if previously set, but will cause errors otherwise).
            stations: List of station IDs to be processed, or "all" to dynamically include all station IDs present in the streams. Will be written to the correlation file metadata. Setting to :code:`None` will cause the metadata to remain untouched (useful if previously set, but will cause errors otherwise).
            channels: List of channels IDs to be processed, or "all" to dynamically include all channel IDs present in the streams. Will be written to the correlation file metadata. Setting to :code:`None` will cause the metadata to remain untouched (useful if previously set, but will cause errors otherwise).
            length: **TO BE CHANGED**
            padding: **TO BE CHANGED**
            shift_len: **TO BE CHANGED**
            interpolate: **TO BE CHANGED**
            threads_per_block: **TO BE CHANGED**
            lower_triangle: **TO BE CHANGED**
            plot_corr_matrix: Whether to plot correlation matrices while the calculations are running.
            plot_dir:  Output directory for the plots.
            plot_filename: Filename for the correlation matrix plots, will append station and channel information.
        """
        ### metadata

        if catalog is not None:
            logger.debug(f"Writing catalog metadata {catalog}")
            self.correlation_file.metadata.write_catalog(catalog)

        # input stations into .zarr file
        if stations is not None:
            if stations == "all":
                stations = np.unique(
                    np.concatenate(
                        [
                            [trace.stats.station for trace in stream.traces]
                            for stream in stream_dict.values()
                        ]
                    )
                )
                logger.debug(
                    f"Writing stations to metadata for all stations present {stations}"
                )
                self.correlation_file.metadata.write_stations(stations)
            else:
                logger.debug(f"Writing stations to metadata {stations}")
                self.correlation_file.metadata.write_stations(stations)

        # input channels into .zarr file
        if channels is not None:
            if channels == "all":
                channels = np.unique(
                    np.concatenate(
                        [
                            [trace.stats.channel for trace in stream.traces]
                            for stream in stream_dict.values()
                        ]
                    )
                )
                logger.debug(
                    f"Writing channels to metadata for all channels present {channels}"
                )
                self.correlation_file.metadata.write_channels(channels)
            else:
                logger.debug(f"Writing channels to metadata {channels}")
                self.correlation_file.metadata.write_channels(channels)

        logger.info(
            f"--- Correlating all events for the selected correlation file and with streams provided ---"
        )

        plot_dir = Path(plot_dir)
        plot_filename = Path(plot_filename)

        # write streams
        logger.debug("Writing stream metadata")
        self.correlation_file.metadata.write_events(list(stream_dict.keys()))

        # get event data
        events = self.correlation_file.metadata.events

        all_event_indices = [int(index) for index in events.keys()]

        # chunk (also for serial processing), according to shard size (writing blocksize, whereas chunks are for reading)
        max_event_index = all_event_indices[-1]

        chunks_to_process = get_chunks_to_process(
            max_event_index, self.shardsize, lower_triangle
        )

        logger.info(f"    # chunks to process: {len(chunks_to_process)}")

        for primary_event_slice, secondary_event_slice in chunks_to_process:
            logger.info(
                f"*** Correlating event indices {primary_event_slice[0]} to {primary_event_slice[1]} against {secondary_event_slice[0]} to {secondary_event_slice[1]} ***"
            )

            self._cross_correlate_slice(
                stream_dict=stream_dict,
                primary_event_slice=primary_event_slice,
                secondary_event_slice=secondary_event_slice,
                # length=length,
                shift_len=shift_len,
                padding=padding,
                interpolate=interpolate,
                threads_per_block=threads_per_block,
                plot=plot_corr_matrix,
                plot_dir=plot_dir,
                plot_filename=plot_filename,
            )

    def _cross_correlate_slice(
        self,
        stream_dict: Dict[str, Stream],
        primary_event_slice: Tuple[int, int],
        secondary_event_slice: Tuple[int, int],
        prepick: float = 0.5,
        postpick: float = 1.5,
        shift_len: float = 1.0,
        padding: float | None = None,
        interpolate: int | None = None,
        threads_per_block: int = 256,
        plot: bool = False,
        plot_dir: str | Path = "correlation_plots",
        plot_filename: str | Path = "correlation.png",
    ):
        # get event data
        events = self.correlation_file.metadata.events

        # get julia
        if self.jl is None:
            from ..interface import JuliaInterface

            logger.info("    Accessing Julia")
            self.jl = JuliaInterface()
        else:
            logger.info("    Using existing Julia instance")

        # get slices
        logger.info("    Slicing by station and channel")
        per_station_per_channel_slices = (
            self.correlation_file._slice_traces_by_station_and_channel(
                primary_event_slice=primary_event_slice,
                secondary_event_slice=secondary_event_slice,
            )
        )

        lower_triangle = primary_event_slice == secondary_event_slice

        for station, per_channel_slice in per_station_per_channel_slices.items():
            for channel, event_indices in per_channel_slice.items():
                logger.info(
                    f"--- Correlating for station {station}, channel {channel} ---"
                )

                logger.info("        Extracting traces...")

                # get traces and event_indices and associated picks
                primary_traces_sliced = []
                primary_event_indices_sliced = []

                # determine phase type to use for extracting picks
                if channel.endswith(self.correlation_file.vertical_channels_postfix):
                    logger.debug("Vertical channel detected!")
                    phase_hint = "P"
                elif channel.endswith(
                    self.correlation_file.horizontal_channels_postfix
                ):
                    logger.debug("Horizontal channel detected!")
                    phase_hint = "S"
                else:
                    logger.warning(
                        "Channel type (phase hint) could not be detected, defaulting to P phase!"
                    )
                    phase_hint = "P"

                logger.info(f"             phase hint: {phase_hint:1s} (pick used)")

                if lower_triangle:
                    # primary
                    primary_event_indices = event_indices

                    if len(primary_event_indices) == 0:
                        logger.warning(
                            "No events for this slice of the matric, skipping!"
                        )
                        continue

                    primary_event_ids = [
                        self.correlation_file.metadata.events[str(event_index)]
                        for event_index in primary_event_indices
                    ]

                    A, primary_trace_dict = extract_traces(
                        stream_dict,
                        primary_event_ids,
                        self.correlation_file.metadata.catalog,
                        station,
                        channel,
                        phase_hint,
                        prepick,
                        postpick,
                        padding=(
                            shift_len if not padding else padding
                        ),  # default to shift_len else set to preset padding
                    )

                    # secondary kept for consistency
                    secondary_event_indices = []
                    B = []
                    secondary_trace_dict = {}

                    logger.info(f"             array shape: {A.shape}")

                    # retrieve samping rate for shift_len
                    sampling_rate = primary_trace_dict[
                        primary_event_ids[0]
                    ].stats.sampling_rate

                    ### do correlation and time it

                    shift_len_samples = next_power_of_2(
                        int(ceil(shift_len * sampling_rate))
                    )

                    logger.info(
                        f"            shift length: {shift_len_samples:<11n} samples (ceiled to nearest power of 2, {shift_len:.1f} s)"
                    )

                    ### transposing and converting data

                    A = self.jl.pyconvert(
                        self.jl.Matrix, A.T
                    )  # ensure column indexing (for julia)

                    ### actual cross correlation

                    logger.info("        Correlating on GPU...")

                    pre_corr_time = time.time()

                    # NOTE: reduced: lower triangle representation
                    # run correlation
                    corrgrams_gpu_reduced = self.jl.Squiggles.norm_correlogram(
                        A, shift_len_samples, threads_per_block
                    )

                    # calculate lags
                    xcorcoeff_GPU_reduced, lags_GPU_reduced = (
                        self.jl.Squiggles.simplelags(
                            corrgrams_gpu_reduced, shift_len_samples
                        )
                    )

                    # move to CPU from GPU
                    xcorcoeff_reduced = self.jl.Squiggles.memcopy(xcorcoeff_GPU_reduced)
                    lags_reduced = self.jl.Squiggles.memcopy(lags_GPU_reduced)

                    # reconstruct full matrix from reduced triangular representation
                    xcorcoeff = self.jl.Squiggles.reconstruct_symmetric(
                        xcorcoeff_reduced, fillwithnans=False
                    )
                    lags = self.jl.Squiggles.reconstruct_antisymmetric(lags_reduced)
                if not lower_triangle:
                    primary_event_indices, secondary_event_indices = event_indices

                    if len(primary_event_indices) == 0:
                        logger.warning(
                            "No primary events for this slice of the matric, skipping!"
                        )
                        continue

                    if len(secondary_event_indices) == 0:
                        logger.warning(
                            "No secondary events for this slice of the matric, skipping!"
                        )
                        continue

                    # primary
                    primary_event_ids = [
                        self.correlation_file.metadata.events[str(event_index)]
                        for event_index in primary_event_indices
                    ]

                    A, primary_trace_dict = extract_traces(
                        stream_dict,
                        primary_event_ids,
                        self.correlation_file.metadata.catalog,
                        station,
                        channel,
                        phase_hint,
                        prepick,
                        postpick,
                        padding=(
                            shift_len if not padding else padding
                        ),  # default to shift_len else set to preset padding
                    )

                    # secondary
                    secondary_event_ids = [
                        self.correlation_file.metadata.events[str(event_index)]
                        for event_index in secondary_event_indices
                    ]

                    B, secondary_trace_dict = extract_traces(
                        stream_dict,
                        secondary_event_ids,
                        self.correlation_file.metadata.catalog,
                        station,
                        channel,
                        phase_hint,
                        prepick,
                        postpick,
                        padding=(
                            shift_len if not padding else padding
                        ),  # default to shift_len else set to preset padding
                    )

                    logger.info(f"             array shape: {A.shape} (A)")
                    logger.info(f"             array shape: {B.shape} (B)")

                    # retrieve samping rate for shift_len
                    sampling_rate = primary_trace_dict[
                        primary_event_ids[0]
                    ].stats.sampling_rate

                    ### do correlation and time it

                    shift_len_samples = next_power_of_2(
                        int(ceil(shift_len * sampling_rate))
                    )

                    logger.info(
                        f"            shift length: {shift_len_samples:<11n} samples (ceiled to nearest power of 2, {shift_len:.1f} s)"
                    )

                    ### transposing and converting data

                    A = self.jl.pyconvert(
                        self.jl.Matrix, A.T
                    )  # ensure column indexing (for julia)

                    B = self.jl.pyconvert(
                        self.jl.Matrix, B.T
                    )  # ensure column indexing (for julia)

                    ### actual cross correlation

                    logger.info("        Correlating on GPU...")

                    pre_corr_time = time.time()

                    # NOTE: no reduction as it is a rectangular computation matrix
                    # run correlation
                    corrgrams_gpu = self.jl.Squiggles.norm_correlogram(
                        A, B, shift_len_samples, threads_per_block
                    )

                    # calculate lags
                    xcorcoeff_GPU, lags_GPU = self.jl.Squiggles.simplelags(
                        corrgrams_gpu, shift_len_samples
                    )

                    # move to CPU from GPU
                    xcorcoeff = self.jl.Squiggles.memcopy(xcorcoeff_GPU)
                    lags = self.jl.Squiggles.memcopy(lags_GPU)

                ### process products

                # cast to numpy array (back to python land yay)
                corr_matrix = np.array(xcorcoeff, dtype=np.float32)
                lag_matrix = -np.array(
                    lags, dtype=np.float32
                )  # NOTE: Squiggles varies A over B, while QuakeCorrelate assumes B is run over A, so lags have to be -1

                # convert lags from samples to seconds
                lag_matrix *= float(1 / sampling_rate)

                post_corr_time = time.time()

                logger.info(
                    f"        Correlation complete!         (in {post_corr_time - pre_corr_time:.3g} s)"
                )

                # sparsify the matrix for saving in zarr (enables better chunking and later processing)
                num_primary_events = primary_event_slice[1] - primary_event_slice[0] + 1
                num_secondary_events = (
                    secondary_event_slice[1] - secondary_event_slice[0] + 1
                )

                sparse_corr_matrix = np.empty(
                    (num_primary_events, num_secondary_events)
                )
                sparse_lag_matrix = np.empty((num_primary_events, num_secondary_events))

                sparse_corr_matrix[:] = np.nan
                sparse_lag_matrix[:] = np.nan

                # NOTE: also check if value is > shift_len, NaN if so
                if lower_triangle:
                    for i, event_id_i in enumerate(event_indices):
                        for j, event_id_j in enumerate(event_indices):
                            if j < i:
                                sparse_corr_matrix[
                                    event_id_i - primary_event_slice[0],
                                    event_id_j - primary_event_slice[0],
                                ] = corr_matrix[i, j]
                                sparse_lag_matrix[
                                    event_id_i - primary_event_slice[0],
                                    event_id_j - primary_event_slice[0],
                                ] = lag_matrix[i, j]
                else:
                    for i, event_id_i in enumerate(primary_event_indices):
                        for j, event_id_j in enumerate(secondary_event_indices):
                            sparse_corr_matrix[
                                event_id_i - primary_event_slice[0],
                                event_id_j - secondary_event_slice[0],
                            ] = corr_matrix[i, j]
                            sparse_lag_matrix[
                                event_id_i - primary_event_slice[0],
                                event_id_j - secondary_event_slice[0],
                            ] = lag_matrix[i, j]

                ### write

                self.correlation_file.write_correlation_matrix(
                    sparse_corr_matrix,
                    station=station,
                    channel=channel,
                    primary_event_slice=primary_event_slice,
                    secondary_event_slice=secondary_event_slice,
                )

                self.correlation_file.write_lag_matrix(
                    sparse_lag_matrix,
                    station=station,
                    channel=channel,
                    primary_event_slice=primary_event_slice,
                    secondary_event_slice=secondary_event_slice,
                )

                logger.info("        Correlations written to disk...")

                if plot:
                    self.plot_corr_matrix(
                        station=station,
                        channel=channel,
                        out_dir=plot_dir,
                        out_filename=plot_filename,
                    )
                    logger.info("    plotted cross correlation matrix!")

    def plot_corr_matrix(
        self,
        station: str,
        channel: str,
        indices: None | List[int] = None,
        out_dir: str | Path = "correlation_plots",
        out_filename: str | Path = "correlation.png",
        slice_stepwise: int | None = None,
        normalized: bool = True,
    ):
        """Utitility function for plotting correlation matrices.

        .. admonition :: Note from the developer

            Subject to change as it is currently not aimed at general use. I typically use my own plotting code, which is based on the one shown below.

        Args:
            station: station to plot
            channel: channel to plot
            indices: Indices of the square matrix to plot.
            plot_dir:  Output directory for the plots.
            plot_filename: Filename for the correlation matrix plots, will append station and channel information.
            slice_stepwise: The slicing step for plotting
            normalized: Whether to adjust the colourmap to the data, or to just set to -1 to 1.
        """
        # ensure types
        out_dir = Path(out_dir)
        out_filename = Path(out_filename)

        # make sure directory exists
        out_dir.mkdir(exist_ok=True, parents=True)

        # get data
        if not slice_stepwise:
            corr_matrix = self.correlation_file[station][channel].corr_values
        else:
            corr_matrix = self.correlation_file[station][channel].corr_values[
                ::slice_stepwise
            ]

        # modify data
        if indices:
            corr_matrix = corr_matrix[indices, :][:, indices]

        if slice_stepwise:
            corr_matrix = corr_matrix[::slice_stepwise, ::slice_stepwise]

        # plot
        fig = figure(figsize=(15, 15), layout="constrained")

        ax = fig.add_subplot()

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        if normalized:
            crmat = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu")
        else:
            max_abs = np.max(np.abs(corr_matrix))

            crmat = ax.imshow(corr_matrix, vmin=-max_abs, vmax=max_abs, cmap="RdBu")

        cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
        cbar.ax.set_ylabel("correlation value")

        ax.set_title(f"correlations, station: {station}, channel: {channel}")

        ax.set_xlabel("event number")
        ax.set_ylabel("event number")

        fig.savefig(
            out_dir
            / (out_filename.stem + f"_{station}_{channel}" + out_filename.suffix)
        )
        del corr_matrix

        close()
