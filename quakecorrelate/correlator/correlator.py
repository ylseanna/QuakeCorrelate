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
            stream_dict: _description_
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
                length=length,
                padding=padding,
                shift_len=shift_len,
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
        length: float | None = None,
        padding: float = 0.0,
        shift_len: float = 1.0,
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

                if lower_triangle:
                    ### get all traces from which we need to take the data (sort by channel, station and all event indices in slice)
                    primary_traces_sliced = [
                        (
                            [
                                trace
                                for trace in stream
                                if trace.stats.channel == channel
                                and trace.stats.station == station
                            ][0]
                            if [
                                trace
                                for trace in stream
                                if trace.stats.channel == channel
                                and trace.stats.station == station
                            ]
                            else []
                        )
                        for stream in [
                            stream_dict[event_id]
                            for event_index, event_id in events.items()
                            if int(event_index) in event_indices
                        ]
                    ]

                    logger.info(
                        f"        number of traces: {len(primary_traces_sliced):<11n}"
                    )

                    # empty secondary so the code can be common for both A = A and A = B
                    secondary_traces_sliced = []
                elif not lower_triangle:
                    # now the event_indices variable is a tuple with two lists
                    primary_event_indices, secondary_event_indices = event_indices

                    ### get all traces from which we need to take the data (sort by channel, station and all event indices in slice)
                    primary_traces_sliced = [
                        (
                            [
                                trace
                                for trace in stream
                                if trace.stats.channel == channel
                                and trace.stats.station == station
                            ][0]
                            if [
                                trace
                                for trace in stream
                                if trace.stats.channel == channel
                                and trace.stats.station == station
                            ]
                            else []
                        )
                        for stream in [
                            stream_dict[event_id]
                            for event_index, event_id in events.items()
                            if int(event_index) in primary_event_indices
                        ]
                    ]

                    # empty secondary so the code can be common for both A = A and A = B
                    secondary_traces_sliced = [
                        (
                            [
                                trace
                                for trace in stream
                                if trace.stats.channel == channel
                                and trace.stats.station == station
                            ][0]
                            if [
                                trace
                                for trace in stream
                                if trace.stats.channel == channel
                                and trace.stats.station == station
                            ]
                            else []
                        )
                        for stream in [
                            stream_dict[event_id]
                            for event_index, event_id in events.items()
                            if int(event_index) in secondary_event_indices
                        ]
                    ]

                    logger.info(
                        f"        number of traces: {len(primary_traces_sliced):<11n} + {len(secondary_traces_sliced):<11n} (A, B)"
                    )

                    if (
                        len(primary_traces_sliced) == 0
                        or len(secondary_traces_sliced) == 0
                    ):
                        logger.warning(
                            "One side of the computation array has no traces - SKIPPING..."
                        )
                        continue

                if any(
                    [
                        False if len(trace) > 0 else True
                        for trace in chain(
                            primary_traces_sliced, secondary_traces_sliced
                        )
                    ]
                ):
                    logger.warning("Zero length traces found - SKIPPING...")
                    continue

                logger.info("        Collecting trace metadata...")

                ### calculate relevant values for processing based on input arguments

                # get required metadata
                sampling_rates = [
                    trace.stats.sampling_rate
                    for trace in chain(primary_traces_sliced, secondary_traces_sliced)
                ]
                nptss = [
                    trace.stats.npts
                    for trace in chain(primary_traces_sliced, secondary_traces_sliced)
                ]
                trace_lengths = [
                    trace.stats.npts * trace.stats.delta
                    for trace in chain(primary_traces_sliced, secondary_traces_sliced)
                ]

                # trace lengths
                unique_trace_lengths, unique_trace_lengths_counts = np.unique(
                    trace_lengths, return_counts=True
                )

                if len(unique_trace_lengths) > 1:
                    logger.warning(
                        "Inhomogeneous trace lengths, continuing with most frequent, but be aware..."
                    )
                    trace_length = unique_trace_lengths[
                        np.argmax(unique_trace_lengths_counts)
                    ]
                else:
                    trace_length = unique_trace_lengths[0]

                logger.info(f"            trace length: {trace_length:<11.1f} s")

                # sampling rate
                unique_sampling_rates = np.unique(sampling_rates)

                if len(unique_sampling_rates) > 1:
                    logger.warning("Inhomogeneous sampling rates, skipping...")
                    continue

                sampling_rate = unique_sampling_rates[0]

                logger.info(f"           sampling rate: {sampling_rate:<11.1f} Hz")

                # total length of all samples (can accomodate a few samples offset)
                unique_nptss, unique_nptss_counts = np.unique(nptss, return_counts=True)

                if len(unique_nptss) > 1:
                    npts = unique_nptss[np.argmax(unique_nptss_counts)]
                else:
                    npts = unique_nptss[0]

                logger.info(f"       number of samples: {npts:<11.1f}")

                # slicing

                if length or padding:
                    logger.info("        Slicing data...")

                    if length:
                        logger.info(f"        length to slice: {trace_length:<11.1f} s")

                    if padding:
                        logger.info(f"                padding: {trace_length:<11.1f} s")

                    # num samples and start and end of slicing

                    # if padding, start not at zero
                    start = int(padding * sampling_rate)

                    # if fixed template length is set, calculate length
                    if length:
                        end = int(start + length * sampling_rate)
                    else:
                        end = int(
                            npts - padding * sampling_rate
                        )  # go to end, unless padding

                    num_samples = end - start

                    logger.info(
                        f"       number of samples: {num_samples:<11n} (sliced from {npts}, start: {start}, end: {end})"
                    )
                else:
                    num_samples = npts

                ### get data

                logger.info("        Preparing data...")

                # data input matrices
                A = []
                B = []

                # extract data from trace
                for trace in primary_traces_sliced:
                    if padding or length:
                        trace_data = trace.data[start:end]
                    else:
                        trace_data = trace.data

                    if (
                        len(trace_data) != num_samples
                    ):  # sometimes a single trace is of by a few samples (or improper length is set)
                        logger.warning(
                            "One trace did not have the right number of samples (%s samples) for the requested length (%s samples), padding, but be aware",
                            len(trace_data),
                            num_samples,
                        )

                    A.append(trace_data)

                if not lower_triangle:
                    for trace in secondary_traces_sliced:
                        if padding or length:
                            trace_data = trace.data[start:end]
                        else:
                            trace_data = trace.data

                        if (
                            len(trace_data) != num_samples
                        ):  # sometimes a single trace is of by a few samples (or improper length is set)
                            logger.warning(
                                "One trace did not have the right number of samples (%s samples) for the requested length (%s samples), padding, but be aware",
                                len(trace_data),
                                num_samples,
                            )

                        B.append(trace_data)

                ### pad signal

                padded_npts = next_power_of_2(int(num_samples))

                A = np.array([to_length(trace, padded_npts) for trace in A])
                B = np.array([to_length(trace, padded_npts) for trace in B])

                logger.info(
                    f"           signal length: {padded_npts:<11n} samples (zero-padded to nearest power of 2, {padded_npts / sampling_rate:.1f} s)"
                )

                ### do correlation and time it

                shift_len_samples = next_power_of_2(
                    int(ceil(shift_len * sampling_rate))
                )

                logger.info(
                    f"            shift length: {shift_len_samples:<11n} samples (ceiled to nearest power of 2, {shift_len:.1f} s)"
                )

                if lower_triangle:
                    logger.info(f"             array shape: {A.shape}")
                if not lower_triangle:
                    logger.info(f"             array shape: {A.shape} (A)")
                    logger.info(f"             array shape: {B.shape} (B)")

                ### transposing and converting data

                A = self.jl.pyconvert(
                    self.jl.Matrix, A.T
                )  # ensure column indexing (for julia)

                if not lower_triangle:
                    B = self.jl.pyconvert(
                        self.jl.Matrix, B.T
                    )  # ensure column indexing (for julia)

                ### actual cross correlation

                logger.info("        Correlating on GPU...")

                pre_corr_time = time.time()

                if lower_triangle:  # NOTE: reduced: lower triangle representation
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
                elif (
                    not lower_triangle
                ):  # NOTE: no reduction as it is a rectangular computation matrix
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

                # cast to numpy array (back to python land yay)
                corr_matrix = np.array(
                    xcorcoeff, dtype=np.float32
                )  # convert julia array and revert to row indexing
                lag_matrix = np.array(
                    lags, dtype=np.float32
                )  # convert julia array and revert to row indexing

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
