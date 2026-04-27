import logging
import time
from math import ceil
from pathlib import Path
from typing import List, Dict

import numpy as np
from matplotlib import use as mpl_use
from obspy import Catalog, Stream

from ..util.helper_functions import next_power_of_2, to_length
from .interface import JuliaInterface
from .correlation_file import CorrelationFile

mpl_use("Agg")
from matplotlib.pyplot import close, figure

# from scipy.interpolate import CubicSpline


logger = logging.getLogger(__name__)

type EventID = str


class Correlator:
    "Correlator Class"

    def __init__(
        self,
        corr_file: str | Path = Path("correlations.zarr"),
        overwrite_corr_file: bool = False,
        catalog: Catalog | None = None,
        stations: List[str] | None = None,
        channels: List[str] | None = None,
    ):
        logger.info("Initialising Seismic Correlator")
        corr_file = Path(corr_file)

        logger.info(
            f"Creating .zar directory at {corr_file}, overwriting?: {overwrite_corr_file}"
        )
        self.correlation_file = CorrelationFile(
            filename=corr_file, overwrite_file=overwrite_corr_file
        )

        if catalog is not None:
            logger.debug(f"Writing catalog metadata {catalog}")
            self.correlation_file.write_catalog(catalog)

        if stations is not None:
            logger.debug(f"Writing station metadata {stations}")
            self.correlation_file.write_stations(stations)

        if channels is not None:
            logger.debug(f"Writing channel metadata {channels}")
            self.correlation_file.write_channels(channels)

    def cross_correlate(
        self,
        stream_dict: Dict[EventID, Stream],
        length: float | None = None,
        padding: float = 0.0,
        shift_len: float = 1.0,
        interpolate: int | None = None,
        threads_per_block: int = 256,
        plot: bool = False,
        out_dir: str | Path = "correlation_plots",
        out_filename: str | Path = "correlation.png",
    ):
        out_dir = Path(out_dir)
        out_filename = Path(out_filename)

        # get julia
        logger.info("Accessing Julia")
        jl = JuliaInterface()

        # write streams
        logger.debug("Writing stream metadata")
        self.correlation_file.write_events(list(stream_dict.keys()))

        # get event data
        events = self.correlation_file.get_events(as_df=False)

        # get slices
        logger.info("Slicing by station and channel")
        per_station_per_channel_slices = (
            self.correlation_file.slice_traces_by_station_and_channel()
        )

        # NOTE: ceil to power of 2 for shift_len

        for station, per_channel_slice in per_station_per_channel_slices.items():
            for channel, event_indices in per_channel_slice.items():
                logger.info(
                    f"--- Correlating for station {station}, channel {channel} ---"
                )

                ### get all traces from which we need to take the data (sort by channel, station and all event indices in slice)
                traces_sliced = [
                    [
                        trace
                        for trace in stream
                        if trace.stats.channel == channel
                        and trace.stats.station == station
                    ][0]
                    for stream in [
                        stream_dict[event_id]
                        for event_index, event_id in events.items()
                        if int(event_index) in event_indices
                    ]
                ]

                logger.info(f"    number of traces: {len(traces_sliced):<11n}")

                logger.info("    Collecting trace metadata...")

                ### calculate relevant values for processing based on input arguments

                # get required metadata
                sampling_rates = [trace.stats.sampling_rate for trace in traces_sliced]
                nptss = [trace.stats.npts for trace in traces_sliced]
                trace_lengths = [
                    trace.stats.npts * trace.stats.delta for trace in traces_sliced
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

                logger.info(f"        trace length: {trace_length:<11.1f} s")

                # sampling rate
                unique_sampling_rates = np.unique(sampling_rates)

                if len(unique_sampling_rates) > 1:
                    logger.warning("Inhomogeneous sampling rates, skipping...")
                    continue

                sampling_rate = unique_sampling_rates[0]

                logger.info(f"       sampling rate: {sampling_rate:<11.1f} Hz")

                # total length of all samples (can accomodate a few samples offset)
                unique_nptss, unique_nptss_counts = np.unique(nptss, return_counts=True)

                if len(unique_nptss) > 1:
                    npts = unique_nptss[np.argmax(unique_nptss_counts)]
                else:
                    npts = unique_nptss[0]

                logger.info(f"   number of samples: {npts:<11.1f}")

                # slicing

                if length or padding:
                    logger.info("    Slicing data...")

                    if length:
                        logger.info(f"    length to slice: {trace_length:<11.1f} s")

                    if padding:
                        logger.info(f"            padding: {trace_length:<11.1f} s")

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
                        f"   number of samples: {num_samples:<11n} (sliced from {npts}, start: {start}, end: {end})"
                    )
                else:
                    num_samples = npts

                ### get data

                logger.info("    Preparing data...")

                data = []

                # extract data from trace
                for trace in traces_sliced:
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
                        # del event_indices[
                        #     index
                        # ]  # remove this event from the events that are to be correlated
                        # continue

                    data.append(trace_data)

                ### pad signal

                padded_npts = next_power_of_2(int(num_samples))

                data = np.array([to_length(trace, padded_npts) for trace in data])

                logger.info(
                    f"       signal length: {padded_npts:<11n} samples (zero-padded to nearest power of 2, {padded_npts / sampling_rate:.1f} s)"
                )

                ### do correlation and time it

                shift_len_samples = next_power_of_2(
                    int(ceil(shift_len * sampling_rate))
                )

                logger.info(
                    f"        shift length: {shift_len_samples:<11n} samples (ceiled to nearest power of 2, {shift_len:.1f} s)"
                )

                logger.info(f"         array shape: {data.shape}")

                ### transposing and converting data

                data = jl.pyconvert(
                    jl.Matrix, data.T
                )  # ensure column indexing (for julia)

                ### actual cross correlation

                logger.info("    Correlating on GPU...")

                pre_corr_time = time.time()

                # run correlation
                corrgrams_gpu_reduced = jl.Squiggles.norm_correlogram(
                    data, shift_len_samples, threads_per_block
                )

                # calculate lags
                xcorcoeff_GPU_reduced, lags_GPU_reduced = jl.Squiggles.simplelags(
                    corrgrams_gpu_reduced, shift_len_samples
                )

                # move to CPU from GPU
                xcorcoeff_reduced = jl.Squiggles.memcopy(xcorcoeff_GPU_reduced)
                lags_reduced = jl.Squiggles.memcopy(lags_GPU_reduced)
                
                # # move to CPU from GPU
                # xcorcoeff= jl.Squiggles.memcopy(xcorcoeff_GPU_reduced)
                # lags = jl.Squiggles.memcopy(lags_GPU_reduced)

                # reconstruct full matrix from reduced triangular representation
                xcorcoeff = jl.Squiggles.reconstruct_symmetric(xcorcoeff_reduced)
                lags = jl.Squiggles.reconstruct_antisymmetric(lags_reduced)

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
                    f"    Correlation complete!         (in {post_corr_time - pre_corr_time:.3g} s)"
                )

                # sparsify the matrix for saving in zarr (enables better chunking and later processing)
                num_events = len(events)

                sparse_corr_matrix = np.empty((num_events, num_events))
                sparse_lag_matrix = np.empty((num_events, num_events))

                sparse_corr_matrix[:] = np.nan
                sparse_lag_matrix[:] = np.nan

                for i, event_id_i in enumerate(event_indices):
                    for j, event_id_j in enumerate(event_indices):
                        sparse_corr_matrix[event_id_i, event_id_j] = corr_matrix[i, j]
                        sparse_lag_matrix[event_id_i, event_id_j] = lag_matrix[i, j]

                ### write

                self.correlation_file.write_correlation_matrix(
                    sparse_corr_matrix,
                    station=station,
                    channel=channel,
                )

                self.correlation_file.write_lag_matrix(
                    sparse_lag_matrix,
                    station=station,
                    channel=channel,
                )

                logger.info("    Correlations written to disk...")

                if plot:
                    Path("./correlation_plots").mkdir(exist_ok=True, parents=True)

                    fig = figure(figsize=(7, 7), layout="constrained")

                    ax = fig.add_subplot()

                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position("top")

                    crmat = ax.imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu")

                    cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
                    cbar.ax.set_ylabel("correlation value")

                    ax.set_title(
                        f"correlations, station: {station}, channel: {channel}"
                    )

                    ax.set_xlabel("event number")
                    ax.set_ylabel("event number")

                    fig.savefig(
                        out_dir
                        / (
                            out_filename.stem
                            + f"_{station}_{channel}"
                            + out_filename.suffix
                        )
                    )

                    del fig

                    fig = figure(figsize=(7, 7), layout="constrained")

                    ax = fig.add_subplot()

                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position("top")

                    crmat = ax.imshow(
                        (sparse_corr_matrix),
                        vmin=-1,
                        vmax=1,
                        cmap="RdBu",
                    )

                    cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
                    cbar.ax.set_ylabel("correlation value")

                    ax.set_title(
                        f"correlations, station: {station}, channel: {channel}"
                    )

                    ax.set_xlabel("event index")
                    ax.set_ylabel("event index")

                    fig.savefig(
                        out_dir
                        / (
                            out_filename.stem
                            + f"_{station}_{channel}_with_empty_vals"
                            + out_filename.suffix
                        )
                    )

                    close(fig)

                    logger.info("    plotted cross correlation matrix!")

        # if plot:
        #     fig = figure(figsize=(7, 7), layout="constrained")

        #     ax = fig.add_subplot()

        #     ax.xaxis.tick_top()
        #     ax.xaxis.set_label_position("top")

        #     crmat = ax.imshow(
        #         np.nanmean(vertical_component_corr_matrices, axis=2),
        #         vmin=0,
        #         vmax=1,
        #         cmap="viridis",
        #     )

        #     cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
        #     cbar.ax.set_ylabel("correlation value")

        #     ax.set_title("correlations, cross-channel mean, vertical channels")

        #     ax.set_xlabel("event index")
        #     ax.set_ylabel("event index")

        #     fig.savefig(
        #         Path(".")
        #         / (out_filename.stem + "_cross_channel_vertical" + out_filename.suffix)
        #     )

        #     close()

        #     print("--- plotted mean vertical cross-channel correlation matrix!")

        #     fig = figure(figsize=(7, 7), layout="constrained")

        #     ax = fig.add_subplot()

        #     ax.xaxis.tick_top()
        #     ax.xaxis.set_label_position("top")

        #     crmat = ax.imshow(
        #         np.nanmean(horizontal_component_corr_matrices, axis=2),
        #         vmin=0,
        #         vmax=1,
        #         cmap="viridis",
        #     )

        #     cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
        #     cbar.ax.set_ylabel("correlation value")

        #     ax.set_title("correlations, cross-channel mean, horizontal channels")

        #     ax.set_xlabel("event index")
        #     ax.set_ylabel("event index")

        #     fig.savefig(
        #         Path(".")
        #         / (
        #             out_filename.stem
        #             + "_cross_channel_horizontal"
        #             + out_filename.suffix
        #         )
        #     )

        #     close()

        #     print("--- plotted mean horizontal cross-channel correlation matrix!")
