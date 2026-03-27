import logging
import time
from math import ceil
from pathlib import Path
from typing import List

import numpy as np
from matplotlib.pyplot import figure
from obspy import Catalog
from scipy.interpolate import CubicSpline

from .interface import JuliaInterface
from .io.correlation_file import CorrelationFile
from ..util.helper_functions import next_power_of_2

logger = logging.getLogger(__name__)


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
        stream_dict,
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
        per_station_slices = self.correlation_file.slice_by_station_and_channel()

        # NOTE: ceil to power of 2 for shift_len

        for station, per_channel_slice in per_station_slices.items():
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

                ### calculate relevant values for processing based on input arguments

                # get required metadata
                sampling_rates = [trace.stats.sampling_rate for trace in traces_sliced]
                nptss = [trace.stats.npts for trace in traces_sliced]

                # sampling rate
                unique_sampling_rates = np.unique(sampling_rates)

                if len(unique_sampling_rates) > 1:
                    logger.warning("Inhomogeneous sampling rates, skipping...")
                    continue

                sampling_rate = unique_sampling_rates[0]

                logger.info(f"       sampling rate: {sampling_rate:<11.1f} Hz")

                # num samples and start and end of slicing

                # total length of all samples (can accomodate a few samples offset)
                max_npts = max(nptss)

                # if padding, start not at zero
                start = int(padding * sampling_rate)

                # if fixed template length is set, calculate length
                if length:
                    end = int(start + length * sampling_rate)
                else:
                    end = int(
                        max_npts - padding * sampling_rate
                    )  # go to end, unless padding

                num_samples = end - start

                logger.info(
                    f"       num samples:   {num_samples:<11n} (sliced from {max_npts}, start: {start}, end: {end})"
                    if max_npts != num_samples
                    else ""
                )

                ### get data

                data = []

                # extract data from trace
                for index, trace in enumerate(traces_sliced):
                    trace_sliced = trace.data[start:end]

                    if (
                        len(trace_sliced) != num_samples
                    ):  # sometimes a single trace is of by a few samples (or improper length is set)
                        logger.warning(
                            "One trace did not have the right number of samples (%s samples) for the requested length (%s samples), skipping",
                            len(trace_sliced),
                            num_samples,
                        )
                        del event_indices[
                            index
                        ]  # remove this event from the events that are to be correlated
                        continue

                    data.append(trace_sliced)

                data = np.array(data).T  # ensure column indexing (for julia)

                ### interpolate (if set)

                if interpolate:
                    logger.info(f"    Interpolating to {interpolate} subdivisions...")
                    pre_interp_time = time.time()

                    interpolator = CubicSpline(list(range(data.shape[0])), data)

                    interpolated_data = interpolator(
                        np.linspace(0, data.shape[0], data.shape[0] * 2**interpolate)
                    )

                    post_interp_time = time.time()

                    logger.info(
                        f"    Interpolation complete!         (in {post_interp_time - pre_interp_time:.3g} s)"
                    )

                    # apply new values
                    sampling_rate *= 2**interpolate
                    data = interpolated_data

                    # print new effective rates
                    logger.info(
                        f"       sampling rate: {sampling_rate:<11.1f} Hz (effective)"
                    )

                    logger.info(
                        f"       num samples:   {interpolated_data.shape[0]:<11n}"
                    )

                ### do correlation and time it

                shift_len_samples = next_power_of_2(
                    int(ceil(shift_len * sampling_rate))
                )

                logger.info(
                    f"       shift length:  {shift_len:<11.1f} s (ceiled to {shift_len_samples} samples)"
                )

                logger.info("    Correlating on GPU...")

                pre_corr_time = time.time()

                mat_cpu, lag_cpu = jl.CCC.ncc_cuda(
                    data, data, shift_len_samples, threads_per_block
                )

                corr_matrix = np.array(
                    mat_cpu
                ).T  # convert julia array and revert to row indexing
                lag_matrix = np.array(
                    lag_cpu
                ).T  # convert julia array and revert to row indexing

                # convert lags from samples to seconds
                lag_matrix *= sampling_rate

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

                self.correlation_file.write_correlation_matrix(
                    sparse_corr_matrix,
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

                    crmat = ax.imshow(corr_matrix, vmin=0, vmax=1, cmap="viridis")

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
                        vmin=0,
                        vmax=1,
                        cmap="viridis",
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

                    del fig

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
