from pathlib import Path
from typing import List

import numpy as np
from matplotlib.pyplot import close, figure
from obspy import Catalog

from ..core.interface import JuliaInterface
from ..core.io.correlation_file_zarr import SeismicCorrelationFile


class SeismicCorrelator:
    def __init__(
        self,
        corr_file: str | Path = Path("correlations.zarr"),
        overwrite_corr_file: bool = False,
        catalog: Catalog | None = None,
        stations: List[str] | None = None,
        channels: List[str] | None = None,
    ):
        corr_file = Path(corr_file)

        self.correlation_file = SeismicCorrelationFile(
            filename=corr_file, overwrite_file=overwrite_corr_file
        )

        if catalog is not None:
            self.correlation_file.write_catalog(catalog)

        if stations is not None:
            self.correlation_file.write_stations(stations)

        if channels is not None:
            self.correlation_file.write_channels(channels)

    def cross_correlate(
        self,
        stream_dict,
        shift_len: float = 1,
        plot: bool = False,
        out_dir: str | Path = "correlation_plots",
        out_filename: str | Path = "correlation.png",
    ):
        out_dir = Path(out_dir)
        out_filename = Path(out_filename)

        # get julia
        jl = JuliaInterface()

        # write streams
        self.correlation_file.write_events(list(stream_dict.keys()))

        # get event data
        events = self.correlation_file.get_events(as_df=False)

        # get slices
        per_station_slices = self.correlation_file.slice_by_station_and_channel()

        # NOTE: ceil to power of 2 for shift_len

        # if plot:
        #     events = self.correlation_file.get_events()

        #     vertical_component_corr_matrices = np.empty(
        #         (len(events), len(events), len(per_station_slices.keys()))
        #     )
        #     vertical_component_corr_matrices[:] = np.nan

        #     horizontal_component_corr_matrices = np.empty(
        #         (len(events), len(events), len(per_station_slices.keys()))
        #     )
        #     horizontal_component_corr_matrices[:] = np.nan

        for station, per_channel_slice in per_station_slices.items():
            for channel, event_indices in per_channel_slice.items():
                print(f"--- Correlating for station {station}, channel {channel} ---")

                data = np.array(
                    [
                        [
                            trace.data[:249]
                            for trace in stream
                            if trace.stats.channel == channel
                            and trace.stats.station == station
                        ][0]
                        for stream in [
                            stream
                            for stream in [
                                stream_dict[event_id]
                                for event_index, event_id in events.items()
                                if int(event_index) in event_indices
                            ]
                        ]
                    ]
                ).T

                mat_cpu, lag_cpu = jl.CCC.ncc_cuda(data, data, 128, 256)

                corr_matrix = np.array(mat_cpu).T
                lag_matrix = np.array(lag_cpu).T

                sparse_corr_matrix = np.empty(
                    (max(event_indices) + 1, max(event_indices) + 1)
                )
                sparse_lag_matrix = np.empty(
                    (max(event_indices) + 1, max(event_indices) + 1)
                )

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

                print("    correlation complete!")

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

                    close()

                    fig = figure(figsize=(7, 7), layout="constrained")

                    ax = fig.add_subplot()

                    ax.xaxis.tick_top()
                    ax.xaxis.set_label_position("top")

                    crmat = ax.imshow(
                        (
                            sparse_corr_matrix
                        ),
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

                    close()

                    print("    plotted cross correlation matrix!")

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
