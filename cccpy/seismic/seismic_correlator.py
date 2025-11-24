from pathlib import Path
from typing import List

import numpy as np
from matplotlib.pyplot import close, figure

from ..core.interface import Julia
from ..core.io import CorrelationFile


class SeismicCorrelator:
    def __init__(
        self,
        corr_file: str | Path = Path("correlation.sqlite"),
        overwrite_corr_file: bool = False,
        stations: List[str] | None = None,
        channels: List[str] | None = None,
    ):
        corr_file = Path(corr_file)

        self.correlation_file = CorrelationFile(
            filename=corr_file, seismic_data=True, overwrite_file=overwrite_corr_file
        )

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
        jl = Julia()

        # write streams
        self.correlation_file.write_streams(stream_dict)

        slices = self.correlation_file.slice_by_station_and_channel()

        # NOTE: ceil to power of 2 for shift_len

        if plot:
            events = self.correlation_file.get_events()

            vertical_component_corr_matrices = np.empty(
                (len(events), len(events), len(slices))
            )
            vertical_component_corr_matrices[:] = np.nan

            horizontal_component_corr_matrices = np.empty(
                (len(events), len(events), len(slices))
            )
            horizontal_component_corr_matrices[:] = np.nan

        for slice_i, slice in enumerate(slices):
            print(
                f"--- Correlating for station {slice['station'][1]} channel {slice['channel'][1]} ---"
            )

            data = np.array(
                [
                    [
                        trace.data[:249]
                        for trace in stream
                        if trace.stats.channel == slice["channel"][1]
                        and trace.stats.station == slice["station"][1]
                    ][0]
                    for stream in [
                        stream
                        for stream in [
                            stream_dict[event[1]] for event in slice["events"]
                        ]
                    ]
                ]
            ).T

            mat_cpu, lag_cpu = jl.CCC.ncc_cuda(data, data, 128, 256)

            corr_matrix = np.array(mat_cpu).T
            lag_matrix = np.array(lag_cpu).T

            self.correlation_file.write_correlation_matrix(
                corr_matrix,
                lag_matrix,
                [slice[0] for slice in slice["traces"]],
                [slice[0] for slice in slice["traces"]],
            )

            print("    correlation complete!")

            if plot:
                if slice["channel"][1].endswith(("Z", "3")):
                    for i, event_id_i in enumerate(
                        [event[0] for event in slice["events"]]
                    ):
                        for j, event_id_j in enumerate(
                            [event[0] for event in slice["events"]]
                        ):
                            vertical_component_corr_matrices[
                                event_id_i - 1, event_id_j - 1, slice_i
                            ] = corr_matrix[i, j]
                if slice["channel"][1].endswith(("E", "N", "1", "2")):
                    for i, event_id_i in enumerate(
                        [event[0] for event in slice["events"]]
                    ):
                        for j, event_id_j in enumerate(
                            [event[0] for event in slice["events"]]
                        ):
                            horizontal_component_corr_matrices[
                                event_id_i - 1, event_id_j - 1, slice_i
                            ] = corr_matrix[i, j]

                fig = figure(figsize=(7, 7), layout="constrained")

                ax = fig.add_subplot()

                ax.xaxis.tick_top()
                ax.xaxis.set_label_position("top")

                crmat = ax.imshow(corr_matrix, vmin=0, vmax=1, cmap="viridis")

                cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
                cbar.ax.set_ylabel("correlation value")

                ax.set_title(
                    f"correlations, station: {slice['station'][1]}, channel: {slice['channel'][1]}"
                )

                ax.set_xlabel("event number")
                ax.set_ylabel("event number")

                fig.savefig(
                    out_dir
                    / (
                        out_filename.stem
                        + f"_{slice['station'][1]}_{slice['channel'][1]}"
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
                        vertical_component_corr_matrices[:, :, slice_i]
                        if slice["channel"][1].endswith(("Z", "3"))
                        else horizontal_component_corr_matrices[:, :, slice_i]
                    ),
                    vmin=0,
                    vmax=1,
                    cmap="viridis",
                )

                cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
                cbar.ax.set_ylabel("correlation value")

                ax.set_title(
                    f"correlations, station: {slice['station'][1]}, channel: {slice['channel'][1]}"
                )

                ax.set_xlabel("event index")
                ax.set_ylabel("event index")

                fig.savefig(
                    out_dir
                    / (
                        out_filename.stem
                        + f"_{slice['station'][1]}_{slice['channel'][1]}_with_empty_vals"
                        + out_filename.suffix
                    )
                )

                close()

                print("    plotted cross correlation matrix!")

        if plot:
            fig = figure(figsize=(7, 7), layout="constrained")

            ax = fig.add_subplot()

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

            crmat = ax.imshow(
                np.nanmean(vertical_component_corr_matrices, axis=2),
                vmin=0,
                vmax=1,
                cmap="viridis",
            )

            cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
            cbar.ax.set_ylabel("correlation value")

            ax.set_title(
                "correlations, cross-channel mean, vertical channels"
            )

            ax.set_xlabel("event index")
            ax.set_ylabel("event index")

            fig.savefig(
                Path(".")
                / (out_filename.stem + "_cross_channel_vertical" + out_filename.suffix)
            )

            close()

            print("--- plotted mean vertical cross-channel correlation matrix!")

            fig = figure(figsize=(7, 7), layout="constrained")

            ax = fig.add_subplot()

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

            crmat = ax.imshow(
                np.nanmean(horizontal_component_corr_matrices, axis=2),
                vmin=0,
                vmax=1,
                cmap="viridis",
            )

            cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
            cbar.ax.set_ylabel("correlation value")

            ax.set_title(
                "correlations, cross-channel mean, horizontal channels"
            )

            ax.set_xlabel("event index")
            ax.set_ylabel("event index")

            fig.savefig(
                Path(".")
                / (out_filename.stem + "_cross_channel_horizontal" + out_filename.suffix)
            )

            close()

            print("--- plotted mean horizontal cross-channel correlation matrix!")
