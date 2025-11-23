from pathlib import Path
from typing import List

import numpy as np

from ..core.io import CorrelationFile

from ..core.interface import Julia

from matplotlib.pyplot import figure, close


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
        jl = Julia()

        # write streams
        self.correlation_file.write_streams(stream_dict)

        slices = self.correlation_file.slice_by_station_and_channel()

        # NOTE: ceil to power of 2 for shift_len

        for slice in slices:
            print(
                f"Correlating for station {slice['station'][1]} channel {slice['channel'][1]}"
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
                slice["trace_ids"],
                slice["trace_ids"],
            )

            if plot:

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

                ax.set_xlabel("event index")
                ax.set_ylabel("event index")

                fig.savefig(
                    out_dir
                    / (
                        out_filename.stem
                        + f"_{slice['station'][1]}_{slice['channel'][1]}"
                        + out_filename.suffix
                    )
                )

                close()
