import numpy as np
from pathlib import Path
from typing import List
from juliacall import Main as jl

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

    def cross_correlate(self, stream_dict, shift_len: float = 1):
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
                        trace.data[:704]
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
