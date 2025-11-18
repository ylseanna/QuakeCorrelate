from ..core.io import CorrelationFile
from pathlib import Path
import numpy as np


import matplotlib.pyplot as plt


class SeismicCorrelator:
    def __init__(
        self,
        corr_file: str | Path = Path("correlation.sqlite"),
        overwrite_corr_file: bool = False,
    ):
        corr_file = Path(corr_file)

        self.correlation_file = CorrelationFile(
            filename=corr_file, seismic_data=True, overwrite_file=overwrite_corr_file
        )

    def cross_correlate(self, stream_dict, shift_len: float = 1):
        slices = self.correlation_file.slice_by_station_and_channel()

        # NOTE: ceil to power of 2 for shift_len

        for slice in slices:
            print(
                f"Correlating for station {slice['station'][1]} channel {slice['channel'][1]}"
            )

            data = np.array(
                [
                    [
                        trace.data
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
            )

            print(data)
