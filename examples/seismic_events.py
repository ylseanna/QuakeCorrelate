import numpy as np
from matplotlib.pyplot import figure
from pathlib import Path

import cccpy

from obspy import read

stream_dict = dict()

for mseed_path in sorted(
    Path(
        "/home/yadevries/PhD/areas/Bardarbunga/rematch_pipeline_instances/full_catalog_4/detections/2017/163"
    ).glob("*/**/*_st_cc.ms")
):
    event_id = mseed_path.parent.stem
    stream = read(mseed_path, format="MSEED")

    stream_dict[event_id] = stream


cccpy.cross_correlate_seismic_events(stream_dict)
