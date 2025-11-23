import time
from pathlib import Path

from obspy import read

from cccpy.seismic import cross_correlate_seismic_events

stream_dict = dict()

for mseed_path in sorted(Path("./examples/test_data_bardarbunga_2017").glob("**/*_long.ms")):
    event_id = mseed_path.stem
    stream = read(mseed_path, format="MSEED")

    stream_dict[event_id] = stream

if stream_dict:
    start = time.time()

    cross_correlate_seismic_events(stream_dict)

    end = time.time()
    print("execution time", (end - start))
else:
    raise TypeError("Nothing to correlate!!")