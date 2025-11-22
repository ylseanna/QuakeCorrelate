from pathlib import Path
from obspy import read
import time
import cccpy


stream_dict = dict()

for mseed_path in sorted(Path("./examples/test_data").glob("*/**/*_st_long.ms")):
    event_id = mseed_path.parent.stem
    stream = read(mseed_path, format="MSEED")

    stream_dict[event_id] = stream

start= time.time()

cccpy.cross_correlate_seismic_events(stream_dict)

end= time.time()
print("execution time", (end-start))
