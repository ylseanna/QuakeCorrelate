import time
from pathlib import Path

from obspy import read, read_events, Catalog

from cccpy.seismic import cross_correlate_seismic_events

stream_dict = dict()

template_catalog = Catalog()

for mseed_path in sorted(
    Path("./examples/test_data_bardarbunga_2017").glob("**/*_long.ms")
):
    event_id = mseed_path.stem
    stream = read(mseed_path, format="MSEED")

    template_event_path = sorted(mseed_path.parent.glob("*.xml"))[0]

    template_event = read_events(template_event_path)[0]

    stream_dict[template_event.resource_id.id] = stream

    template_catalog.append(template_event)

if stream_dict:
    start = time.time()

    cross_correlate_seismic_events(stream_dict, template_catalog, plot_corr_matrix=True)

    end = time.time()
    print("execution time", (end - start))
else:
    raise TypeError("Nothing to correlate!!")
