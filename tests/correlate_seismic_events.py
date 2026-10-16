import logging
import sys
import time
from pathlib import Path

from obspy import Catalog, read, read_events

from quakecorrelate import cross_correlate_seismic_events

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

logger = logging.getLogger("root")

stream_dict = dict()

template_catalog = Catalog()

logger.info("Reading templates...")

for mseed_path in sorted(
    Path("<miniseed_file_location>").glob("**/*.ms")
):
    
    event_id = mseed_path.stem
    
    logger.info(f"Reading template {event_id}")
    
    stream = read(mseed_path, format="MSEED")

    template_event_path = sorted(mseed_path.parent.glob("*.xml"))[0]

    template_event = read_events(template_event_path)[0]
    

    stream_dict[template_event.resource_id.id] = stream

    template_catalog.append(template_event)

if stream_dict:
    start = time.time()

    cross_correlate_seismic_events(
        stream_dict, template_catalog, plot_corr_matrix=True
    )

    end = time.time()
    logging.info(f"execution time {(end - start)}")
else:
    raise TypeError("Nothing to correlate!!")
