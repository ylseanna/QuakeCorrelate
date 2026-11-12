import logging
from typing import Dict, List, Literal, Tuple

from numpy import array, ndarray, unique, zeros
from obspy import Catalog, Stream, Trace, UTCDateTime

from ..util.helper_functions import next_power_of_2, to_length

logger = logging.getLogger(__name__)


def extract_traces(
    stream_dict: Dict[str, Stream],
    event_ids: List[str],
    catalog: Catalog,
    station: str,
    channel: str,
    phase_hint: Literal["P", "S"],
    prepick: float,
    postpick: float,
    padding: float,
) -> Tuple[ndarray, List[Trace]]:
    """Function to scan through a stream dictionary and find and cut the traces for a given number of events,
    using the associated station ID, channel ID, phase hint (P or S, i.e. horizontal or vertical trace) and the given
    cutting parameters around the pick: prepick, postpick and padding.

    Args:
        stream_dict: Dictionary keyed by event ID with the Obspy Stream object that contain the data for each station and phase to be correlated.
        event_ids: List of all event IDs to be extracted.
        catalog: CorrelationFile.metadata.catalog object that contains the relevant event for the traces to be extracted, namely the picks.
        station: Station ID to be extracted for.
        channel: Channel ID to be extracted for.
        phase_hint: Phase hint (P or S) to be extracted for.
        prepick: Number of seconds prior to the pick time that should be extracted from the original Trace data.
        postpick: Number of seconds after the pick time that should be extracted from the original Trace data.
        padding: Amount of padding in seconds to be added on either side of the extraction window, as used during the correlation step.

    Returns:
        A square Numpy array where the first axis is the trace number (by ordered Event ID) and the second axis the data itself (in samples), also returns an dictionary of the original trace object indexed by Event ID.
    """
    logger.info(f"        Accessing traces and picks ({phase_hint})...")

    # loop throught stream dict for event and find correct traces for channel and station
    trace_dict = {}
    pick_dict = {}
    for event_id, stream in [
        (event_id, stream_dict[event_id]) for event_id in event_ids
    ]:
        trace_dict[event_id] = (
            [
                trace
                for trace in stream
                if trace.stats.channel == channel and trace.stats.station == station
            ][0]
            if [
                trace
                for trace in stream
                if trace.stats.channel == channel and trace.stats.station == station
            ]
            else []
        )

    # use phase hint to find corresponding pick (NOTE: determined from channel postfix in earlier code, e.g. HH<E> vs HH<Z>)
    for event_id in event_ids:
        event = catalog[event_id]

        find_station_channel_phase_pick = (
            lambda item: item["channel_name"] == channel
            and item["station_name"] == station
            and item["phase_hint"] == phase_hint
        )

        pick_dict[event_id] = next(
            (pick for pick in event["picks"] if find_station_channel_phase_pick(pick)),
            None,
        )

    # look if sampling rates match, otherwise resample (just easy to implement)
    logger.info("        Checking trace consistency...")

    # check sampling rates, if inhomogeneous resample to most common
    sampling_rates = [trace.stats.sampling_rate for trace in trace_dict.values()]

    # all unique (most common first)
    unique_sampling_rates = unique(sampling_rates)

    # get most common and assign as default for run
    sampling_rate = unique_sampling_rates[0]

    # check if inhomogeneous
    if len(unique_sampling_rates) > 1:
        logger.warning("Inhomogeneous sampling rates, resampling to most common...")

        for tr in trace_dict.values():
            if tr.stats.sampling_rate != sampling_rate:
                tr.resample(sampling_rate)

    # list sampling rate
    logger.info(f"           sampling rate: {sampling_rate:<11.1f} Hz")

    # slice traces
    logger.info("        Slicing traces to picks...")

    logger.info(f"                 prepick: {prepick:<11.1f} s")
    logger.info(f"                postpick: {postpick:<11.1f} s")
    logger.info(f"                 padding: {padding:<11.1f} s")
    logger.info(
        f"            total length: {prepick + postpick + 2 * padding:<11.1f} s (expected, may vary by a few samples)"
    )

    for event_id, tr in trace_dict.items():
        pick_time = UTCDateTime(pick_dict[event_id]["time"])

        trace_dict[event_id] = tr.slice(
            pick_time - (prepick + padding), pick_time + (postpick + padding)
        )

    logger.info("        Checking sliced trace consistency...")

    # check number of sample
    nptss = [trace.stats.npts for trace in trace_dict.values()]

    # all unique (most common first)
    unique_npts = unique(nptss)

    npts = max(unique_npts)

    # check if inhomogeneous
    if len(unique_npts) > 1:
        logger.warning(
            f"Inhomogeneous number of samples, will pad to highest number... (most common num: {unique_npts[0]})"
        )

    logger.info(f"             num samples: {npts:<12n} samples")
    logger.info(f"            trace length: {npts * 1/sampling_rate:<11.1f} s")

    logger.info("        Preparing data matrix...")
    # assign data to matrix
    trace_data = zeros((len(event_ids), npts))

    for index, event_id in enumerate(event_ids):
        tr = trace_dict[event_id]

        trace_data[index, : tr.stats.npts + 1] = tr.data

    ### pad signal
    logger.info("        padding signal to next power of two...")
    padded_npts = next_power_of_2(int(npts))

    trace_data = array([to_length(trace, padded_npts) for trace in trace_data])

    logger.info(
        f"           signal length: {padded_npts:<11n} samples (zero-padded to nearest power of 2, {padded_npts / sampling_rate:.1f} s)"
    )

    return trace_data, trace_dict
