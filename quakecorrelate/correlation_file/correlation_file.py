import logging
import time
from pathlib import Path
from shutil import copyfileobj
from typing import Dict, List, Literal, Tuple, TypedDict
from multiprocessing import Pool, set_start_method

import numpy as np
import pandas as pd
import zarr
from matplotlib.pyplot import close, figure
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event

from ..util import ensure_unique_values, get_chunks_to_process
from .dtcc import _write_dtcc_parallel, _write_single_event_pair_to_dtcc

logger = logging.getLogger(__name__)


class CorrMatrixDict(TypedDict):
    station: str
    channel: str
    corr_matrix: np.ndarray


class CorrelationFile:
    """Main interface for the correlation file written to during cross correlations.

    Args:
        filename: The filename of the .zarr file used to write correlations to, not used if *in_memory* is enabled
        in_memory: Description.
        overwrite_file: Whether or not to create a fresh correlation file, overwriting the existing one if needed.
        chunksize: The (squared) size of the chunks used for reading data arrays from the correlation file, determines the size of chunks when processing the correlations into a output format (e.g. dt.cc).
        shardsize: The (squared) size of the shards used for writing data arrays from the correlation file, determines the size of chunks processed during cross correlations.


    """

    def __init__(
        self,
        filename: str | Path = "./correlations.zarr",
        in_memory: bool = False,
        overwrite_file: bool = False,
        chunksize: int = 100,
        shardsize: int = 1000,
    ):
        # args
        self.filename: str = str(filename)
        """The filename of the .zarr file used to write correlations to"""
        self.chunksize: int = chunksize
        """The (squared) size of the chunks used for reading data arrays from the correlation file, determines the size of chunks when processing the correlations into a output format (e.g. dt.cc)"""
        self.shardsize: int = shardsize
        """The (squared) size of the shards used for writing data arrays from the correlation file, determines the size of chunks processed during cross correlations."""

        filename = Path(filename)

        if overwrite_file:
            root_args = dict(mode="w")
        else:
            root_args = dict()

        # open (or create) zarr file
        self.root = zarr.open_group(
            store=(
                filename if not in_memory else zarr.storage.MemoryStore()
            ),  # keep story in memory, suitable for small correlation batches used for internal processes
            **root_args,  # overwrite or else default
        )

        # get attributes
        if "catalog" in self.root.attrs:
            self.catalog: Dict | None = self.root.attrs["catalog"]
            """Dictionary of event catalog entries, including picks, as stored in the metadata and used for processing and exporting, keyed by event ID"""
        else:
            self.catalog: Dict | None = None

        if "stations" in self.root.attrs:
            self.stations: Dict | None = self.root.attrs["stations"]
            """Dictionary of stations as stored in the metadata and used for processing and exporting, keyed by station index."""
        else:
            self.stations: Dict | None = None

        if "channels" in self.root.attrs:
            self.channels: Dict | None = self.root.attrs["channels"]
            """Dictionary of stations as stored in the metadata and used for processing and exporting, keyed by station index."""
        else:
            self.channels: Dict | None = None

        if "events" in self.root.attrs:
            self.events: Dict | None = self.root.attrs["events"]
        else:
            self.events: Dict | None = None
            
        if "id_mapping" in self.root.attrs:
            self.id_mapping: Dict | None = self.root.attrs["id_mapping"]
        else:
            self.id_mapping: Dict | None = None

    ### add indexing to CorrelationFile object - just refer straight to root

    def __getitem__(self, index):
        return self.root[index]

    ### WRITE

    def write_catalog(self, catalog: Catalog):
        """Writes an Obspy catalogue to the metadata of the correlation file. Must include at least one Origin and the 
        relevant Picks for the streams that are correlated
        
        Arguments:
            catalog: Obspy Catalog object that is to be written to the metadata. Must include at least one Origin and the picks for the streams to be correlated."""

        keyed_catalog_dict = {}

        for event in catalog:
            event_id = event.resource_id.id

            event_dict = {}

            # add origin information
            origin = event.preferred_origin() or event.origins[0] or None

            if origin:
                event_dict["origin"] = {
                    "time": origin.time.isoformat(),
                    "lon": origin.longitude,
                    "lat": origin.latitude,
                    "depth": origin.depth,
                }

                magnitude = (
                    event.preferred_magnitude().mag or event.magnitudes[0].mag or None
                )

                if magnitude:
                    event_dict["origin"]["mag"] = magnitude

            picks = []

            for pick in event.picks:
                picks.append(
                    {
                        "station_name": pick.waveform_id.station_code,
                        "channel_name": pick.waveform_id.channel_code,
                        "phase_hint": pick.phase_hint,
                        "time": pick.time.isoformat(),
                    }
                )

            event_dict["picks"] = picks

            keyed_catalog_dict[event_id] = event_dict

        self.root.attrs["catalog"] = keyed_catalog_dict

        # make attribute
        self.catalog = self.root.attrs["catalog"]

    def write_stations(self, stations: List[str]):
        """Writes a list of station IDs to the metadata of the correlation file. Must include at least all the station IDs that are to be correlated for.
        
        Arguments:
            stations: List of station IDs (e.g. STA, STAT) that are to be written to the metadata. Must include all the station IDs that are to be correlated for."""
        
        # ensure sorting
        stations.sort()

        # insert stations into attributes
        if "stations" not in self.root.attrs:  # if not old values exist just insert
            keyed_station_dict = ensure_unique_values(
                {str(index): str(station) for index, station in enumerate(stations)}
            )
        else:  # combine both dictionaries, with new keys
            keyed_station_dict_old = self.root.attrs["stations"]

            max_index = max([int(key) for key in keyed_station_dict_old.keys()])

            keyed_station_dict_new = ensure_unique_values(
                {
                    str(index + 1 + max_index): str(station)
                    for index, station in enumerate(stations)
                }
            )

            keyed_station_dict = ensure_unique_values(
                keyed_station_dict_old | keyed_station_dict_new
            )  # by appending to end we ensure proper ordering

        # insert into attributes
        self.root.attrs["stations"] = keyed_station_dict

        # make attribute
        self.stations = self.root.attrs["stations"]

    def write_channels(self, channels: List[str]):
        """Writes a list of channel IDs to the metadata of the correlation file. Must include at least all the channel IDs that are to be correlated for.
        
        Arguments:
            stations: List of channel IDs (e.g. HHZ, EHN) that are to be written to the metadata. Must include all the channel IDs that are to be correlated for."""
        # ensure sorting
        channels.sort()

        # insert channels into attributes
        if "channels" not in self.root.attrs:  # if not old values exist just insert
            keyed_channel_dict = ensure_unique_values(
                {str(index): str(channel) for index, channel in enumerate(channels)}
            )
        else:  # combine both dictionaries, with new keys
            keyed_channel_dict_old = self.root.attrs["channels"]

            max_index = max([int(key) for key in keyed_channel_dict_old.keys()])

            keyed_channel_dict_new = ensure_unique_values(
                {
                    str(index + 1 + max_index): str(channel)
                    for index, channel in enumerate(channels)
                }
            )

            keyed_channel_dict = ensure_unique_values(
                keyed_channel_dict_old | keyed_channel_dict_new
            )  # by appending to end we ensure proper ordering

        # insert into attributes
        self.root.attrs["channels"] = keyed_channel_dict

        # link to attribute
        self.channels = self.root.attrs["channels"]

    def write_events(self, events: List[str]):
        """Writes a list of event IDs Must include at least all the event IDs that are to be correlated for, amd must correspond to the Event.resource_id.id() for the Catalog object embedded in the metadata.
        Will assigns internal integer indices to the events that correspond to the index in the, e.g., correlation and lag matrices.
        
        Arguments:
            events: List of events IDs (e.g. 20170415_test_event) that are to be written to the metadata. Must include all the event IDs that are to be correlated for."""
        # ensure sorting
        events.sort()

        # insert events into attributes
        if "events" not in self.root.attrs:  # if not old values exist just insert
            keyed_event_dict = ensure_unique_values(
                {str(index): str(event) for index, event in enumerate(events)}
            )
        else:  # combine both dictionaries, with new keys
            keyed_event_dict_old = self.root.attrs["events"]

            max_index = max([int(key) for key in keyed_event_dict_old.keys()])

            keyed_event_dict_new = ensure_unique_values(
                {
                    str(index + 1 + max_index): str(event)
                    for index, event in enumerate(events)
                }
            )

            keyed_event_dict = ensure_unique_values(
                keyed_event_dict_old | keyed_event_dict_new
            )  # by appending to end we ensure proper ordering

        # insert into attributes
        self.root.attrs["events"] = keyed_event_dict

        # link to attribute
        self.events = self.root.attrs["events"]

    def write_correlation_matrix(
        self,
        corr_matrix: np.ndarray,
        station: str,
        channel: str,
        primary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        secondary_event_slice: Literal["all"] | Tuple[int, int] = "all",
    ):
        # retrieve metadata from file
        events = self.root.attrs["events"]

        event_indices = [int(index) for index in events.keys()]

        max_event_index = event_indices[-1]

        # check if array exists (can probably be made more compact, but this helps readability)
        if station in self.root:
            if channel in self.root[station]:
                if "corr_values" in self.root[station][channel]:
                    array_exists = True
                else:
                    array_exists = False
            else:
                array_exists = False
        else:
            array_exists = False

        # create or access array
        if not array_exists:
            corr_array = self.root.create_array(
                f"{station}/{channel}/corr_values",
                # data=corr_matrix.astype(np.dtype("float32")),
                shape=(max_event_index + 1, max_event_index + 1),
                dtype="float32",
                chunks=(self.chunksize, self.chunksize),
                shards=(self.shardsize, self.shardsize),
                fill_value=np.nan,
            )
        else:
            corr_array = self.root[station][channel]["corr_values"]

        # write data
        if primary_event_slice == "all" and secondary_event_slice == "all":
            corr_array[0:max_event_index, 0:max_event_index] = corr_matrix.astype(
                np.dtype("float32")
            )
        else:
            corr_array[
                primary_event_slice[0] : primary_event_slice[1] + 1,
                secondary_event_slice[0] : secondary_event_slice[1] + 1,
            ] = corr_matrix.astype(np.dtype("float32"))

    def write_lag_matrix(
        self,
        lag_matrix: np.ndarray,
        station: str,
        channel: str,
        primary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        secondary_event_slice: Literal["all"] | Tuple[int, int] = "all",
    ):
        # retrieve metadata from file
        events = self.root.attrs["events"]

        event_indices = [int(index) for index in events.keys()]

        max_event_index = event_indices[-1]

        # check if array exists (can probably be made more compact, but this helps readability)
        if station in self.root:
            if channel in self.root[station]:
                if "lag_values" in self.root[station][channel]:
                    array_exists = True
                else:
                    array_exists = False
            else:
                array_exists = False
        else:
            array_exists = False

        # create or access array
        if not array_exists:
            lag_array = self.root.create_array(
                f"{station}/{channel}/lag_values",
                # data=lag_matrix.astype(np.dtype("float32")),
                shape=(max_event_index + 1, max_event_index + 1),
                dtype="float32",
                chunks=(self.chunksize, self.chunksize),
                shards=(self.shardsize, self.shardsize),
                fill_value=np.nan,
            )
        else:
            lag_array = self.root[station][channel]["lag_values"]

        # write data
        if primary_event_slice == "all" and secondary_event_slice == "all":
            lag_array[0:max_event_index, 0:max_event_index] = lag_matrix.astype(
                np.dtype("float32")
            )
        else:
            lag_array[
                primary_event_slice[0] : primary_event_slice[1] + 1,
                secondary_event_slice[0] : secondary_event_slice[1] + 1,
            ] = lag_matrix.astype(np.dtype("float32"))

    ### SLICE

    def _slice_traces_by_station_and_channel(
        self,
        primary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        secondary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        with_corr_file_index: bool = True,
    ) -> (
        Dict[str, Dict[str, List[int]]]
        | Dict[str, Dict[str, Tuple[List[int], List[int]]]]
    ):
        event_indices = [int(index) for index in self.events.keys()]

        stations = list(self.stations.values())

        channels = list(self.channels.values())

        slice_dict = {}

        if primary_event_slice == secondary_event_slice:

            event_indices_slice = (
                event_indices
                if primary_event_slice == "all"
                else event_indices[primary_event_slice[0] : primary_event_slice[1] + 1]
            )

            for event_index in event_indices_slice:
                event_id = self.events[str(event_index)]

                event_info = self.catalog[event_id]

                for pick in event_info["picks"]:
                    if (
                        pick["station_name"] in stations
                        and pick["channel_name"] in channels
                    ):
                        if pick["station_name"] not in slice_dict:
                            slice_dict[pick["station_name"]] = dict()

                        if pick["channel_name"] not in slice_dict[pick["station_name"]]:
                            slice_dict[pick["station_name"]][
                                pick["channel_name"]
                            ] = list()

                        slice_dict[pick["station_name"]][pick["channel_name"]].append(
                            int(event_index)
                        )
        else:
            # assign which events to process, slice if necessary
            primary_event_indices = (
                event_indices
                if primary_event_slice == "all"
                else event_indices[primary_event_slice[0] : primary_event_slice[1] + 1]
            )

            secondary_event_indices = (
                event_indices
                if primary_event_slice == "all"
                else event_indices[
                    secondary_event_slice[0] : secondary_event_slice[1] + 1
                ]
            )

            for event_index in primary_event_indices:
                event_id = self.events[str(event_index)]

                event_info = self.catalog[event_id]

                for pick in event_info["picks"]:
                    if (
                        pick["station_name"] in stations
                        and pick["channel_name"] in channels
                    ):
                        if pick["station_name"] not in slice_dict:
                            slice_dict[pick["station_name"]] = dict()

                        if pick["channel_name"] not in slice_dict[pick["station_name"]]:
                            slice_dict[pick["station_name"]][pick["channel_name"]] = (
                                list(),
                                list(),
                            )

                        slice_dict[pick["station_name"]][pick["channel_name"]][
                            0
                        ].append(int(event_index))

            for event_index in secondary_event_indices:
                event_id = self.events[str(event_index)]

                event_info = self.catalog[event_id]

                for pick in event_info["picks"]:
                    if (
                        pick["station_name"] in stations
                        and pick["channel_name"] in channels
                    ):
                        if pick["station_name"] not in slice_dict:
                            slice_dict[pick["station_name"]] = dict()

                        if pick["channel_name"] not in slice_dict[pick["station_name"]]:
                            slice_dict[pick["station_name"]][pick["channel_name"]] = (
                                list(),
                                list(),
                            )

                        slice_dict[pick["station_name"]][pick["channel_name"]][
                            1
                        ].append(int(event_index))

        return slice_dict

    def _get_chunked_event_pair_corr_and_lag_values(
        self,
        primary_event_slice: Tuple[int, int],
        secondary_event_slice: Tuple[int, int],
        stations: Literal["all"] | List[str] = "all",
        channels: Literal["all", "all_horiz", "all_vert"] | List[str] = "all",
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
    ):
        # make empty initial array with indexes
        station_channel_index = []

        # dynamically get all available station names, or not
        if stations == "all":
            station_names = list(self.root.group_keys())
        else:
            station_names = stations

        for station_name in station_names:
            if station_name in self.root:
                # dynamically get all available channel names, or not
                if (
                    channels == "all"
                    or channels == "all_horiz"
                    or channels == "all_vert"
                ):
                    channel_names = []
                    for channel_name, group in self.root[station_name].groups():
                        if channels == "all_horiz" and not channel_name.endswith(
                            tuple(horizontal_channels_postfix)
                        ):
                            continue  # skip iteration if not horizontal channel code
                        elif channels == "all_vert" and not channel_name.endswith(
                            tuple(vertical_channels_postfix)
                        ):
                            continue  # skip iteration if not vertical channel code

                        channel_names.append(channel_name)
                else:
                    channel_names = channels

                for channel_name in channel_names:
                    if channel_name in self.root[station_name]:
                        station_channel_index.append((station_name, channel_name))
                    else:
                        logger.warning(
                            "channel %s is not available for station %s in this correlation file",
                            channel_name,
                            station_name,
                        )
            else:
                logger.warning(
                    "stations %s is not available in this correlation file",
                    station_name,
                )

        if primary_event_slice == "all":
            station, channel = station_channel_index[0]
            primary_event_slice = (
                0,
                self.root[station][channel]["corr_values"].shape[0] - 1,
            )

        if secondary_event_slice == "all":
            station, channel = station_channel_index[0]
            secondary_event_slice = (
                0,
                self.root[station][channel]["corr_values"].shape[1] - 1,
            )

        # get corr values
        corr_values_array = np.empty(
            (
                primary_event_slice[1] - primary_event_slice[0] + 1,
                secondary_event_slice[1] - secondary_event_slice[0] + 1,
                len(station_channel_index),
            )
        )

        for index, station_channel in enumerate(station_channel_index):
            station, channel = station_channel

            corr_values_array[:, :, index] = self.root[station][channel]["corr_values"][
                primary_event_slice[0] : primary_event_slice[1] + 1,
                secondary_event_slice[0] : secondary_event_slice[1] + 1,
            ]

        # get lag values
        lag_values_array = np.empty(
            (
                primary_event_slice[1] - primary_event_slice[0] + 1,
                secondary_event_slice[1] - secondary_event_slice[0] + 1,
                len(station_channel_index),
            )
        )

        for index, station_channel in enumerate(station_channel_index):
            station, channel = station_channel

            lag_values_array[:, :, index] = self.root[station][channel]["lag_values"][
                primary_event_slice[0] : primary_event_slice[1] + 1,
                secondary_event_slice[0] : secondary_event_slice[1] + 1,
            ]

        return corr_values_array, lag_values_array, station_channel_index

    def _get_single_event_pair_lag_values(
        self,
        event_index_a: int,
        event_index_b: int,
        stations: Literal["all"] | List[str] = "all",
        channels: Literal["all", "all_horiz", "all_vert"] | List[str] = "all",
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
    ) -> Dict[str, Dict[str, float]]:
        lag_values_dict = {}

        # dynamically get all available station names, or not
        if stations == "all":
            station_names = list(self.root.group_keys())
        else:
            station_names = stations

        for station_name in station_names:
            if station_name in self.root:
                lag_values_dict[station_name] = {}

                # dynamically get all available channel names, or not
                if (
                    channels == "all"
                    or channels == "all_horiz"
                    or channels == "all_vert"
                ):
                    channel_names = []
                    for channel_name, group in self.root[station_name].groups():
                        if channels == "all_horiz" and not channel_name.endswith(
                            tuple(horizontal_channels_postfix)
                        ):
                            continue  # skip iteration if not horizontal channel code
                        elif channels == "all_vert" and not channel_name.endswith(
                            tuple(vertical_channels_postfix)
                        ):
                            continue  # skip iteration if not vertical channel code

                        channel_names.append(channel_name)
                else:
                    channel_names = channels

                for channel_name in channel_names:
                    if channel_name in self.root[station_name]:
                        lag_value = self.root[station_name][channel_name]["lag_values"][
                            event_index_a, event_index_b
                        ]

                        if not np.isnan(lag_value):
                            lag_values_dict[station_name][channel_name] = lag_value
                    else:
                        logger.warning(
                            "channel %s is not available for station %s in this correlation file",
                            channel_name,
                            station_name,
                        )
            else:
                logger.warning(
                    "stations %s is not available in this correlation file",
                    station_name,
                )

        return lag_values_dict

    ### export

    def to_dtcc(
        self,
        outfile: str | Path = "./dt.cc",
        event_id_mapping: Dict[str, str] | Literal["index"] | None = None,
        lower_triangle: bool = True,
        stations: Literal["all"] | List[str] = "all",
        channels: Literal["all", "all_horiz", "all_vert"] | List[str] = "all",
        min_corr_value: float = 0,
        weight_value_type: Literal[
            "corr_squared", "signed_corr_squared"
        ] = "corr_squared",
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
        tempdir: str | Path = "./.dtcc_chunks",
        multiprocessing: bool = False,
        num_processes: int = 1,
    ):

        logger.info(
            f"--- Exporting all differential times for the selected correlation file ---"
        )

        # ensure arg types
        outfile = Path(outfile)
        tempdir = Path(tempdir)

        logger.info(f"               out file: {outfile}")
        logger.info(f"               temp dir: {tempdir}")

        # retrieve metadata from file
        event_indices = [int(index) for index in self.events.keys()]

        # start assigning chunks for processing
        ### NOTE: also necessary for serial processing as not working chunkwise is significantly slower due to continual de-compression of chunk files

        max_event_index = event_indices[-1]

        chunks_to_process = get_chunks_to_process(
            max_event_index, self.chunksize, lower_triangle
        )

        logger.info(f"    # chunks to process: {len(chunks_to_process)}")

        time_A = time.time()

        if not multiprocessing or num_processes == 1:  # serial processing
            for primary_event_slice, secondary_event_slice in chunks_to_process:
                time_a = time.time()

                self._write_dtcc_file(
                    outfile=tempdir
                    / (
                        outfile.stem
                        + f"_{primary_event_slice[0]}_{primary_event_slice[1]}_vs_{secondary_event_slice[0]}_{secondary_event_slice[1]}"
                        + outfile.suffix
                    ),
                    event_id_mapping=event_id_mapping,
                    lower_triangle=lower_triangle,
                    primary_event_slice=primary_event_slice,
                    secondary_event_slice=secondary_event_slice,
                    stations=stations,
                    channels=channels,
                    min_corr_value=min_corr_value,
                    weight_value_type=weight_value_type,
                    horizontal_channels_postfix=horizontal_channels_postfix,
                    vertical_channels_postfix=vertical_channels_postfix,
                )

                time_b = time.time()

                logger.info(
                    f"--- Exported event indices {primary_event_slice[0]} to {primary_event_slice[1]} against {secondary_event_slice[0]} to {secondary_event_slice[1]}, took {time_b - time_a:.3g} s"
                )

        else:  # if multiprocessing

            # logger.info(
            #     f"--- Exported event indices {primary_event_slice[0]} to {primary_event_slice[1]} against {secondary_event_slice[0]} to {secondary_event_slice[1]}, took {time_b - time_a:.3g} s"
            # )

            shared_params = [
                self.filename,
                self.chunksize,
                self.shardsize,
                tempdir,
                outfile,
                event_id_mapping,
                lower_triangle,
                stations,
                channels,
                min_corr_value,
                weight_value_type,
                horizontal_channels_postfix,
                vertical_channels_postfix,
            ]

            # ensure we are forking
            set_start_method("fork", force=True)

            pool = Pool(processes=num_processes)

            pool.starmap(
                _write_dtcc_parallel,
                [
                    (
                        primary_event_slice,
                        secondary_event_slice,
                        *shared_params,
                    )
                    for primary_event_slice, secondary_event_slice in chunks_to_process
                ],
            )

        time_B = time.time()

        logger.info(f"Exported all event indices, total time {time_B - time_A:.3g} s")

        logger.info(f"Collecting partial files")

        time_a = time.time()

        with outfile.open("wb") as wfd:
            for partial_file in sorted(tempdir.glob("*")):
                with partial_file.open("rb") as fd:
                    copyfileobj(fd, wfd)

        time_b = time.time()

        logger.info(
            f"Collected all partial files in {outfile}, took {time_b - time_a:.3g} s"
        )

    def _write_dtcc_file(
        self,
        outfile: Path = "./dt.cc",
        event_id_mapping: Dict[str, str] | Literal["index"] | None = None,
        lower_triangle: bool = True,
        primary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        secondary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        stations: Literal["all"] | List[str] = "all",
        channels: Literal["all", "all_horiz", "all_vert"] | List[str] = "all",
        min_corr_value: float = 0,
        weight_value_type: Literal[
            "corr_squared", "signed_corr_squared"
        ] = "corr_squared",
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
    ):
        # retrieve metadata from file
        events = self.events

        catalog = self.catalog

        event_indices = [int(index) for index in events.keys()]

        # event_id_mapping
        if event_id_mapping == "index":
            key_times = [[k, catalog[v]["origin"]["time"]] for k, v in events.items()]

            key_times.sort(key=lambda x: x[1])

            event_id_mapping = {k_t[0]: i + 1 for i, k_t in enumerate(key_times)}

            self.id_mapping = {events[k]: v for k, v in event_id_mapping.items()}
            """Mapping of event IDS to an index-based (starting at 1), sorted by origin times. Will be None unless an index-based mapping was requested for an export to a dt.cc file."""
            
            self.root.attrs["id_mapping"] = self.id_mapping
        elif event_id_mapping is None:
            event_id_mapping = events

        # make out directory if necessary
        if not outfile.parent.exists():
            outfile.parent.mkdir(exist_ok=True, parents=True)

        # assign which indices to process, slice if necessary
        primary_event_indices = (
            event_indices
            if primary_event_slice == "all"
            else event_indices[primary_event_slice[0] : primary_event_slice[1] + 1]
        )

        secondary_event_indices = (
            event_indices
            if secondary_event_slice == "all"
            else event_indices[secondary_event_slice[0] : secondary_event_slice[1] + 1]
        )

        # get values in chunk
        corr_values_array, lag_values_array, station_channel_index = (
            self._get_chunked_event_pair_corr_and_lag_values(
                primary_event_slice,
                secondary_event_slice,
                stations=stations,
                channels=channels,
                horizontal_channels_postfix=horizontal_channels_postfix,
                vertical_channels_postfix=vertical_channels_postfix,
            )
        )

        # open file and run loop
        with outfile.open("+w") as dt_file:
            for event_index_a in primary_event_indices:
                for event_index_b in (
                    [
                        event_index
                        for event_index in secondary_event_indices
                        if event_index < event_index_a
                    ]
                    if lower_triangle
                    else secondary_event_indices
                ):
                    # retrieve events
                    event_a = catalog[events[str(event_index_a)]]

                    event_b = catalog[events[str(event_index_b)]]

                    # unpack corr values
                    corr_values = {}
                    lag_values = {}

                    for index, station_channel in enumerate(station_channel_index):
                        station, channel = station_channel

                        if station not in corr_values:
                            corr_values[station] = {}

                        if station not in lag_values:
                            lag_values[station] = {}

                        corr_values[station][channel] = corr_values_array[
                            event_index_a - primary_event_slice[0],
                            event_index_b - secondary_event_slice[0],
                            index,
                        ]

                        lag_values[station][channel] = lag_values_array[
                            event_index_a - primary_event_slice[0],
                            event_index_b - secondary_event_slice[0],
                            index,
                        ]

                    # write single pair to file
                    _write_single_event_pair_to_dtcc(
                        dt_file,
                        corr_values,
                        lag_values,
                        event_index_a,
                        event_index_b,
                        event_id_mapping=event_id_mapping,
                        event_a=event_a,
                        event_b=event_b,
                        min_corr_value=min_corr_value,
                        weight_value_type=weight_value_type,
                        horizontal_channels_postfix=horizontal_channels_postfix,
                        vertical_channels_postfix=vertical_channels_postfix,
                    )

        return event_id_mapping

    def to_catalog_file(
        self,
        outfile: str | Path = "./catalog.txt",
        format: Literal["ddsync"] = "ddsync",
    ):
        raise NotImplementedError()
        # ensure type
        outfile = Path(outfile)

        # get metadata
        events = self.root.attrs["events"]

        catalog = self.root.attrs["catalog"]

        with outfile.open("w+") as destination:
            for event_id in events.values():
                event = catalog[event_id]

                print([key for key in event.keys()])

                print(event["origin"])

                # destination.write(line)
