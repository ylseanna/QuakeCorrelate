import logging
import time
from pathlib import Path
from shutil import copyfileobj
from typing import Dict, List, Literal, Tuple, TypedDict

import numpy as np
import pandas as pd
import zarr
from matplotlib.pyplot import close, figure
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event

from ...util import ensure_unique_values, get_chunks_to_process

logger = logging.getLogger(__name__)


class CorrMatrixDict(TypedDict):
    station: str
    channel: str
    corr_matrix: np.ndarray


class CorrelationFile:
    "Correlation File interface"

    def __init__(
        self,
        filename: str | Path = "./correlations.zarr",
        in_memory: bool = False,
        overwrite_file: bool = False,
        chunksize: int = 100,
        shardsize: int = 1000,
    ):
        # args
        self.filename: Path = Path(filename)
        self.chunksize = chunksize
        self.shardsize = shardsize

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

    ### WRITE

    def write_catalog(self, catalog: Catalog):
        """Writes a summary catalog of all the events

        Args:
            catalog (Catalog): _description_

        Note:
            Uses a non-spec XML file internally as JSON can not be read by obspy natively
        """
        keyed_catalog_dict = {}

        for event in catalog:
            event_id = event.resource_id.id

            event_dict = {}

            # add origin information for
            origin = event.preferred_origin() or event.origins[0] or None

            if origin:
                event_dict["origin"] = {
                    "time": origin.time.isoformat(),
                    "lon": origin.longitude,
                    "lat": origin.latitude,
                    "depth": origin.depth,
                }

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

    def write_stations(self, stations: list):
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

    def write_channels(self, channels: list):
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

    def write_events(self, events: list):
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

    def slice_traces_by_station_and_channel(
        self,
        primary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        secondary_event_slice: Literal["all"] | Tuple[int, int] = "all",
        with_corr_file_index: bool = True,
    ) -> (
        Dict[str, Dict[str, List[int]]]
        | Dict[str, Dict[str, Tuple[List[int], List[int]]]]
    ):
        """Method that slices the available traces for all available events by station and channel, listing only events per slice that contain a trace for that station and channel combination.

        Returns:
            Dict[str, Dict[str, List[int]]]: A dictionary keyed by station names (str), where the values are dictionaries keyed by channel names (str), where the values are lists of event indices (str)
        """

        events = self.root.attrs["events"]

        event_indices = [int(index) for index in events.keys()]

        stations = list(self.get_stations(as_df=False).values())

        channels = list(self.get_channels(as_df=False).values())

        catalog = self.root.attrs["catalog"]

        if primary_event_slice == secondary_event_slice:
            slice_dict = {}

            event_indices_slice = (
                event_indices
                if primary_event_slice == "all"
                else event_indices[primary_event_slice[0] : primary_event_slice[1] + 1]
            )

            for event_index in event_indices_slice:
                event_id = events[str(event_index)]

                event_info = catalog[event_id]

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

            return slice_dict
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

            slice_dict = {}

            for event_index in primary_event_indices:
                event_id = events[str(event_index)]

                event_info = catalog[event_id]

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
                event_id = events[str(event_index)]

                event_info = catalog[event_id]

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

    ### GET

    def get_stations(self, as_df=True, include_channels: bool = False):
        stations = self.root.attrs["stations"]

        if as_df:
            stations_df = pd.DataFrame.from_records(
                [(index, name) for index, name in stations.items()],
                columns=["index", "name"],
                index="index",
            )

            return stations_df
        else:
            return stations

    def get_channels(self, as_df=True):
        channels = self.root.attrs["channels"]

        if as_df:
            channels_df = pd.DataFrame.from_records(
                [(index, name) for index, name in channels.items()],
                columns=["index", "name"],
                index="index",
            )

            return channels_df
        else:
            return channels

    def get_events(self, as_df: bool = True):
        events = self.root.attrs["events"]

        if as_df:
            events_df = pd.DataFrame.from_records(
                [(index, name) for index, name in events.items()],
                columns=["index", "name"],
                index="index",
            )

            return events_df
        else:
            return events

    def get_correlation_matrix(
        self,
        stations: Literal["all"] | List[str] = "all",
        channels: Literal["all", "all_horiz", "all_vert"] | List[str] = "all",
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
    ) -> np.ndarray | List[CorrMatrixDict]:
        corr_matrix_list = []

        if stations == "all":
            stations = list(self.root.group_keys())

        for station_name in stations:
            if station_name in self.root:
                if (
                    channels == "all"
                    or channels == "all_horiz"
                    or channels == "all_vert"
                ):
                    for channel_name, group in self.root[station_name].groups():
                        if channels == "all_horiz" and not channel_name.endswith(
                            tuple(horizontal_channels_postfix)
                        ):
                            continue  # skip iteration if not horizontal channel code
                        elif channels == "all_vert" and not channel_name.endswith(
                            tuple(vertical_channels_postfix)
                        ):
                            continue  # skip iteration if not vertical channel code

                        corr_matrix_list.append(
                            {
                                "station": station_name,
                                "channel": channel_name,
                                "corr_matrix": group["corr_values"][:],
                            }
                        )
                else:
                    for channel_name in channels:
                        if channel_name in self.root[station_name]:
                            corr_matrix_list.append(
                                {
                                    "station": station_name,
                                    "channel": channel_name,
                                    "corr_matrix": self.root[station_name][
                                        channel_name
                                    ]["corr_values"][:],
                                }
                            )
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

        if len(corr_matrix_list) > 1:
            return corr_matrix_list
        elif len(corr_matrix_list) == 1:
            return corr_matrix_list[0]["corr_matrix"]
        else:
            logger.warning(
                "No correlation matrices with these search parameters are available in this correlation file"
            )

    def get_single_event_pair_correlation_value(
        self,
        event_index_a: int,
        event_index_b: int,
        stations: Literal["all"] | List[str] = "all",
        channels: Literal["all", "all_horiz", "all_vert"] | List[str] = "all",
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
    ) -> Dict[str, Dict[str, float]]:
        correlation_dict = {}

        # dynamically get all available station names, or not
        if stations == "all":
            station_names = list(self.root.group_keys())
        else:
            station_names = stations

        for station_name in station_names:
            if station_name in self.root:
                correlation_dict[station_name] = {}

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
                        corr_value = self.root[station_name][channel_name][
                            "corr_values"
                        ][event_index_a, event_index_b]

                        if not np.isnan(corr_value):
                            correlation_dict[station_name][channel_name] = corr_value
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

        return correlation_dict

    def get_chunked_event_pair_corr_and_lag_values(
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

    def get_single_event_pair_lag_values(
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

    ### PLOT

    def plot_by_station_and_channel(
        self,
        stations: Literal["all"] | List[str] = "all",
        channels: Literal["all"] | List[str] = "all",
        out_dir: str | Path = "correlation_plots",
        out_filename: str | Path = "correlation.png",
        include_empty_values: bool = True,
    ):
        # ensure pathlib objects
        out_dir = Path(out_dir)
        out_filename = Path(out_filename)

        out_dir.mkdir(exist_ok=True, parents=True)

        # inputs
        slices = self.slice_by_station_and_channel()

        if stations != "all":
            raise NotImplementedError(
                "Whoops, I gave you the option but I still need to write this code"
            )
        elif stations == "all":
            append_station_and_channel_name = True
        elif len(stations) > 1:
            raise NotImplementedError(
                "Whoops, I gave you the option but I still need to write this code"
            )
        elif len(stations) == 1:
            append_station_and_channel_name = False

        if channels != "all":
            raise NotImplementedError(
                "Whoops, I gave you the option but I still need to write this code"
            )

        for slice in slices:
            event_indices = (
                sorted(
                    [event[0] for event in slice["events"] + slice["excluded_events"]]
                )
                if include_empty_values
                else [event[0] for event in slice["events"]]
            )

            trace_indices = [
                (
                    [trace[0] for trace in slice["traces"] if trace[1] == event_id][0]
                    if [trace[0] for trace in slice["traces"] if trace[1] == event_id]
                    else np.nan
                )
                for event_id in event_indices
            ]

            correlations = self.get_correlations(
                as_df=False,
                where=f"primary_trace_id IN ({str([trace[0] for trace in slice['traces']])[1:-1]})",
            )

            correlation_dict = {
                str(correlation_outer[1]): {
                    str(correlation_inner[2]): (
                        correlation_inner[3],
                        correlation_inner[4],
                    )
                    for correlation_inner in correlations
                    if correlation_inner[1] == correlation_outer[1]
                }
                for correlation_outer in correlations
            }

            if include_empty_values:
                correllation_matrix = np.empty((len(trace_indices), len(trace_indices)))
                lag_matrix = np.empty((len(trace_indices), len(trace_indices)))

                correllation_matrix[:] = np.nan
                lag_matrix[:] = np.nan

                for i, primary_trace_id in enumerate(trace_indices):
                    for j, secondary_trace_id in enumerate(trace_indices):

                        if not np.isnan(primary_trace_id) and not np.isnan(
                            secondary_trace_id
                        ):
                            correllation_matrix[i, j], lag_matrix[i, j] = (
                                correlation_dict[str(primary_trace_id)][
                                    str(secondary_trace_id)
                                ]
                            )

            else:
                correllation_matrix = np.zeros((len(trace_indices), len(trace_indices)))
                lag_matrix = np.zeros((len(trace_indices), len(trace_indices)))

                for i, primary_trace_id in enumerate(trace_indices):
                    for j, secondary_trace_id in enumerate(trace_indices):

                        correllation_matrix[i, j], lag_matrix[i, j] = correlation_dict[
                            str(primary_trace_id)
                        ][str(secondary_trace_id)]

            fig = figure(figsize=(7, 7), layout="constrained")

            ax = fig.add_subplot()

            ax.xaxis.tick_top()
            ax.xaxis.set_label_position("top")

            crmat = ax.imshow(correllation_matrix, vmin=0, vmax=1, cmap="viridis")

            cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
            cbar.ax.set_ylabel("correlation value")

            ax.set_title(
                f"correlations, station: {slice['station'][1]}, channel: {slice['channel'][1]}"
            )

            ax.set_xlabel("event index")
            ax.set_ylabel("event index")

            if append_station_and_channel_name:
                fig.savefig(
                    out_dir
                    / (
                        out_filename.stem
                        + f"_{slice['station'][1]}_{slice['channel'][1]}"
                        + out_filename.suffix
                    )
                )
            else:
                fig.savefig(out_dir / out_filename, bbox_inches="tight")

            close()

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
        events = self.root.attrs["events"]

        event_indices = [int(index) for index in events.keys()]

        # start assigning chunks for processing
        ### NOTE: also necessary for serial processing as not working chunkwise is significantly slower due to continual de-compression of chunk files

        max_event_index = event_indices[-1]

        chunks_to_process = get_chunks_to_process(
            max_event_index, self.chunksize, lower_triangle
        )

        logger.info(f"    # chunks to process: {len(chunks_to_process)}")

        time_A = time.time()

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
        events = self.root.attrs["events"]

        catalog = self.root.attrs["catalog"]

        event_indices = [int(index) for index in events.keys()]

        # event_id_mapping
        if event_id_mapping == "index":
            event_id_mapping = {k: k for k in events.keys()}
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
            self.get_chunked_event_pair_corr_and_lag_values(
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
                    self._write_single_event_pair_to_dtcc(
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

        if event_id_mapping == "index":
            return events
        elif isinstance(event_id_mapping, dict):
            return event_id_mapping

    def _write_single_event_pair_to_dtcc(
        self,
        dt_file,
        corr_values,
        lag_values,
        event_index_a: str,
        event_index_b: str,
        event_id_mapping: Dict[str, str],
        event_a: Event,
        event_b: Event,
        min_corr_value: float = 0,
        weight_value_type: Literal[
            "corr_squared", "signed_corr_squared"
        ] = "corr_squared",
        horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
        vertical_channels_postfix: Tuple[str] = ("Z", "3"),
    ):
        single_pair_text_block = ""

        for station_name in corr_values.keys():
            # start with p phase differential times
            p_phase_channels = [
                channel_name
                for channel_name in corr_values[station_name].keys()
                if channel_name.endswith(vertical_channels_postfix)
            ]
            if p_phase_channels:
                if len(p_phase_channels) > 1:
                    if (
                        sum(
                            [
                                not np.isnan(
                                    corr_values[station_name][p_phase_channel_name]
                                )
                                for p_phase_channel_name in p_phase_channels
                            ]
                        )
                        > 1
                    ):
                        logger.warning(
                            f"more than 1 vertical phase channel found: {p_phase_channels} for event pair {event_index_a} {event_index_b}, you may want to check"
                        )

                for channel_name in p_phase_channels:
                    # corr and lag values
                    corr_value = corr_values[station_name][channel_name]

                    # stop writing if correlation value is not enough
                    if abs(corr_value) < min_corr_value:
                        continue

                    lag_value = lag_values[station_name][channel_name]

                    if np.isnan(corr_value) or np.isnan(lag_value):
                        continue

                    # catalog differential times
                    event_a_pick_time = UTCDateTime(
                        [
                            pick
                            for pick in event_a["picks"]
                            if pick["station_name"] == station_name
                            and pick["channel_name"] == channel_name
                            and pick["phase_hint"] == "P"
                        ][0]["time"]
                    )

                    event_b_pick_time = UTCDateTime(
                        [
                            pick
                            for pick in event_b["picks"]
                            if pick["station_name"] == station_name
                            and pick["channel_name"] == channel_name
                            and pick["phase_hint"] == "P"
                        ][0]["time"]
                    )

                    catalog_differential_time = (
                        event_b_pick_time - UTCDateTime(event_b["origin"]["time"])
                    ) - (event_a_pick_time - UTCDateTime(event_a["origin"]["time"]))

                    # weight value
                    if weight_value_type == "corr_squared":
                        weight_value = corr_value * corr_value
                    elif weight_value_type == "signed_corr_squared":
                        weight_value = abs(corr_value) * corr_value

                    #
                    single_pair_text_block += f"{station_name:11s} {catalog_differential_time + lag_value:7.4f} {weight_value:5.2f} P\n"

            s_phase_channels = [
                channel_name
                for channel_name in corr_values[station_name].keys()
                if channel_name.endswith(horizontal_channels_postfix)
            ]

            if s_phase_channels:
                if len(s_phase_channels) > 1:
                    if (
                        sum(
                            [
                                not np.isnan(
                                    corr_values[station_name][s_phase_channel_name]
                                )
                                for s_phase_channel_name in s_phase_channels
                            ]
                        )
                        > 1
                    ):
                        logger.warning(
                            f"more than 1 horizontal phase channel found: {s_phase_channels} for event pair {event_index_a} {event_index_b}, just so you know"
                        )

                else:
                    for channel_name in s_phase_channels:
                        # corr and lag values
                        corr_value = corr_values[station_name][channel_name]

                        # stop writing if correlation value is not enough
                        if abs(corr_value) < min_corr_value:
                            continue

                        lag_value = lag_values[station_name][channel_name]

                        if np.isnan(corr_value) or np.isnan(lag_value):
                            continue

                        # catalog differential times
                        event_a_pick_time = UTCDateTime(
                            [
                                pick
                                for pick in event_a["picks"]
                                if pick["station_name"] == station_name
                                and pick["channel_name"] == channel_name
                                and pick["phase_hint"] == "S"
                            ][0]["time"]
                        )

                        event_b_pick_time = UTCDateTime(
                            [
                                pick
                                for pick in event_b["picks"]
                                if pick["station_name"] == station_name
                                and pick["channel_name"] == channel_name
                                and pick["phase_hint"] == "S"
                            ][0]["time"]
                        )

                        catalog_differential_time = (
                            event_b_pick_time - UTCDateTime(event_b["origin"]["time"])
                        ) - (event_a_pick_time - UTCDateTime(event_a["origin"]["time"]))

                        # weight value
                        if weight_value_type == "corr_squared":
                            weight_value = corr_value * corr_value
                        elif weight_value_type == "signed_corr_squared":
                            weight_value = abs(corr_value) * corr_value

                        #
                        single_pair_text_block += f"{station_name:11s} {catalog_differential_time + lag_value:7.4f} {weight_value:5.2f} S\n"

        # write header
        if len(single_pair_text_block.splitlines()) > 0:
            single_pair_text_block = (
                f"# {event_id_mapping[str(event_index_a)]} {event_id_mapping[str(event_index_b)]} 0.0\n"
                + single_pair_text_block
            )

        # flush text block
        if len(single_pair_text_block.split("\n")) > 1:
            dt_file.write(single_pair_text_block)

    def _plot_single_corr_matrix(self, corr_matrix):
        pass
