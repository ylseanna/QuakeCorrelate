import logging
from pathlib import Path
from typing import List, Literal, Tuple, TypedDict

import numpy as np
import pandas as pd
import zarr
from matplotlib.pyplot import close, figure
from obspy import Catalog
from zarr.core.group import GroupMetadata
from zarr.storage import LocalStore

from ...util import ensure_unique_values


class CorrelationSlice(TypedDict):
    station: str | Tuple[int, str]
    channel: str | Tuple[int, str]
    events: List[str] | List[Tuple[int, str]]
    excluded_events: List[str] | List[Tuple[int, str]]
    traces: List[str] | List[Tuple[int, str]]


class SeismicCorrelationFile:
    def __init__(
        self,
        filename: str | Path | None,
        in_memory: bool = False,
        overwrite_file: bool = False,
    ):
        # args
        self.filename: Path = Path(filename)

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

    # def write_streams(self, stream_dict: dict, external_catalog: bool = False):
    #     """Function to add the stream pointers to the correlation file.

    #     Will modify both the events table (each event_id) and the traces table (each individual trace in the dictionary).

    #     Args:
    #         stream_dict (Dict[str, Stream]):
    #             A dictionary containing event streams keyed by their event id (string)
    #         external_catalog (bool):
    #             Option which determines whether event_ids are taken from the stream dict (no event metadata) or an external catalog has already been inserted into the table
    #     """

    #     if not self.is_seismic:
    #         raise NotImplementedError(
    #             "This correlation file is not configured for seismic data"
    #         )

    #     ### add access event_ids to add to events table

    #     event_ids = list(stream_dict.keys())

    #     # ensure sorting
    #     event_ids.sort()

    #     # make into insert variables
    #     vars = [(event,) for event in event_ids]

    #     # insert events
    #     self.cursor.executemany(
    #         "INSERT INTO events ('external_id') VALUES (?) ON CONFLICT DO NOTHING",
    #         vars,
    #     )

    #     if self.cursor.rowcount == 0:
    #         logging.warning(
    #             "No new events were added, they had the same id as previously supplied"
    #         )
    #     elif self.cursor.rowcount < len(vars):
    #         logging.warning(
    #             f"{len(vars) - self.cursor.rowcount} events weren't added, they had the same id as previously supplied"
    #         )

    #     self.connection.commit()

    #     ### insert traces (NOTE: foreign keys have to be properly referenced!!!)

    #     # request all stations with their primary ids
    #     stations = [
    #         request for request in self.cursor.execute("SELECT * FROM stations")
    #     ]

    #     # request all channel ids with their primary ids
    #     channels = [
    #         request for request in self.cursor.execute("SELECT * FROM channels")
    #     ]

    #     # request all event ids with their primary ids
    #     events = [request for request in self.cursor.execute("SELECT * FROM events")]

    #     # find corresponding indices for trace metadata and save as parameters for database
    #     vars = []

    #     for event_id, stream in stream_dict.items():
    #         event_index = [event for event in events if event[1] == event_id][0][0]

    #         for trace in stream:
    #             try:
    #                 station_index = [
    #                     station
    #                     for station in stations
    #                     if station[1] == trace.stats.station
    #                 ][0][0]
    #             except:
    #                 logging.warning(
    #                     "Trace's station not present in correlation file, skipping"
    #                 )
    #                 continue

    #             try:
    #                 channel_index = [
    #                     channel
    #                     for channel in channels
    #                     if channel[1] == trace.stats.channel
    #                 ][0][0]
    #             except:
    #                 logging.warning(
    #                     "Trace's channel not present in correlation file, skipping"
    #                 )
    #                 continue

    #             vars.append((event_index, station_index, channel_index))

    #     # insert traces
    #     self.cursor.executemany(
    #         "INSERT INTO traces ('event_id', 'station_id', 'channel_id') VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
    #         vars,
    #     )

    #     if self.cursor.rowcount == 0:
    #         logging.warning(
    #             "No new traces were added, they had the same event, station and channel ids as previously supplied"
    #         )
    #     elif self.cursor.rowcount < len(vars):
    #         logging.warning(
    #             f"{len(vars) - self.cursor.rowcount} traces weren't added, they had the same event, station and channel ids  as previously supplied"
    #         )

    #     self.connection.commit()

    def write_correlation_matrix(
        self,
        corr_matrix: np.ndarray,
        lag_matrix: np.ndarray,
        station: str,
        channel: str,
    ):
        self.root.create_array(
            f"{station}/{channel}/corr_values",
            data=corr_matrix,
            chunks=(100, 100),
            shards=(1000, 1000),
        )
        self.root.create_array(
            f"{station}/{channel}/lag_values",
            data=lag_matrix,
            chunks=(100, 100),
            shards=(1000, 1000),
        )

    def slice_by_station_and_channel(
        self, with_corr_file_index: bool = True
    ) -> List[CorrelationSlice]:

        slice_dict = {}

        events = self.root.attrs["events"]

        catalog = self.root.attrs["catalog"]

        for event_index, event_id in events.items():
            event_info = catalog[event_id]

            for pick in event_info["picks"]:
                if pick["station_name"] not in slice_dict:
                    slice_dict[pick["station_name"]] = {}

                if pick["channel_name"] not in slice_dict[pick["station_name"]]:
                    slice_dict[pick["station_name"]][pick["channel_name"]] = []

                slice_dict[pick["station_name"]][pick["channel_name"]].append(
                    int(event_index)
                )

        # Correlation slice list to return
        # slice_list = []

        # for station_id, station in stations:
        #     traces = [
        #         request
        #         for request in self.cursor.execute(
        #             f"SELECT * FROM traces WHERE station_id = {station_id}"
        #         )
        #     ]

        #     channels_per_station = np.unique([trace[3] for trace in traces])

        #     for channel_id in channels_per_station:
        #         traces_per_channel = [
        #             trace for trace in traces if trace[3] == channel_id
        #         ]
        #         if with_corr_file_index:
        #             slice_list.append(
        #                 {
        #                     "station": (station_id, station),
        #                     "channel": [
        #                         channel_tuple
        #                         for channel_tuple in channels
        #                         if channel_tuple[0] == channel_id
        #                     ][0],
        #                     "events": [
        #                         event_tuple
        #                         for event_tuple in events
        #                         if event_tuple[0]
        #                         in [trace[1] for trace in traces_per_channel]
        #                     ],
        #                     "excluded_events": [
        #                         event_tuple
        #                         for event_tuple in events
        #                         if event_tuple[0]
        #                         not in [trace[1] for trace in traces_per_channel]
        #                     ],
        #                     "traces": [trace for trace in traces_per_channel],
        #                 }
        #             )

        return slice_dict

    def get_stations(self, as_df=True, include_channels: bool = False):

        if not self.is_seismic:
            raise NotImplementedError(
                "This correlation file is not configured for seismic data"
            )

        # load stations
        stations = [
            request for request in self.cursor.execute("SELECT * FROM stations")
        ]

        if not include_channels:  # exit and return now
            if as_df:
                stations_df = pd.DataFrame.from_records(
                    stations, columns=["index", "name"]
                )

                return stations_df
            else:
                return stations

        # load channels
        channels = self.get_channels(as_df=False)

        station_array = []
        for station_id, station in stations:
            station_dict = {}

            station_dict["station"] = (station_id, station)

            traces = [
                request
                for request in self.cursor.execute(
                    f"SELECT * FROM traces WHERE station_id = {station_id}"
                )
            ]

            channels_per_station = np.unique([trace[3] for trace in traces])

            station_dict["channels"] = [
                channel for channel in channels if channel[0] in channels_per_station
            ]

            station_array.append(station_dict)

        if as_df:
            raise NotImplementedError
        else:
            return station_array

    def get_channels(self, as_df=True):
        channels = [
            request for request in self.cursor.execute("SELECT * FROM channels")
        ]

        if as_df:
            channels_df = pd.DataFrame.from_records(channels, columns=["index", "name"])

            return channels_df
        else:
            return channels

    def get_events(self, as_df: bool = True, expand_from_catalog: bool = False):
        events = self.root.attrs["events"]

        if as_df:
            raise NotImplementedError
        else:
            return events

    def get_correlations(self, as_df: bool = True, where: str | None = None):
        correlations = [
            request
            for request in self.cursor.execute(
                f"SELECT * FROM correlations {'WHERE ' + where if where else ''}"
            )
        ]

        if as_df:
            correlations_df = pd.DataFrame.from_records(
                correlations,
                columns=[
                    "index",
                    "primary_trace_id",
                    "secondary_trace_id",
                    "correlation_value",
                    "lag",
                ],
            )

            return correlations_df
        else:
            return correlations

    def get_cross_channel_correlations(
        self,
        out_dir: str | Path = "correlation_plots",
        out_filename: str | Path = "cross_channel_correlation.png",
    ):
        raise NotImplementedError

        # ensure pathlib objects
        out_dir = Path(out_dir)
        out_filename = Path(out_filename)

        events = self.get_events(as_df=False, get_traces=True)

        correllation_matrix = np.empty((len(events), len(events)))
        correllation_matrix[:] = np.nan

        for i, primary_event in enumerate(events):
            for j, secondary_event in enumerate(events):
                print(i, j)

                self.cursor.execute(
                    "SELECT *  FROM correlations WHERE primary_trace_id IN ({}) INTERSECT SELECT *  FROM correlations WHERE secondary_trace_id IN ({})".format(
                        str([trace[0] for trace in primary_event["traces"]])[1:-1],
                        str([trace[0] for trace in secondary_event["traces"]])[1:-1],
                    )
                )

                correlations = self.cursor.fetchall()

                if correlations:
                    mean_corr_val = np.mean(
                        [correlation[4] for correlation in correlations]
                    )

                    correllation_matrix[i, j] = mean_corr_val

        fig = figure(figsize=(7, 7), layout="constrained")

        ax = fig.add_subplot()

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position("top")

        crmat = ax.imshow(correllation_matrix, vmin=0, vmax=1, cmap="viridis")

        cbar = fig.colorbar(crmat, ax=ax, shrink=0.6)
        cbar.ax.set_ylabel("correlation value")

        ax.set_title("correlations, cross-channel")

        ax.set_xlabel("event index")
        ax.set_ylabel("event index")

        fig.savefig(out_dir / out_filename, bbox_inches="tight")

        close()

        return correllation_matrix

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

            print(correllation_matrix)

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
