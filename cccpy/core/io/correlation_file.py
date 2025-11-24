import logging
import sqlite3
from pathlib import Path
from typing import List, Literal, Tuple, TypedDict

import numpy as np
import pandas as pd
from matplotlib.pyplot import figure, close


class CorrelationSlice(TypedDict):
    station: str | Tuple[int, str]
    channel: str | Tuple[int, str]
    events: List[str] | List[Tuple[int, str]]
    excluded_events: List[str] | List[Tuple[int, str]]
    traces: List[str] | List[Tuple[int, str]]


class CorrelationFile:
    def __init__(
        self,
        filename: str | Path | None,
        in_memory: bool = False,
        seismic_data: bool = False,
        overwrite_file: bool = False,
    ):
        # args
        self.filename: Path = Path(filename)
        self.is_seismic: bool = seismic_data

        # connect to database
        if in_memory:
            self.connection = sqlite3.connect(":memory:")
        else:
            if not filename:
                raise ValueError(
                    "Filename has to be set if the database is not in-memory"
                )

            if overwrite_file:
                if self.filename.exists():
                    self.filename.unlink()

            self.connection = sqlite3.connect(filename)

        self.cursor = self.connection.cursor()

        # create tables
        self.cursor.execute(
            """--sql
            CREATE TABLE IF NOT EXISTS correlations (
                correlation_id INTEGER PRIMARY KEY,
                primary_trace_id INT,
                secondary_trace_id INT,
                correlation_value FLOAT,
                lag FLOAT,
                FOREIGN KEY (primary_trace_id) REFERENCES traces(trace_id),
                FOREIGN KEY (secondary_trace_id) REFERENCES traces(trace_id)
            );
        """
        )

        if not self.is_seismic:
            self.cursor.executescript(
                """--sql
            BEGIN;
            CREATE TABLE IF NOT EXISTS traces (
                trace_id INTEGER PRIMARY KEY
            );
            COMMIT;
        """
            )
        else:
            self.cursor.executescript(
                """--sql
            BEGIN;
            CREATE TABLE IF NOT EXISTS traces (
                trace_id INTEGER PRIMARY KEY,
                event_id INT,
                station_id INT,
                channel_id INT,
                FOREIGN KEY (event_id) REFERENCES events (event_id),
                FOREIGN KEY (station_id) REFERENCES stations (station_id),
                FOREIGN KEY (channel_id) REFERENCES channels (channel_id),
                CONSTRAINT unq UNIQUE (event_id, station_id, channel_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY,
                external_id TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS stations (
                station_id INTEGER PRIMARY KEY,
                station_name TEXT UNIQUE
            );
            CREATE TABLE IF NOT EXISTS channels (
                channel_id INTEGER PRIMARY KEY,
                channel_name TEXT UNIQUE
            );
            COMMIT;
        """
            )

    def write_stations(self, stations: list):
        if not self.is_seismic:
            raise NotImplementedError(
                "This correlation file is not configured for seismic data"
            )

        # ensure sorting
        stations.sort()

        # make into insert variables
        vars = [(station,) for station in stations]

        # insert
        self.cursor.executemany(
            "INSERT INTO stations ('station_name') VALUES (?) ON CONFLICT DO NOTHING",
            vars,
        )

        if self.cursor.rowcount == 0:
            logging.warning(
                "No new stations were added, they had the same id as previously supplied"
            )
        elif self.cursor.rowcount < len(vars):
            logging.warning(
                f"{len(vars) - self.cursor.rowcount} stations weren't added, they had the same id as previously supplied"
            )

        self.connection.commit()

    def write_channels(self, channels: list):
        if not self.is_seismic:
            raise NotImplementedError(
                "This correlation file is not configured for seismic data"
            )

        # ensure sorting
        channels.sort()

        # make into insert variables
        vars = [(channel,) for channel in channels]

        # insert
        self.cursor.executemany(
            "INSERT INTO channels ('channel_name') VALUES (?) ON CONFLICT DO NOTHING",
            vars,
        )

        if self.cursor.rowcount == 0:
            logging.warning(
                "No new channels were added, they had the same id as previously supplied"
            )
        elif self.cursor.rowcount < len(vars):
            logging.warning(
                f"{len(vars) - self.cursor.rowcount} channels weren't added, they had the same id as previously supplied"
            )

        self.connection.commit()

    def write_streams(self, stream_dict: dict, external_catalog: bool = False):
        """Function to add the stream pointers to the correlation file.

        Will modify both the events table (each event_id) and the traces table (each individual trace in the dictionary).

        Args:
            stream_dict (Dict[str, Stream]):
                A dictionary containing event streams keyed by their event id (string)
            external_catalog (bool):
                Option which determines whether event_ids are taken from the stream dict (no event metadata) or an external catalog has already been inserted into the table
        """

        if not self.is_seismic:
            raise NotImplementedError(
                "This correlation file is not configured for seismic data"
            )

        ### add access event_ids to add to events table

        event_ids = list(stream_dict.keys())

        # ensure sorting
        event_ids.sort()

        # make into insert variables
        vars = [(event,) for event in event_ids]

        # insert events
        self.cursor.executemany(
            "INSERT INTO events ('external_id') VALUES (?) ON CONFLICT DO NOTHING",
            vars,
        )

        if self.cursor.rowcount == 0:
            logging.warning(
                "No new events were added, they had the same id as previously supplied"
            )
        elif self.cursor.rowcount < len(vars):
            logging.warning(
                f"{len(vars) - self.cursor.rowcount} events weren't added, they had the same id as previously supplied"
            )

        self.connection.commit()

        ### insert traces (NOTE: foreign keys have to be properly referenced!!!)

        # request all stations with their primary ids
        stations = [
            request for request in self.cursor.execute("SELECT * FROM stations")
        ]

        # request all channel ids with their primary ids
        channels = [
            request for request in self.cursor.execute("SELECT * FROM channels")
        ]

        # request all event ids with their primary ids
        events = [request for request in self.cursor.execute("SELECT * FROM events")]

        # find corresponding indices for trace metadata and save as parameters for database
        vars = []

        for event_id, stream in stream_dict.items():
            event_index = [event for event in events if event[1] == event_id][0][0]

            for trace in stream:
                try:
                    station_index = [
                        station
                        for station in stations
                        if station[1] == trace.stats.station
                    ][0][0]
                except:
                    logging.warning(
                        "Trace's station not present in correlation file, skipping"
                    )
                    continue

                try:
                    channel_index = [
                        channel
                        for channel in channels
                        if channel[1] == trace.stats.channel
                    ][0][0]
                except:
                    logging.warning(
                        "Trace's channel not present in correlation file, skipping"
                    )
                    continue

                vars.append((event_index, station_index, channel_index))

        # insert traces
        self.cursor.executemany(
            "INSERT INTO traces ('event_id', 'station_id', 'channel_id') VALUES (?, ?, ?) ON CONFLICT DO NOTHING",
            vars,
        )

        if self.cursor.rowcount == 0:
            logging.warning(
                "No new traces were added, they had the same event, station and channel ids as previously supplied"
            )
        elif self.cursor.rowcount < len(vars):
            logging.warning(
                f"{len(vars) - self.cursor.rowcount} traces weren't added, they had the same event, station and channel ids  as previously supplied"
            )

        self.connection.commit()

    def write_correlation_matrix(
        self,
        corr_matrix: np.ndarray,
        lag_matrix: np.ndarray,
        column_trace_ids: List[int],
        row_trace_ids: List[int],
    ):

        if corr_matrix.shape != (len(column_trace_ids), len(row_trace_ids)):
            raise ValueError(
                "Correlation matrix shape and column and row trace ids lengths do not match"
            )

        primary_trace_ids, secondary_trace_ids = np.meshgrid(
            column_trace_ids, row_trace_ids
        )

        vars = [
            (
                int(primary_trace_id),
                int(secondary_trace_id),
                float(corr_value),
                float(lag_value),
            )
            for primary_trace_id, secondary_trace_id, corr_value, lag_value in zip(
                primary_trace_ids.reshape(-1),
                secondary_trace_ids.reshape(-1),
                corr_matrix.reshape(-1),
                lag_matrix.reshape(-1),
            )
        ]

        self.cursor.executemany(
            "INSERT INTO correlations ('primary_trace_id', 'secondary_trace_id', 'correlation_value', 'lag') VALUES (?, ?, ?, ?) ON CONFLICT DO NOTHING",
            vars,
        )

        self.connection.commit()

    def slice_by_station_and_channel(
        self, with_corr_file_index: bool = True
    ) -> List[CorrelationSlice]:
        """Function to get a list of dicts containing events divided by station and channel.
        This is useful for cross correlation of seismic traces.

        Args:
            with_corr_file_index (bool):
            whether the station, channel and events are returned with the correlation file id. This can be kept internally to remove some later lookups

        Raises:
            NotImplementedError: If the correlation file is not configured for seismic data.

        Returns:
            List[CorrelationSlice]: List of dictionaries structured as {station: str, channel: str, events: List[str]}
        """

        if not self.is_seismic:
            raise NotImplementedError(
                "This correlation file is not configured for seismic data"
            )

        # metadata entries in correlation file
        stations = self.get_stations(as_df=False)
        channels = self.get_channels(as_df=False)
        events = self.get_events(as_df=False)

        # Correlation slice list to return
        slice_list = []

        for station_id, station in stations:
            traces = [
                request
                for request in self.cursor.execute(
                    f"SELECT * FROM traces WHERE station_id = {station_id}"
                )
            ]

            channels_per_station = np.unique([trace[3] for trace in traces])

            for channel_id in channels_per_station:
                traces_per_channel = [
                    trace for trace in traces if trace[3] == channel_id
                ]
                if with_corr_file_index:
                    slice_list.append(
                        {
                            "station": (station_id, station),
                            "channel": [
                                channel_tuple
                                for channel_tuple in channels
                                if channel_tuple[0] == channel_id
                            ][0],
                            "events": [
                                event_tuple
                                for event_tuple in events
                                if event_tuple[0]
                                in [trace[1] for trace in traces_per_channel]
                            ],
                            "excluded_events": [
                                event_tuple
                                for event_tuple in events
                                if event_tuple[0]
                                not in [trace[1] for trace in traces_per_channel]
                            ],
                            "traces": [trace for trace in traces_per_channel],
                        }
                    )

        return slice_list

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

        if not self.is_seismic:
            raise NotImplementedError(
                "This correlation file is not configured for seismic data"
            )

        channels = [
            request for request in self.cursor.execute("SELECT * FROM channels")
        ]

        if as_df:
            channels_df = pd.DataFrame.from_records(channels, columns=["index", "name"])

            return channels_df
        else:
            return channels

    def get_events(self, as_df=True, get_traces=False):
        if not self.is_seismic:
            raise NotImplementedError(
                "This correlation file is not configured for seismic data"
            )

        events = [request for request in self.cursor.execute("SELECT * FROM events")]

        if not get_traces:
            if as_df:
                events_df = pd.DataFrame.from_records(
                    events, columns=["index", "event_id"]
                )

                return events_df
            else:
                return events

        event_array = []
        for event_id, event in events:
            event_dict = {}

            event_dict["event"] = (event_id, event)

            traces = [
                request
                for request in self.cursor.execute(
                    f"SELECT * FROM traces WHERE event_id = {event_id}"
                )
            ]

            event_dict["traces"] = traces

            event_array.append(event_dict)

        if as_df:
            raise NotImplementedError
        else:
            return event_array

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
