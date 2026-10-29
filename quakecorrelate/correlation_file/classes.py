from pathlib import Path
from typing import Dict, List

import zarr
from obspy import Catalog
from zarr import Array

from ..util import ensure_unique_values

class ChannelGroup(zarr.Group):
    """Wrapper class for :code:`zarr.Group` (`doc <https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group>`_)  to add functionality for the :code:`.zarr`-file structure used by QuakeCorrelate.

    Comprises of a group for each channel that has been correlated for. It contains arrays for the correlation values and lag values, if the computation has run.
    """

    def __init__(self, GroupInstance: zarr.Group):
        super(ChannelGroup, self).__init__(GroupInstance._async_group)
        self.channel: str = str(Path(self.name).stem).replace("/", "")
        """The channel ID of the group"""

        # make attributes for available arrays
        self.corr_values: Array = self["corr_values"] if "corr_values" in self else None
        """The :code:`.zarr`-array instance containing the computed correlation values."""
        self.lag_values: Array = self["lag_values"] if "lag_values" in self else None
        """The :code:`.zarr`-array instance containing the computed lag values, in seconds."""


class StationGroup(zarr.Group):
    """Wrapper class for :code:`zarr.Group` (`doc <https://zarr.readthedocs.io/en/stable/api/zarr/group/#zarr.Group>`_) to add functionality for the :code:`.zarr`-file structure used by QuakeCorrelate.

    Comprises of a group for each station containing groups for each channel that has been correlated for.
    """

    def __init__(self, GroupInstance: zarr.Group):
        super(StationGroup, self).__init__(GroupInstance._async_group)
        self.channels: List[str] = [channel_name for channel_name, _ in self.members()]
        """List of channel IDs listed for the station ID associated with this group"""
        self.station: str = str(self.name).replace("/", "")
        """The station ID of the group"""

    def __getitem__(self, path):
        return ChannelGroup(super().__getitem__(path))

    def __iter__(self):
        for _, channel_group in self.members():
            yield ChannelGroup(channel_group)


class CorrelationFileMetadata:
    """Wrapper class for a number of metadata parameters for the Correlation File object.

    They alias properties set in the attributes table of the .zarr file"""

    def __init__(self, zarr_file_attributes: zarr.Group.attrs):

        self.attrs = zarr_file_attributes

        # get attributes
        if "catalog" in self.attrs:
            self.catalog: Dict | None = self.attrs["catalog"]
            """Dictionary of event catalog entries, including picks, as stored in the metadata and used for processing and exporting, keyed by event ID"""
        else:
            self.catalog: Dict | None = None

        if "stations" in self.attrs:
            self.stations: Dict | None = self.attrs["stations"]
            """Dictionary of stations as stored in the metadata and used for processing and exporting, keyed by station index."""
        else:
            self.stations: Dict | None = None

        if "channels" in self.attrs:
            self.channels: Dict | None = self.attrs["channels"]
            """Dictionary of stations as stored in the metadata and used for processing and exporting, keyed by station index."""
        else:
            self.channels: Dict | None = None

        if "events" in self.attrs:
            self.events: Dict | None = self.attrs["events"]
            """Dictionary of event IDs as stored in the metadata and used for processing and exporting, keyed by index in the, e.g, correlation and lag matrices."""
        else:
            self.events: Dict | None = None

        if "id_mapping" in self.attrs:
            self.id_mapping: Dict | None = self.attrs["id_mapping"]
            """Mapping of event IDs to an index-based (starting at 1), sorted by origin times. Will be :code:`None` unless an index-based mapping was requested for an export to a dt.cc file."""
        else:
            self.id_mapping: Dict | None = None

    def write_catalog(self, catalog: Catalog):
        """Writes an Obspy catalogue to the metadata of the correlation file.

        Must include at least one Origin and the
        relevant Picks for the streams that are correlated

        Arguments:
            catalog: Obspy :code:`Catalog` object that is to be written to the metadata. Must include at least one :code:`Origin` and the :code:`Pick` objects for the streams to be correlated.
        """

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

        self.attrs["catalog"] = keyed_catalog_dict

        # make attribute
        self.catalog = self.attrs["catalog"]

    def write_stations(self, stations: List[str]):
        """Writes a list of station IDs to the metadata of the correlation file.

        Must include at least all the station IDs that are to be correlated for.

        Arguments:
            stations: List of station IDs (e.g. :code:`STA`, :code:`STAT`) that are to be written to the metadata. Must include all the station IDs that are to be correlated for.
        """

        # ensure sorting
        stations.sort()

        # insert stations into attributes
        if "stations" not in self.attrs:  # if not old values exist just insert
            keyed_station_dict = ensure_unique_values(
                {str(index): str(station) for index, station in enumerate(stations)}
            )
        else:  # combine both dictionaries, with new keys
            keyed_station_dict_old = self.attrs["stations"]

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
        self.attrs["stations"] = keyed_station_dict

        # make attribute
        self.stations = self.attrs["stations"]

    def write_channels(self, channels: List[str]):
        """Writes a list of channel IDs to the metadata of the correlation file.

        Must include at least all the channel IDs that are to be correlated for.

        Arguments:
            stations: List of channel IDs (e.g. :code:`HHZ`, :code:`EHN`) that are to be written to the metadata. Must include all the channel IDs that are to be correlated for.
        """
        # ensure sorting
        channels.sort()

        # insert channels into attributes
        if "channels" not in self.attrs:  # if not old values exist just insert
            keyed_channel_dict = ensure_unique_values(
                {str(index): str(channel) for index, channel in enumerate(channels)}
            )
        else:  # combine both dictionaries, with new keys
            keyed_channel_dict_old = self.attrs["channels"]

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
        self.attrs["channels"] = keyed_channel_dict

        # link to attribute
        self.channels = self.attrs["channels"]

    def write_events(self, events: List[str]):
        """Writes a list of event IDs.

        Must include at least all the event IDs that are to be correlated for. These IDs must correspond to the various :code`Event.resource_id.id()` for the Catalog object embedded in the metadata.
        Will assigns internal integer indices to the events that correspond to the index in the, e.g., correlation and lag matrices.

        Arguments:
            events: List of events IDs (e.g. :code:`20170415_test_event`) that are to be written to the metadata. Must include all the event IDs that are to be correlated for.
        """
        # ensure sorting
        events.sort()

        # insert events into attributes
        if "events" not in self.attrs:  # if not old values exist just insert
            keyed_event_dict = ensure_unique_values(
                {str(index): str(event) for index, event in enumerate(events)}
            )
        else:  # combine both dictionaries, with new keys
            keyed_event_dict_old = self.attrs["events"]

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
        self.attrs["events"] = keyed_event_dict

        # link to attribute
        self.events = self.attrs["events"]
