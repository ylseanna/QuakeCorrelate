import logging
import time
from pathlib import Path
from typing import Dict, List, Literal, Tuple

from numpy import isnan
from obspy import UTCDateTime
from obspy.core.event import Event

logger = logging.getLogger(__name__)


def _write_single_event_pair_to_dtcc(
    dt_file,
    corr_values,
    lag_values,
    event_index_a: str,
    event_index_b: str,
    event_id_mapping: Dict[str, str],
    event_a: Event,
    event_b: Event,
    min_corr_value: float = 0,
    weight_value_type: Literal["corr_squared", "signed_corr_squared"] = "corr_squared",
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
                            not isnan(corr_values[station_name][p_phase_channel_name])
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

                if isnan(corr_value) or isnan(lag_value):
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
                single_pair_text_block += f"{station_name:5s} {catalog_differential_time + lag_value:2.4f} {weight_value:1.4f} P\n"

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
                            not isnan(corr_values[station_name][s_phase_channel_name])
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

                    if isnan(corr_value) or isnan(lag_value):
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
                    single_pair_text_block += f"{station_name:5s} {catalog_differential_time + lag_value:2.4f} {weight_value:1.4f} S\n"

    # write header
    if len(single_pair_text_block.splitlines()) > 0:
        single_pair_text_block = (
            f"# {event_id_mapping[str(event_index_a)]} {event_id_mapping[str(event_index_b)]} 0.0\n"
            + single_pair_text_block
        )

    # flush text block
    if len(single_pair_text_block.split("\n")) > 1:
        dt_file.write(single_pair_text_block)


def _write_dtcc_parallel(
    primary_event_slice,
    secondary_event_slice,
    filename: str | Path = "./correlations.zarr",
    chunksize: int = 100,
    shardsize: int = 1000,
    tempdir: str | Path = "./.dtcc_chunks",
    outfile: Path = "./dt.cc",
    event_id_mapping: Dict[str, str] | Literal["index"] | None = None,
    lower_triangle: bool = True,
    stations: Literal["all"] | List[str] = "all",
    channels: Literal["all", "all_horiz", "all_vert"] | List[str] = "all",
    min_corr_value: float = 0,
    weight_value_type: Literal["corr_squared", "signed_corr_squared"] = "corr_squared",
    horizontal_channels_postfix: Tuple[str] = ("E", "N", "1", "2"),
    vertical_channels_postfix: Tuple[str] = ("Z", "3"),
):
    from .correlation_file import CorrelationFile
    
    filename = Path(filename)
    tempdir = Path(tempdir)
    outfile = Path(outfile)

    correlation_file = CorrelationFile(
        filename=filename,
        in_memory=False,
        overwrite_file=False,
        chunksize=chunksize,
        shardsize=shardsize,
    )

    time_a = time.time()

    correlation_file._write_dtcc_file(
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
