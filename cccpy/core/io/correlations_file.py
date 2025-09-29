from pathlib import Path
import sqlite3


class Correlations:
    def __init__(
        self,
        filename: str | Path | None,
        in_memory: bool = False,
        seismic_data: bool = False,
    ):
        self.filename = filename

        # init
        if in_memory:
            self.connection = sqlite3.connect(":memory:")

        else:
            if not filename:
                raise ValueError(
                    "Filename has to be set if the database is not in-memory"
                )

            self.connection = sqlite3.connect(filename)

        self.cursor = self.connection.cursor()

        # create tables

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS correlations (
                correlation_id INT AUTO_INCREMENT PRIMARY KEY,
                primary_trace_id INT,
                secondary_trace_id INT,
                correlation_value FLOAT,
                lag FLOAT,
                FOREIGN KEY (primary_trace_id) REFERENCES traces(trace_id),
                FOREIGN KEY (secondary_trace_id) REFERENCES traces(trace_id)
            );
        """)

        if not seismic_data:
            self.cursor.executescript("""
            BEGIN;
            CREATE TABLE IF NOT EXISTS traces (
                trace_id INT AUTO_INCREMENT PRIMARY KEY
            );
            COMMIT;
        """)
        else:
            self.cursor.executescript("""
            BEGIN;
            CREATE TABLE IF NOT EXISTS traces (
                trace_id INT AUTO_INCREMENT PRIMARY KEY,
                event_id INT,
                station_id INT,
                channel_id INT,
                FOREIGN KEY (event_id) REFERENCES events (event_id),
                FOREIGN KEY (station_id) REFERENCES stations (station_id),
                FOREIGN KEY (channel_id) REFERENCES channels (channel_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id INT AUTO_INCREMENT PRIMARY KEY,
                external_id TEXT
            );
            CREATE TABLE IF NOT EXISTS stations (
                station_id INT AUTO_INCREMENT PRIMARY KEY,
                station_name TEXT
            );
            CREATE TABLE IF NOT EXISTS channels (
                channel_id INT AUTO_INCREMENT PRIMARY KEY,
                channel_name TEXT
            );
            COMMIT;
        """)
