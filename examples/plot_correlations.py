from cccpy.cccpy.core.io.correlation_file_sqlite import CorrelationFile

correlation_file = CorrelationFile(filename="correlation.sqlite", seismic_data=True)

correlation_file.plot_by_station_and_channel(include_empty_values=True)
