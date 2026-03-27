from quakecorrelate.core.io.correlation_file_sqlite import CorrelationFile

correlation_file = CorrelationFile(filename="correlation.sqlite", seismic_data=True)

correlation_file.get_cross_channel_correlations()
