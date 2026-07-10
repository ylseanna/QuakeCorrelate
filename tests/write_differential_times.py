from quakecorrelate import CorrelationFile

### assume default correlation file output from correlate_seismic_event example

correlation_file = CorrelationFile()

correlation_file.to_dtcc(event_id_mapping="index")