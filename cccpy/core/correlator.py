from .io.correlation_file import CorrelationFile
from pathlib import Path


class Correlator:
    def __init__(self, corr_file: str | Path = Path("correlation.sqlite")):
        corr_file = Path(corr_file)

        self.correlation_file = CorrelationFile(filename=corr_file)

