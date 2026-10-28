__version__ = "0.1.0"
__author__ = "Ylse Anna de Vries <yad2@hi.is>, Gabriele Benedetti <gab28@hi.is>"

from dotenv import load_dotenv

from .functions import cross_correlate_seismic_events  # noqa: F401
from .correlation_file import CorrelationFile  # noqa: F401
from .correlator.correlator import Correlator  # noqa: F401
from .interface.julia import setup_julia

load_dotenv()


