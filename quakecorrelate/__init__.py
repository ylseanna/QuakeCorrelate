__version__ = "0.1.0"
__author__ = "Ylse Anna de Vries <yad2@hi.is>, Gabriele Benedetti <gab28@hi.is>"

from dotenv import load_dotenv

from . import signals  # noqa: F401
from .core.interface import ncc  # noqa: F401
from .functions import cross_correlate_seismic_events  # noqa: F401

load_dotenv()


