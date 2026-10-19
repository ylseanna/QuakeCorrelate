import sys
from pathlib import Path


# Project information
project = 'QuakeCorrelate'
copyright = '2026, Ylse Anna de Vries, Gabriele Benedetti'
author = 'Ylse Anna de Vries, Gabriele Benedetti'
version = '1.0'
release = '1.0.0'
sys.path.insert(0, str(Path('..').resolve()))

extensions = [
    'sphinx.ext.autodoc',  # Core library for html generation from docstrings
    'sphinx.ext.autosummary',  # Create neat summary tables
]
autosummary_generate = True  # Turn on sphinx.ext.autosummary

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# HTML theme configuration
html_theme = 'pydata_sphinx_theme'

### AUTODOC

