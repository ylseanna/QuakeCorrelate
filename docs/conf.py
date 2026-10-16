# Project information
project = 'QuakeCorrelate'
copyright = '2026, Ylse Anna de Vries, Gabriele Benedetti'
author = 'Ylse Anna de Vries, Gabriele Benedetti'
version = '1.0'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
]

# HTML theme configuration
html_theme = 'python_docs_theme'

### AUTODOC

import sys
from pathlib import Path

sys.path.insert(0, str(Path('..', 'quakecorrelate').resolve()))