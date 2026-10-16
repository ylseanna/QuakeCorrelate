import sys
from pathlib import Path

extensions = [
    'sphinx.ext.autodoc',
]

sys.path.insert(0, str(Path('..', 'quakecorrelate').resolve()))