# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
from datetime import date
import os
import sys

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))

# The master toctree document.
master_doc = "index"

# -- Project information -----------------------------------------------------

project = "QuakeCorrelate"
copyright = "2025-{}, Ylse Anna de Vries, Gabriele Benedetti".format(date.today().year)
author = "Ylse Anna de Vries, Gabriele Benedetti"

# The full version, including alpha/beta/rc tags
# import quakecorrelate

version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------

# html_logo = "images/logo_small.png"

html_context = {
    # ...
    "default_mode": "light"
}

html_static_path = ["_static"]


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.email",
    "sphinx_favicon",
    # "nbsphinx"
]
napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = False
napoleon_preprocess_types = True
# # Add any paths that contain templates here, relative to this directory.
# templates_path = ['_templates']

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "obspy": ("https://docs.obspy.org", None),
    "zarr": ("https://zarr.readthedocs.io", None),
}


email_automode = True


# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

master_doc = "index"
html_theme = "pydata_sphinx_theme"


# # Add any paths that contain custom static files (such as style sheets) here,
# # relative to this directory. They are copied after the builtin static files,
# # so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

html_sidebars = {
    "installation": [],
    "quickstart": [],
}

html_logo = "_static/logos/quakecorrelatelogo_mediumres.png"

favicons = [
    # generic icons compatible with most browsers
    {"href": "favicon/favicon.svg"},
    "favicon/favicon-96x96.png",
    "favicon/favicon.ico",
    {"rel": "shortcut icon", "sizes": "any", "href": "favicon.ico"},
    # chrome specific
    "favicon/web-app-manifest-192x192.png",
    # apple icons
    {"rel": "apple-touch-icon", "href": "favicon/apple-touch-icon.png"},
    # to show it works as well with absolute urls
]

html_css_files = [
    "css/custom.css",
]

html_theme_options = {
    "logo": {
        "text": "QuakeCorrelate",
    },
    "codetextcolor": "red"
}
