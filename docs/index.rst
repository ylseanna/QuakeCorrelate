==============================
 QuakeCorrelate Documentation
==============================

.. image:: https://img.shields.io/badge/License-GPLv3-blue.svg
   :alt: license
   :target: https://www.gnu.org/licenses/gpl-3.0

.. <p align="center">
..   <img src="https://github.com/ylseanna/QuakeView/blob/main/resources/icons/128x128.png" alt="QuakeView Icon"/>
.. </p>
.. <p align="center">
..   <a href="https://www.gnu.org/licenses/gpl-3.0">
..     <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" />
..   </a>
..   <a href="https://doi.org/10.5281/zenodo.19372719">
..     <img src="https://zenodo.org/badge/1000148409.svg" alt="DOI">
..   </a>
.. </p>

QuakeCorrelate is a Python package to run large-scale cross correlation of phase arrivals of seismic events. The resulting cross correlation and differential 
time values can then be used in further processing, notably for methods such as double difference relocation, repeater analysis or other 
waveform based analysis of large seismic catalogues.

QuakeCorrelate is an open-source package written in Python. It embeds the Julia package `Squiggles.jl <https://www.github.com/gbene/Squiggles.jl>`_ to run its
GPU-based cross correlations. Currently it only support GPU-based cross correlation, on CUDA and Metal GPU architectures.

QuakeCorrelate is not yet released on PyPI, but can be installed from the repository (https://github.com/QuakeCorrelate), 
see the sections :doc:`usage/quickstart` or :doc:`usage/installation`.

The package is still under active development.

-----------------
Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   quickstart
   installation
   apireference

-----------------
Features
-----------------

- Cross correlation with standard *Obspy* inputs
- :code:`.zarr` file efficient and parallelisable cross correlation matrix storage with a custom interface.
- :code:`dt.cc` file output for running `HypoDD <https://github.com/fwaldhauser/HypoDD>`_, `Growclust <https://github.com/dttrugman/GrowClust3D.jl>`_ or `DDsync <https://github.com/eliasrh/DDSync>`_

-----------------
Contact
-----------------

The developers can be contacted at:

Ylse Anna de Vries <yad2@hi.is>

If you find a bug or wish to request a feature, you can go to the issues section on github: https://github.com/ylseanna/QuakeCorrelate/issues.

---------------
Declarations
---------------

^^^^^^^^^^^
Funding
^^^^^^^^^^^

The software was developed as part of a PhD project funded by grant nr. 2410397-051 from the Icelandic Centre for Research (RANNÍS).
