<p align="center">
  <img src="https://github.com/ylseanna/QuakeCorrelate/blob/main/static/quakecorrelatelogo_mediumres.png" alt="QuakeCorrelate Icon"/>
</p>
<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0">
    <img src="https://img.shields.io/badge/License-GPLv3-blue.svg" />
  </a>
</p>

# QuakeCorrelate

QuakeCorrelate is a Python package to run large-scale cross correlation of phase arrivals of seismic events. The resulting cross correlation and differential 
time values can then be used in further processing, notably for methods such as double difference relocation, repeater analysis or other 
waveform based analysis of large seismic catalogues.

QuakeCorrelate is an open-source package written in Python. It embeds the Julia package [Squiggles.jl](https://www.github.com/gbene/Squiggles.jl) to run its
GPU-based cross correlations. Currently it only support GPU-based cross correlation, on CUDA and Metal GPU architectures.

QuakeCorrelate is not yet released on PyPI, but can be installed from the source, see also in the [documentation](https://quakecorrelate.readthedocs.io) for a [guide](https://quakecorrelate.readthedocs.io/installation).

The package is still under active development.

## Features

- Cross correlation with standard *Obspy* inputs
- `.zarr` file efficient and parallelisable cross correlation matrix storage with a custom interface.
- `dt.cc` file output for running [HypoDD](https://github.com/fwaldhauser/HypoDD), [Growclust](https://github.com/dttrugman/GrowClust3D.jl) or [DDsync](https://github.com/eliasrh/DDSync).

## Contact

The developers can be contacted at:

<a href="&#109;&#97;&#105;&#108;&#116;&#111;&#58;&#121;&#97;&#100;&#50;&#64;&#104;&#105;&#46;&#105;&#115;">&#89;&#108;&#115;&#101;&#32;&#65;&#110;&#110;&#97;&#32;&#100;&#101;&#32;&#86;&#114;&#105;&#101;&#115;</a> (<a href="&#109;&#97;&#105;&#108;&#116;&#111;&#58;&#121;&#97;&#100;&#50;&#64;&#104;&#105;&#46;&#105;&#115;">&#121;&#97;&#100;&#50;&#64;&#104;&#105;&#46;&#105;&#115;</a>)

If you find a bug or wish to request a feature, you can go to the issues section on github: https://github.com/ylseanna/QuakeCorrelate/issues.

## Declarations

### Funding

The software was developed as part of a PhD project funded by grant nr. 2410397-051 from the Icelandic Centre for Research (RANNÍS).
