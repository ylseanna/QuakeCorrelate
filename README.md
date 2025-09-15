# CCCpy
Python wrapper for the Julia package CCC (Cuda Cross correlation).

It aims to enable fast, open-source 1D cross correlation for seismic data analysis, e.g. generating differential times for double-difference relocation, or repeater analysis.

## Current Goals
- benchmark to other commonly used waveform cross correlation softwares
- using JuliaCall enable interfacing with obspy-compliant waveform storage formats, to make CCC as widely avaialable as possible
