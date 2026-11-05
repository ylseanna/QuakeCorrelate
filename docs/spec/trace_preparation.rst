--------------------
Trace preparation
--------------------

This page details how traces are prepared in QuakeCorrelate.

.. figure:: /_static/figures/signal_cutout.svg
   :alt: Illustration of how input parameters relate to how traces are prepared in QuakeCorrelate for correlation.

   Structure of how traces are prepared for (cross) correlation in QuakeCorrelate. It shows in particular how the padding is removed for
   the secondary (moving) trace.
   The illustration includes all relevant parameters, such as :code:`prepick`, :code:`postpick`, :code:`shift_len` and :code:`padding`.
   Note that :code:`shift_len` and :code:`padding` are, and should typically be, the same value, but this can be changed using input parameters.