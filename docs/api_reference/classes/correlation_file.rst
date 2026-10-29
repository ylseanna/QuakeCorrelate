^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
quakecorrelate.CorrelationFile
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: quakecorrelate.correlation_file.CorrelationFile
    :members:
    :inherited-members:
    :undoc-members:

==================
Usage examples
==================

~~~~~~~~~~~~~~~~~~~~~~~~~~
Correlation file indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~

:code:`CorrelationFile` objects access the underlying :code:`.zarr`-object containing any correlation outputs through indexing based on the
station ID, channel ID, and array type (:code:`corr_values`, :code:`lag_values`). These will then return the :code:`.zar` array objects, which can be indexed to access specific values, e.g.::

    from quakecorrelate import CorrelationFile

    cf = CorrelationFile()

    cf["STA1"]["HHZ"]["corr_values"]

Should return the :code:`.zar` array object with the calculated correlation values for station :code:`STA1`, channel :code:`HHZ`.
